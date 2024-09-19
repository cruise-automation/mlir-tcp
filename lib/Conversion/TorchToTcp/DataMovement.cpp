//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcp.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Conversion/TorchToLinalg/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// Adopted from
// https://sourcegraph.com/github.com/llvm/torch-mlir@6b3a7d07c2c76f5e8437ff4e88110899621557b9/-/blob/lib/Conversion/TorchToLinalg/DataMovement.cpp?L42
template <typename OpTy, typename OpAdaptor>
LogicalResult prepareArgumentsForSlicingOp(OpTy op, OpAdaptor adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           SmallVector<Value> &resultShape,
                                           SmallVector<Value> &offsets,
                                           SmallVector<Value> &strides) {
  Location loc = op.getLoc();
  auto input = adaptor.getSelf();
  RankedTensorType inputType =
      input.getType().template cast<RankedTensorType>();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value negone = rewriter.create<arith::ConstantIndexOp>(loc, -1);

  int64_t dim;
  if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
    return op->emitError("unimplemented: dim is not constant");

  int64_t inputRank = inputType.getRank();
  dim = toPositiveDim(dim, inputRank);
  if (!isValidDim(dim, inputRank))
    return rewriter.notifyMatchFailure(op, "dim is statically invalid");

  SmallVector<Value> inputShape = getTensorSizes(rewriter, loc, input);
  Value dimSize = inputShape[dim];

  Value torchTypeStart = op.getStart();
  Value torchTypeEnd = op.getEnd();
  Value builtinTypeStart = adaptor.getStart();
  Value builtinTypeEnd = adaptor.getEnd();

  if (torchTypeStart.getType().isa<OptionalType>() ||
      torchTypeEnd.getType().isa<OptionalType>())
    return rewriter.notifyMatchFailure(op, "unimplemented optional type arg");

  Value stepIndex = castIntToIndex(rewriter, loc, adaptor.getStep());
  Value start = toPositiveValidDim(rewriter, loc, torchTypeStart,
                                   builtinTypeStart, zero, dimSize);

  // We cannot use to positive valid dim as for negative strides we need to
  // clamp to `-1` so that the full tensor bounds are available:
  Value end = builtinTypeEnd;
  if (torchTypeEnd.getType().isa<Torch::NoneType>()) {
    end = dimSize;
  } else {
    end = castIntToIndex(rewriter, loc, end);
    Value endcmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, end, zero);
    Value endadd = rewriter.create<arith::AddIOp>(loc, end, dimSize);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, endadd, end);
    endcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, end,
                                            zero);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, negone, end);
    endcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, end,
                                            dimSize);
    end = rewriter.create<arith::SelectOp>(loc, endcmp, dimSize, end);
  }

  // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
  resultShape = getTensorSizes(rewriter, loc, input);
  Value len = rewriter.create<arith::SubIOp>(loc, end, start);

  // We check the difference between start and end to determine the total size:
  Value stepcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 stepIndex, zero);
  Value stepsign = rewriter.create<arith::SelectOp>(loc, stepcmp, one, negone);
  Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
  resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, stepsign);
  resultSize = rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);

  // Clamp the size to [0, ...]:
  Value szcmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               resultSize, zero);
  resultSize = rewriter.create<arith::SelectOp>(loc, szcmp, zero, resultSize);
  resultShape[dim] = resultSize;

  strides.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);

  offsets[dim] = start;
  strides[dim] = stepIndex;
  return success();
}

// Adopted from
// https://sourcegraph.com/github.com/llvm/torch-mlir@6b3a7d07c2c76f5e8437ff4e88110899621557b9/-/blob/lib/Conversion/TorchToLinalg/DataMovement.cpp?L1552
class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenCatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();

    // Collect all the tensors to be concatenated.
    auto tensorList = op.getTensors();
    SmallVector<Value> tensorsTorchType;
    if (!getListConstructElements(tensorList, tensorsTorchType))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    auto tensors =
        getTypeConvertedValues(rewriter, loc, typeConverter, tensorsTorchType);

    RankedTensorType newResultType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    int rank = newResultType.getRank();
    Value dimValue = op.getDim();
    int64_t dim;
    if (!matchPattern(dimValue, m_TorchConstantInt(&dim)))
      return op.emitError("unimplemented: dim is not constant");
    dim = toPositiveDim(dim, rank);
    if (!isValidDim(dim, rank))
      return rewriter.notifyMatchFailure(op, "dim is statically invalid");

    auto outElemType = newResultType.getElementType();
    for (size_t i = 0; i < tensors.size(); ++i) {
      auto inputType = cast<RankedTensorType>(tensors[i].getType());
      if (inputType.getElementType() != outElemType) {
        tensors[i] = torch_to_linalg::convertTensorToElementType(
            rewriter, loc, tensors[i], outElemType);
      }
    }

    llvm::SmallVector<Value> filteredTensors;
    for (auto tensor : tensors) {
      auto inputType = cast<RankedTensorType>(tensor.getType());
      if (inputType.getDimSize(dim) != 0) {
        filteredTensors.push_back(tensor);
      }
    }

    rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, newResultType, dim,
                                                  filteredTensors);
    return success();
  }
};

// Adopted from
// https://sourcegraph.com/github.com/llvm/torch-mlir@6b3a7d07c2c76f5e8437ff4e88110899621557b9/-/blob/lib/Conversion/TorchToLinalg/DataMovement.cpp?L1516
class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    auto input = adaptor.getSelf();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .cast<RankedTensorType>();

    SmallVector<Value> resultShape;
    SmallVector<Value> offsets;
    SmallVector<Value> strides;
    if (failed(prepareArgumentsForSlicingOp<AtenSliceTensorOp,
                                            AtenSliceTensorOpAdaptor>(
            op, adaptor, rewriter, resultShape, offsets, strides))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tcp::SliceOp>(op, resultType, input, offsets,
                                              resultShape, strides);
    return success();
  }
};

class ConvertAtenGatherOp : public OpConversionPattern<AtenGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getSelf();
    auto indices = adaptor.getIndex();
    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(op->getResult(0).getType())
                                      .template cast<RankedTensorType>();

    int64_t dim = 0;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return op.emitError("dim on torch.gather must be an int constant");
    auto inputType = input.getType().cast<RankedTensorType>();
    dim = Torch::toPositiveDim(dim, inputType.getRank());

    bool sparseGrad = false;
    if (!matchPattern(op.getSparseGrad(), m_TorchConstantBool(&sparseGrad)))
      return op.emitError(
          "sparse_grad on torch.gather must be a bool constant");
    if (sparseGrad)
      return op.emitError("unimplemented: sparse_grad is not supported yet");

    rewriter.replaceOpWithNewOp<tcp::GatherOp>(op, resultType, input, indices,
                                               rewriter.getIndexAttr(dim));
    return success();
  }
};

class ConvertAtenIndexSelectOp : public OpConversionPattern<AtenIndexSelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputRank = inputType.getRank();

    auto indices = adaptor.getIndex();
    auto indicesRank = cast<RankedTensorType>(indices.getType()).getRank();
    // As per the semantics of `torch.index_select` op, the indices is
    // always a 1-D tensor. We enforce that check here as some of the
    // utilities used below work only for that case.
    if (indicesRank != 1)
      return rewriter.notifyMatchFailure(op, "indices need to be 1-D");

    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    int64_t dim = 0;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          op, "dim on torch.index_select must be an int constant");
    dim = Torch::toPositiveDim(dim, inputRank);
    if (!isValidDim(dim, inputRank))
      return rewriter.notifyMatchFailure(
          op, "dim on torch.index_select is statically invalid");

    auto indicesRankBroadcasted = torch_to_tcp::broadcastRank0Dor1DToND(
        rewriter, indices, inputRank, dim);
    auto indicesBroadcasted = torch_to_tcp::broadcastShapeExceptDims(
        rewriter, indicesRankBroadcasted, input,
        llvm::SmallDenseSet<int64_t>{dim});
    rewriter.replaceOpWithNewOp<tcp::GatherOp>(
        op, resultType, input, indicesBroadcasted, rewriter.getIndexAttr(dim));
    return success();
  }
};

class ConvertAtenIndexTensorHackedTwin
    : public OpConversionPattern<AtenIndexTensorHackedTwinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ------- Matching the OP -------
    auto self = adaptor.getSelf();
    auto selfType = cast<RankedTensorType>(self.getType());
    auto indicesList = op.getIndices();
    SmallVector<Value> indices;
    if (!getListConstructElements(indicesList, indices))
      return op.emitError("Failed to match list of indices");

    for(unsigned int i = 0; i < indices.size(); i++) {
      auto ttype = cast<RankedTensorType>(getTypeConverter()->convertType(indices[i].getType()));
      if(ttype.getRank() != selfType.getRank() - i) {
        // Can use tensor.gather instead for this.  But will require that there are some broadcasting to get the shapes to match
        // what is expected
        return failure("Failed to rewrite Tensor_hacked_twin.  Need the element gather for this");
      }
      for(int j = 1; j < ttype.getRank(); j++) {
        if(ttype.getShape()[j] != 1)
          return failure("Expected the axes >=1 to have size 1");
      }
    }

    // ------ Rewriting the OP ---------

    indices = getTypeConvertedValues(rewriter, op.getLoc(), getTypeConverter(),
                                     indices);



    for(unsigned int i = 0; i < indices.size(); i++) {
      auto idx = indices[i];
      auto ttype = cast<RankedTensorType>(idx.getType());
      auto selfType = cast<RankedTensorType>(self.getType());
      SmallVector<int64_t> outShape(selfType.getShape());
      outShape[i] = ttype.getNumElements();
      auto outType = RankedTensorType::get(
          outShape, cast<RankedTensorType>(self.getType()).getElementType());
      
      auto expandedShape = torch_to_tcp::broadcastRankInLeadingDims(rewriter, idx, outShape.size() - ttype.getRank());
      SmallVector<Value> broadcastValues;
      SmallVector<int64_t> broadcastAxes;
      

      for(unsigned int j = 0; j < selfType.getRank(); j++) {
        if(j != i) {
          broadcastAxes.push_back(j);
          broadcastValues.push_back(rewriter.create<tensor::DimOp>(op.getLoc(), self, j));
        }
      }

      auto broadcastedShape = rewriter.create<tcp::BroadcastOp>(
        op.getLoc(), 
        RankedTensorType::get(outShape, ttype.getElementType()),
        expandedShape,
        broadcastValues,
        rewriter.getI64ArrayAttr(broadcastAxes)
        );

        auto gather = rewriter.create<tcp::GatherOp>(
          op.getLoc(), outType, self, broadcastedShape.getResult(), rewriter.getIndexAttr(i)
        );
        self = gather.getResult();
    }

    /*for (unsigned int i = 0; i < indices.size(); i++) {
      auto idx = indices[i];
      int numNonOneAxis = 0;
      auto ttype = cast<RankedTensorType>(idx.getType());
      if(ttype.getRank() != indices.size() - i) {
        // there is a version of this op, where everything comes in as a single dim and then is should instead select the different indicies from each?
        // so the difference would be if it keeps the dim or shrinks it.  But not 100% clear on what the definition of the different semantics are
        return op.emitError("unsure what to do");
      }
      for (int j = 0; j < ttype.getRank(); j++)
        if (ttype.getShape()[j] != 1)
          numNonOneAxis++;
      if (numNonOneAxis > 1)
        return op.emitError(
            "Expected the input shape to have a single non-one axis");
      // convert it to a 1-dim vector
      if (ttype.getRank() != 1) {
        ReassociationIndices reassocIndices;
        for (int j = 0; j < ttype.getRank(); j++)
          reassocIndices.push_back(j);
        SmallVector<ReassociationIndices> ri = {reassocIndices};
        auto reshape =
            rewriter.create<tensor::CollapseShapeOp>(op.getLoc(), idx, ri);
        idx = reshape.getResult();
      }

      SmallVector<int64_t> outShape(
          cast<RankedTensorType>(self.getType()).getShape());
      outShape[i] = ttype.getNumElements();
      auto outType = RankedTensorType::get(
          outShape, cast<RankedTensorType>(self.getType()).getElementType());

      auto gather = rewriter.create<tcp::GatherOp>(
          op.getLoc(), outType, self, idx, rewriter.getIndexAttr(i));
      self = gather.getResult();
    }*/

    // assert(op.getType() == self.getType());

    rewriter.replaceOp(op, self);
    return success();
  }
};

} // namespace

void torch_to_tcp::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet) {
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenCatOp, AtenCatOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenSliceTensorOp,
                                                   AtenSliceTensorOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenGatherOp,
                                                   AtenGatherOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenIndexSelectOp,
                                                   AtenIndexSelectOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<
      ConvertAtenIndexTensorHackedTwin, AtenIndexTensorHackedTwinOp>(
      typeConverter, patterns, target, convertTorchOpsSet);
}
