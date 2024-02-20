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
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
// Following function is copied from
// https://sourcegraph.com/github.com/llvm/torch-mlir@main/-/blob/lib/Conversion/TorchToLinalg/DataMovement.cpp?L42
// TODO: Expose this function in a header to reuse.
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

  int64_t step;
  if (!matchPattern(op.getStep(), m_TorchConstantInt(&step))) {
    if (!op.getStep().getType().template isa<Torch::NoneType>())
      return op->emitError("unimplemented: step is not constant");
    step = 1;
  }

  Value start = toPositiveValidDim(rewriter, loc, torchTypeStart,
                                   builtinTypeStart, zero, dimSize);
  Value end = toPositiveValidDim(rewriter, loc, torchTypeEnd, builtinTypeEnd,
                                 dimSize, dimSize);

  // end >= start ? end : start
  Value endSgeStart = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, end, start);
  end = rewriter.create<arith::SelectOp>(loc, endSgeStart, end, start);
  Value stepIndex = rewriter.create<arith::ConstantIndexOp>(loc, step);

  // Slice logic: resultSize = floordiv(end - start + step - 1,  step)
  resultShape = getTensorSizes(rewriter, loc, input);
  Value len = rewriter.create<arith::SubIOp>(loc, end, start);
  Value resultSize = rewriter.create<arith::AddIOp>(loc, len, stepIndex);
  resultSize = rewriter.create<arith::SubIOp>(loc, resultSize, one);
  resultSize = rewriter.create<arith::FloorDivSIOp>(loc, resultSize, stepIndex);
  resultShape[dim] = resultSize;

  strides.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);

  offsets[dim] = start;
  strides[dim] = rewriter.create<arith::MulIOp>(loc, strides[dim], stepIndex);
  return success();
}

class ConvertAtenCatOp : public OpConversionPattern<AtenCatOp> {
public:
  using OpConversionPattern<AtenCatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenCatOp catOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> inputs;
    if (!getListConstructElements(adaptor.getTensors(), inputs))
      return rewriter.notifyMatchFailure(
          catOp, "aten.cat operands must be a list of tensors");

    auto tensorInputs = getTypeConvertedValues(rewriter, catOp->getLoc(),
                                               getTypeConverter(), inputs);

    int64_t dim;
    if (!matchPattern(catOp.getDim(), m_TorchConstantInt(&dim)))
      return rewriter.notifyMatchFailure(
          catOp, "aten.cat dim must be constant integer");

    RankedTensorType resultType = getTypeConverter()
                                      ->convertType(catOp.getType())
                                      .cast<RankedTensorType>();

    dim = toPositiveDim(dim, resultType.getRank());

    rewriter.replaceOpWithNewOp<tcp::ConcatOp>(
        catOp, resultType, tensorInputs,
        rewriter.getIntegerAttr(rewriter.getI64Type(), dim));
    return success();
  }
};

class ConvertAtenSliceTensorOp : public OpConversionPattern<AtenSliceTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenSliceTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        op.getSelf().getType().dyn_cast<torch::Torch::ValueTensorType>();
    auto outputType = op.getType().dyn_cast<torch::Torch::ValueTensorType>();
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "Expected Input/Output to be ValueTensorType");

    Location loc = op.getLoc();
    const TypeConverter *typeConverter = getTypeConverter();

    auto input = adaptor.getSelf();
    RankedTensorType resultType =
        typeConverter->convertType(op->getResult(0).getType())
            .cast<RankedTensorType>();

    SmallVector<Value> resultShape;
    SmallVector<Value> offsets;
    SmallVector<Value> strides;
    if (failed(prepareArgumentsForSlicingOp<AtenSliceTensorOp,
                                            AtenSliceTensorOpAdaptor>(
            op, adaptor, rewriter, resultShape, offsets, strides))) {
      return failure();
    }

    Value result = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, offsets, resultShape, strides);

    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType, result);
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
}
