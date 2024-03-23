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
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

template <typename AtenOpT>
bool checkZerosOnesOpAttributes(AtenOpT op, RankedTensorType outType) {
  // check output type
  if (!outType)
    return false;
  if (!outType.getElementType().isIntOrFloat())
    return false;

  // check default layout
  int64_t memoryLayout;
  if (!op.getLayout().getType().template isa<Torch::NoneType>() &&
      (!matchPattern(op.getLayout(), m_TorchConstantInt(&memoryLayout)) ||
       memoryLayout != 0)) {
    return false;
  }

  // check default pin_memory
  bool pinMemory;
  if (!op.getPinMemory().getType().template isa<Torch::NoneType>() &&
      (!matchPattern(op.getPinMemory(), m_TorchConstantBool(&pinMemory)) ||
       pinMemory)) {
    return false;
  }

  return true;
}

template <typename AtenOpT>
class ConvertAtenBroadcastLikeOps : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;

  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();

    ArrayRef<int64_t> inputShape = inputType.getShape();

    SmallVector<Value> newDimSizes;
    if (!getListConstructElements(op.getSize(), newDimSizes))
      return rewriter.notifyMatchFailure(
          op, "Broadcasted shape must be a list of scalars");

    int64_t newLeadingDims = newDimSizes.size() - inputType.getRank();
    if (newLeadingDims > 0) {
      input = torch_to_tcp::broadcastRankInLeadingDims(rewriter, input,
                                                       newLeadingDims);
    }

    SmallVector<int64_t> axes;
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < static_cast<int64_t>(newDimSizes.size()); ++i) {
      Value newDimSize = newDimSizes[i];

      bool isNewDim = i < newLeadingDims;
      // At the pytorch level it's possible to broadcast to a dynamic dimension;
      // for example: `torch.broadcast_to(x, y.size())` where `y.size()` has
      // dynamic shapes. When broadcasting to dynamic size, matchPattern fails
      // to read an int out of the MLIR Value. In the "usual" case
      // (`torch.broadcast_to(x, (3, 3))`) matchPattern will succeed.
      int64_t staticDimSize;
      bool isDimDynamic =
          !matchPattern(newDimSize, m_TorchConstantInt(&staticDimSize));
      // pytorch defines "passing -1 as the size for a dimension means not
      // changing the size of that dimension." Short circuit if dim is dynamic
      // as staticDimSize won't have a valid value.
      bool isDimSizePreserved = isDimDynamic ? false : staticDimSize == -1;
      // Short circuit if isNewDim to prevent out of bounds access of
      // inputShape.
      bool doesDimChangeShape =
          isDimDynamic || isNewDim
              ? false
              : staticDimSize != inputShape[i - newLeadingDims];

      if (isNewDim || isDimDynamic ||
          (!isDimSizePreserved && doesDimChangeShape)) {
        axes.push_back(i);
        newDimSize = rewriter.create<torch::TorchConversion::ToI64Op>(
            op->getLoc(), newDimSize);
        resultShape.push_back(rewriter.create<arith::IndexCastOp>(
            op->getLoc(), rewriter.getIndexType(), newDimSize));
      }
    }

    RankedTensorType resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()
            ->convertType(op->getResult(0).getType())
            .template cast<RankedTensorType>();

    auto axesAttr = rewriter.getI64ArrayAttr(axes);
    rewriter.replaceOpWithNewOp<tcp::BroadcastOp>(op, resultType, input,
                                                  resultShape, axesAttr);
    return success();
  }
};

class ConvertValueTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType =
        OpConversionPattern<ValueTensorLiteralOp>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    if (auto elements = op.getValueAttr().dyn_cast<DenseIntElementsAttr>()) {
      Type elementType = resultType.getElementType();
      auto denseIntAttr = elements.mapValues(elementType, [&](const APInt &v) {
        return APInt(elementType.getIntOrFloatBitWidth(), v.getSExtValue());
      });
      rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType, denseIntAttr);
      return success();
    }

    rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType,
                                              adaptor.getValue());
    return success();
  }
};

class ConvertAtenSizeIntOp : public OpConversionPattern<AtenSizeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenSizeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value self = adaptor.getSelf();
    auto type = self.getType().cast<RankedTensorType>();
    if (!isa<ConstantIntOp>(op->getOperand(1).getDefiningOp())) {
      return rewriter.notifyMatchFailure(op, "dim must be a constant int");
    }
    auto constIntOp =
        dyn_cast<ConstantIntOp>(op->getOperand(1).getDefiningOp());
    int idxVal = constIntOp.getValueAttr().getValue().getSExtValue();
    if (idxVal < 0 || idxVal >= type.getRank()) {
      return rewriter.notifyMatchFailure(op, "dim must be in range");
    }
    auto idxOp = rewriter.create<arith::ConstantIndexOp>(loc, idxVal);
    auto dimOp = rewriter.create<tensor::DimOp>(loc, self, idxOp);
    auto result =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), dimOp);

    rewriter.replaceOp(op, result);

    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<RankedTensorType>();
    Type outElemTy = outType.getElementType();

    if (!checkZerosOnesOpAttributes<AtenOpT>(op, outType)) {
      return rewriter.notifyMatchFailure(op, "Attribute checks failed");
    }

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output element type");
    }

    Operation *primListOp = op.getSize().getDefiningOp();
    auto listConstruct = dyn_cast<Torch::PrimListConstructOp>(primListOp);
    if (!listConstruct) {
      return rewriter.notifyMatchFailure(
          op, "Size must come from PrimListConstructOp");
    }
    SmallVector<Value> primListVal;
    for (Value value : listConstruct.getElements()) {
      primListVal.push_back(value);
    }

    SmallVector<int64_t> resultShape =
        torch_to_tcp::getShapeFromPrimList(primListVal);
    Value resultOp = torch_to_tcp::broadcast0DOr1DFromShape(
        rewriter, constOp, primListVal, resultShape);

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

template <typename AtenOpT, int fillVal>
class ConvertAtenZerosOnesLikeOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = OpConversionPattern<AtenOpT>::getTypeConverter()
                       ->convertType(op.getType())
                       .template dyn_cast<RankedTensorType>();
    Type outElemTy = outType.getElementType();

    // TODO: Check the attribute for input vtensor
    if (!op.getMemoryFormat().getType().template isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(
          op, "Only default memory format is supported");

    if (!checkZerosOnesOpAttributes<AtenOpT>(op, outType)) {
      return rewriter.notifyMatchFailure(op, "Attribute checks failed");
    }

    Value constOp;
    if (!torch_to_tcp::getConstTensorWithType(rewriter, op, constOp, outElemTy,
                                              fillVal)) {
      return rewriter.notifyMatchFailure(op, "Unsupported output element type");
    }

    Value resultOp = torch_to_tcp::broadcast0DOr1DToNDAndMatchShape(
        rewriter, constOp, input,
        constOp.getType().cast<RankedTensorType>().getElementType());

    rewriter.replaceOp(op, resultOp);

    return success();
  }
};

} // namespace

void torch_to_tcp::populateMiscPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet) {

#define INSERT_ATEN_MISC_OP_PATTERN(AtenOp)                                    \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<Convert##AtenOp, AtenOp>(   \
      typeConverter, patterns, target, convertTorchOpsSet)
  INSERT_ATEN_MISC_OP_PATTERN(ValueTensorLiteralOp);
  INSERT_ATEN_MISC_OP_PATTERN(AtenSizeIntOp);
#undef INSERT_ATEN_MISC_OP_PATTERN

#define INSERT_ATEN_BROADCAST_PATTERN(AtenOp)                                  \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<                            \
      ConvertAtenBroadcastLikeOps<AtenOp>, AtenOp>(typeConverter, patterns,    \
                                                   target, convertTorchOpsSet)
  INSERT_ATEN_BROADCAST_PATTERN(AtenBroadcastToOp);
  INSERT_ATEN_BROADCAST_PATTERN(AtenExpandOp);
#undef INSERT_ATEN_BROADCAST_PATTERN

#define INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenOpPattern, AtenOp, Val)      \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<                            \
      ConvertAtenOpPattern<AtenOp, Val>, AtenOp>(typeConverter, patterns,      \
                                                 target, convertTorchOpsSet)
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesOp, AtenZerosOp, 0);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesOp, AtenOnesOp, 1);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesLikeOp, AtenZerosLikeOp,
                                 0);
  INSERT_ATEN_ZEROS_ONES_PATTERN(ConvertAtenZerosOnesLikeOp, AtenOnesLikeOp, 1);
#undef INSERT_ATEN_ZEROS_ONES_PATTERN
}
