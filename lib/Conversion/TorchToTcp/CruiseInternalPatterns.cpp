//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"

// Conversion patterns in this file are part of internal we use in cruise_main
// and c/c Code added here can either move outside and upstream or stay (e.g
// case of TorchInternalCrusieOps)

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
template <typename opTy>
class EraseOpConverter : public OpConversionPattern<opTy> {
public:
  using OpConversionPattern<opTy>::OpConversionPattern;
  using OpAdaptor = typename opTy::Adaptor;

  LogicalResult
  matchAndRewrite(opTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertAxisAlignedHardNMS2dOp : public OpConversionPattern<AxisAlignedHardNMS2dOp> {
public:
  using OpConversionPattern<AxisAlignedHardNMS2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AxisAlignedHardNMS2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<AxisAlignedHardNMS2dOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }

    SmallVector<Value> tensorOperands;
    SmallVector<NamedAttribute> namedAttributes;

    // Collect tensor operands as-is
    int idx = 0;
    for (auto operandType : op->getOperandTypes()) {
      if (operandType.isa<ValueTensorType>()) {
        tensorOperands.push_back(adaptor.getOperands()[idx]);
      } else if (!operandType.isa<Torch::IntType>()) {
        op->emitError(
            "Found operands of custom op: " + op->getName().getStringRef() +
            " with unsupported type.");
        return failure();
      }
      idx++;
    }

    // Collect non-tensor operands as named attributes
    Value numClasses = op.getNumClasses();
    Value scoresStride = op.getScoresStride();
    Value scoresSize = op.getScoresSize();
    Value labelsStride = op.getLabelsStride();
    Value boxesStride = op.getBoxesStride();

    auto setNamedAttrs = [&](Value value, StringRef name) {
      int64_t intValue;
      if (matchPattern(value, m_TorchConstantInt(&intValue))) {
        namedAttributes.push_back(rewriter.getNamedAttr(name, rewriter.getI64IntegerAttr(intValue)));
      }
    };

    setNamedAttrs(numClasses, "num_classes");
    setNamedAttrs(scoresStride, "scores_stride");
    setNamedAttrs(scoresSize, "scores_size");
    setNamedAttrs(labelsStride, "labels_stride");
    setNamedAttrs(boxesStride, "boxes_stride");

    auto newOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(
        op, resultTypes, tensorOperands, namedAttributes);
    newOp.setOpName(op->getName().getStringRef());

    return success();
  }
};

// A generic converter for torch OpTy -> tcp::CustomOp with a single config
// operand of ConfigTy type. The converter assumes OpTy have a single operand of
// ConfigTy and the rest are of ValueTensorType type.
// TODO(ahmed.taei): This can also support non-tensor constant operands, the
// converter will match any of them and convert them to attributes.
template <typename OpTy, typename ConfigTy>
class ConvertTorchCustomOpWithSingleConfig : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<OpTy>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }

    Value customTypeOperand;
    SmallVector<Value> tensorOperands;
    int idx = 0;
    for (auto operandType : op.getOperandTypes()) {
      if (operandType.template isa<ConfigTy>()) {
        if (customTypeOperand) {
          op->emitError(
              "Converting custom operation : " + op->getName().getStringRef() +
              " with more than one custom type");
          return failure();
        }
        customTypeOperand = op->getOperands()[idx];
      } else if (operandType.template isa<ValueTensorType>()) {
        tensorOperands.push_back(adaptor.getOperands()[idx]);
      } else {
        op->emitError(
            "Converting custom operation : " + op->getName().getStringRef() +
            " with unsupported custom type ");
        return failure();
      }
      idx++;
    }
    auto newOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(
        op, resultTypes, tensorOperands,
        customTypeOperand.getDefiningOp()->getAttrs());
    // set generic name for executing tensor-rt engine
    if (op->getName().getStringRef() ==
        "torch.tensorrt.execute_engine_variadic")
      newOp.setOpName("tensorrt.execute_engine");
    else
      newOp.setOpName(op->getName().getStringRef());
    return success();
  }
};

class ConvertTorchOclCasprCustomVariadicOp : public OpConversionPattern<OclCasprCustomVariadicOp> {
public:
  using OpConversionPattern<OclCasprCustomVariadicOp>::OpConversionPattern;
  using OpAdaptor = typename OclCasprCustomVariadicOp::Adaptor;

  LogicalResult
  matchAndRewrite(OclCasprCustomVariadicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<OclCasprCustomVariadicOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }

    if (llvm::any_of(op.getOperandTypes(), [](Type oprType) {
          return !oprType.isa<ValueTensorType>();
        })) {
      op->emitError(
          "Converting custom operation : " + op->getName().getStringRef() +
          " with unsupported custom type ");
      return failure();
    }
    auto newOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(
        op, resultTypes, adaptor.getOperands(), op->getAttrs());
    newOp.setOpName(op->getName().getStringRef());

    return success();
  }
};

class ConvertCasprConstIndexOp : public OpConversionPattern<CasprConstIndexOp> {
  using OpConversionPattern<CasprConstIndexOp>::OpConversionPattern;
  using OpAdaptor = typename CasprConstIndexOp::Adaptor;

  LogicalResult
  matchAndRewrite(CasprConstIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
        op, op.getValueAttr().getInt());
    return success();
  }
};

class ConvertCasprShapeTensorDimOp
    : public OpConversionPattern<CasprShapeTensorDimOp> {
  using OpConversionPattern<CasprShapeTensorDimOp>::OpConversionPattern;
  using OpAdaptor = typename CasprShapeTensorDimOp::Adaptor;

  LogicalResult
  matchAndRewrite(CasprShapeTensorDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = dyn_cast<ConstantIntOp>(op.getDimension().getDefiningOp());
    auto axisVal = axis.getValueAttr().getValue().getSExtValue();
    rewriter.replaceOpWithNewOp<tensor::DimOp>(op, adaptor.getInput(), axisVal);
    return success();
  }
};

class ConvertCreateIndexArrayOp : public OpConversionPattern<Torch::CreateIndexArrayOp> {
public:
  using OpConversionPattern<Torch::CreateIndexArrayOp>::OpConversionPattern;
  using OpAdaptor = typename Torch::CreateIndexArrayOp::Adaptor;

  Value createIdxOpFromConst(Value v, Operation * op, PatternRewriter& rewriter) const {
    auto constIntOp = dyn_cast<ConstantIntOp>(v.getDefiningOp());
    auto int_val = constIntOp.getValueAttr().getValue().getSExtValue();
    auto idxOp = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), int_val);
    if(constIntOp->hasOneUse()) {
      rewriter.eraseOp(constIntOp);
    }
    return idxOp;
  }

  LogicalResult
  matchAndRewrite(Torch::CreateIndexArrayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<Torch::CreateIndexArrayOp>::getTypeConverter()
                   ->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tcp::CreateIndexArrayOp>(op, resultTypes,
                                                         adaptor.getOperands());
    return success();
  }
};

class ConvertBindTensorShapeOp : public OpConversionPattern<BindTensorShapeOp> {
public:
  using OpConversionPattern<BindTensorShapeOp>::OpConversionPattern;
  using OpAdaptor = typename BindTensorShapeOp::Adaptor;

  LogicalResult
  matchAndRewrite(BindTensorShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<BindTensorShapeOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tcp::BindTensorShape>(
        op, TypeRange({}), adaptor.getTensor(), adaptor.getShapeArray());
    return success();
  }
};

class ConvertCasprCreateTensorFromIndexOp : public OpConversionPattern<Torch::CasprCreateTensorFromIndexOp> {
public:
  using OpConversionPattern<Torch::CasprCreateTensorFromIndexOp>::OpConversionPattern;
  using OpAdaptor = typename Torch::CasprCreateTensorFromIndexOp::Adaptor;

  LogicalResult
  matchAndRewrite(Torch::CasprCreateTensorFromIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<Torch::CasprCreateTensorFromIndexOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tcp::CasprCreateTensorFromIndexOp>(
        op, resultTypes, adaptor.getIndex());
    return success();
  }
};

class ConvertCasprCreateTensorFromIndexArrayOp : public OpConversionPattern<Torch::CasprCreateTensorFromIndexArrayOp> {
public:
  using OpConversionPattern<Torch::CasprCreateTensorFromIndexArrayOp>::OpConversionPattern;
  using OpAdaptor = typename Torch::CasprCreateTensorFromIndexArrayOp::Adaptor;

  LogicalResult
  matchAndRewrite(Torch::CasprCreateTensorFromIndexArrayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<Torch::CasprCreateTensorFromIndexArrayOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tcp::CasprCreateTensorFromIndexArrayOp>(
        op, resultTypes, adaptor.getShapeArray());
    return success();
  }
};

class ConvertCasprIndexFromTensorOp : public OpConversionPattern<Torch::CasprIndexFromTensorOp> {
public:
  using OpConversionPattern<Torch::CasprIndexFromTensorOp>::OpConversionPattern;
  using OpAdaptor = typename Torch::CasprIndexFromTensorOp::Adaptor;

  LogicalResult
  matchAndRewrite(Torch::CasprIndexFromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<Torch::CasprIndexFromTensorOp>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tcp::CasprIndexFromTensorOp>(
        op, resultTypes, adaptor.getInput());
    return success();
  }
};

class ConvertValueTensorLiteralOp
    : public OpConversionPattern<ValueTensorLiteralOp> {
public:
  using OpConversionPattern<ValueTensorLiteralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ValueTensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType =
        OpConversionPattern<ValueTensorLiteralOp>::getTypeConverter()
            ->convertType(op.getType())
            .cast<RankedTensorType>();

    // check the use of the const op
    // only convert to tcp.const if all uses are for cruise internal ops
    bool useForCruiseInternalOp = true;
    for (Operation *userOp : op.getResult().getUsers()) {
      // TODO: include other cruise intenral ops
      if (!isa<OclCasprCustomVariadicOp>(userOp) &&
          !isa<CasprShapeTensorDimOp>(userOp))
        useForCruiseInternalOp = false;
    }
    if (useForCruiseInternalOp) {
      if (auto elements = op.getValueAttr().dyn_cast<DenseIntElementsAttr>()) {
        Type elementType = resultType.getElementType();
        auto denseIntAttr =
            elements.mapValues(elementType, [&](const APInt &v) {
              return APInt(elementType.getIntOrFloatBitWidth(),
                           v.getSExtValue());
            });
        rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType, denseIntAttr);
        return success();
      }

      rewriter.replaceOpWithNewOp<tcp::ConstOp>(op, resultType,
                                                adaptor.getValue());
      return success();
    }
    return failure();
  }
};
} // namespace


void torch_to_tcp::cruise::populateCruiseInternalPatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<TRTEngineCreateOp>();
  target.addIllegalOp<TRTEngineExecuteVariadicOp>();
  target.addIllegalOp<FusedGatherSubmanifoldConv3DOp>();
  target.addIllegalOp<FusedGatherSubmanifoldConv3DConfigOp>();
  target.addIllegalOp<SparseArrayToDenseMapOp>();
  target.addIllegalOp<SparseArrayToDenseMapConfigOp>();
  target.addIllegalOp<BuildNeighborMapOp>();
  target.addIllegalOp<BuildNeighborMapConfigOp>();
  target.addIllegalOp<OclCasprCustomVariadicOp>();
  target.addIllegalOp<AxisAlignedHardNMS2dOp>();
  target.addIllegalOp<LRDPrefilterPredictionOp>();
  target.addIllegalOp<LRDPrefilterPredictionConfigOp>();
  target.addIllegalOp<OclCasprCustomVariadicOp>();
  target.addIllegalOp<Torch::CreateIndexArrayOp>();
  target.addIllegalOp<BindTensorShapeOp>();
  target.addIllegalOp<CasprShapeTensorDimOp>();
  target.addIllegalOp<CasprConstIndexOp>();
  target.addIllegalOp<Torch::CasprCreateTensorFromIndexOp>();
  target.addIllegalOp<Torch::CasprIndexFromTensorOp>();

  patterns.add<ConvertAxisAlignedHardNMS2dOp>(typeConverter, context);
  patterns.add<ConvertCreateIndexArrayOp>(typeConverter, context);
  patterns.add<ConvertCasprCreateTensorFromIndexOp>(typeConverter, context);
  patterns.add<ConvertCasprCreateTensorFromIndexArrayOp>(typeConverter, context);
  patterns.add<ConvertCasprIndexFromTensorOp>(typeConverter, context);
  patterns.add<ConvertBindTensorShapeOp>(typeConverter, context);
  patterns.add<ConvertCasprShapeTensorDimOp>(typeConverter, context);
  patterns.add<ConvertCasprConstIndexOp>(typeConverter, context);
  patterns.add<ConvertValueTensorLiteralOp>(typeConverter, context);

  patterns.add<ConvertTorchOclCasprCustomVariadicOp>(typeConverter, context);

  patterns.add<ConvertTorchCustomOpWithSingleConfig<TRTEngineExecuteVariadicOp,
                                                    TRTEngineType>,
               ConvertTorchCustomOpWithSingleConfig<
                   FusedGatherSubmanifoldConv3DOp,
                   FusedGatherSubmanifoldConv3DConfigType>,
               ConvertTorchCustomOpWithSingleConfig<
                   SparseArrayToDenseMapOp, SparseArrayToDenseMapConfigType>,
               ConvertTorchCustomOpWithSingleConfig<BuildNeighborMapOp,
                                                    BuildNeighborMapConfigType>,
               ConvertTorchCustomOpWithSingleConfig<LRDPrefilterPredictionOp,
                                                    LRDPrefilterPredictionConfigType>,
               EraseOpConverter<TRTEngineCreateOp>,
               EraseOpConverter<FusedGatherSubmanifoldConv3DConfigOp>,
               EraseOpConverter<BuildNeighborMapConfigOp>,
               EraseOpConverter<SparseArrayToDenseMapConfigOp>,
               EraseOpConverter<LRDPrefilterPredictionConfigOp>>(typeConverter,
                                                                context);
}
