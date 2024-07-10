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
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenGatherOp : public OpConversionPattern<AtenGatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};

    helper.addOperand("self", adaptor.getSelf());
    helper.addOperand("index", adaptor.getIndex());
    helper.addIntAttr("axis", op.getDim());

    return helper.replace();
  }
};

class ConvertAtenIndexTensorHackedTwinOp
    : public OpConversionPattern<AtenIndexTensorHackedTwinOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};

    Value input = adaptor.getSelf();
    auto inputTensorType = input.getType().dyn_cast<RankedTensorType>();
    // Check input is a tensor type.
    if (!inputTensorType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");

    helper.addOperand("self", input);
    helper.addAsMultipleTensorOperands("index_", op.getIndices());

    return helper.replace();
  }
};

class ConvertAten_IndexPutImplOp
    : public OpConversionPattern<Aten_IndexPutImplOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_IndexPutImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};
    helper.addOperand("self", adaptor.getSelf());
    helper.addAsMultipleTensorOperands("index_", adaptor.getIndices());
    helper.addOperand("values", adaptor.getValues());
    helper.addBoolAttr("accumulate", op.getAccumulate());
    helper.addBoolAttr("unsafe", op.getUnsafe());

    return helper.replace();
  }
};

class ConvertAtenConvolutionOp : public OpConversionPattern<AtenConvolutionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};
    helper.addOperand("input", adaptor.getInput());
    helper.addOperand("weight", adaptor.getWeight());
    if (!adaptor.getBias().getType().isa<Torch::NoneType>()) {
      helper.addOperand("bias", adaptor.getBias());
    }

    helper.addListOfIntsAttr("stride", adaptor.getStride());
    helper.addListOfIntsAttr("padding", adaptor.getPadding());
    helper.addListOfIntsAttr("dilation", adaptor.getDilation());
    helper.addListOfIntsAttr("output_padding", adaptor.getOutputPadding());
    helper.addBoolAttr("transposed", op.getTransposed());
    helper.addIntAttr("groups", op.getGroups());

    return helper.replace();
  }
};

class ConvertAtenFakeQuantizePerTensorAffineOp
    : public OpConversionPattern<AtenFakeQuantizePerTensorAffineOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenFakeQuantizePerTensorAffineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};
    helper.addOperand("self", adaptor.getSelf());
    helper.addFloatAttr("scale", op.getScale());
    helper.addIntAttr("zero_point", op.getZeroPoint());
    helper.addIntAttr("quant_min", op.getQuantMin());
    helper.addIntAttr("quant_max", op.getQuantMax());

    return helper.replace();
  }
};

class ConvertAtenFakeQuantizePerTensorAffineTensorQparamsOp
    : public OpConversionPattern<
          AtenFakeQuantizePerTensorAffineTensorQparamsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenFakeQuantizePerTensorAffineTensorQparamsOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};
    helper.addOperand("self", adaptor.getSelf());
    helper.addIntAttr("quant_min", op.getQuantMin());
    helper.addIntAttr("quant_max", op.getQuantMax());

    // scale
    auto scaleOp = op.getScale().getDefiningOp();
    if (!scaleOp)
      return rewriter.notifyMatchFailure(op, "Missing scale operation");
    auto scaleTensor = dyn_cast<torch::Torch::ValueTensorLiteralOp>(scaleOp);
    if (!scaleTensor)
      return rewriter.notifyMatchFailure(
          op, "Scale operation is not ValueTensorLiteralOp");
    auto scaleElements =
        dyn_cast<DenseFPElementsAttr>(scaleTensor.getValueAttr());
    // scale should be a size-1 tensor.
    if (!scaleElements || scaleElements.getNumElements() != 1)
      return rewriter.notifyMatchFailure(op, "Unsupported scale type or size");
    auto scale = (*scaleElements.begin()).convertToDouble();
    helper.addDenseFloatArrayAttr("scale", {scale});

    // zero_point
    auto zeroPointOp = op.getZeroPoint().getDefiningOp();
    int64_t zeroPoint;
    if (!zeroPointOp)
      return rewriter.notifyMatchFailure(op, "Missing zero point operation");
    if (dyn_cast<torch::Torch::AtenZerosOp>(zeroPointOp) ||
        dyn_cast<torch::Torch::AtenZerosLikeOp>(zeroPointOp)) {
      zeroPoint = 0;
    } else {
      auto zeroPointTensor =
          dyn_cast<torch::Torch::ValueTensorLiteralOp>(zeroPointOp);
      if (!zeroPointTensor)
        return rewriter.notifyMatchFailure(
            op, "Zero point operation is not ValueTensorLiteralOp or Zero "
                "operation");
      auto zeroPointElements =
          dyn_cast<DenseIntElementsAttr>(zeroPointTensor.getValueAttr());
      // zero_point should be a size-1 tensor.
      if (!zeroPointElements || zeroPointElements.getNumElements() != 1)
        return rewriter.notifyMatchFailure(
            op, "Unsupported zero point type or size");
      zeroPoint = (*zeroPointElements.begin()).getSExtValue();
    }
    helper.addDenseIntArrayAttr("zero_point", {zeroPoint});

    return helper.replace();
  }
};

class ConvertAtenFakeQuantizePerChannelAffineOp
    : public OpConversionPattern<AtenFakeQuantizePerChannelAffineOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenFakeQuantizePerChannelAffineOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    torch_to_tcp::TorchToTcpCustomOpConversionHelper helper{op, rewriter,
                                                            getTypeConverter()};
    helper.addOperand("self", adaptor.getSelf());
    helper.addIntAttr("axis", op.getAxis());
    helper.addIntAttr("quant_min", op.getQuantMin());
    helper.addIntAttr("quant_max", op.getQuantMax());

    // scale
    auto scaleOp = op.getScale().getDefiningOp();
    if (!scaleOp)
      return rewriter.notifyMatchFailure(op, "Missing scale operation");
    auto scaleTensor = dyn_cast<torch::Torch::ValueTensorLiteralOp>(scaleOp);
    if (!scaleTensor)
      return rewriter.notifyMatchFailure(
          op, "Scale operation is not ValueTensorLiteralOp");
    auto scaleElements =
        dyn_cast<DenseFPElementsAttr>(scaleTensor.getValueAttr());
    if (!scaleElements || scaleElements.getType().getShape().size() != 1)
      return rewriter.notifyMatchFailure(op, "Unsupported scale type or size");
    SmallVector<double> scale;
    for (auto val : scaleElements.getValues<APFloat>())
      scale.push_back(val.convertToDouble());
    helper.addDenseFloatArrayAttr("scale", scale);

    // zero_point
    auto zeroPointOp = op.getZeroPoint().getDefiningOp();
    SmallVector<int64_t> zeroPoint;
    if (!zeroPointOp)
      return rewriter.notifyMatchFailure(op, "Missing zero point operation");
    if (dyn_cast<torch::Torch::AtenZerosOp>(zeroPointOp) ||
        dyn_cast<torch::Torch::AtenZerosLikeOp>(zeroPointOp)) {
      zeroPoint.assign(scale.size(), 0);
    } else {
      auto zeroPointTensor =
          dyn_cast<torch::Torch::ValueTensorLiteralOp>(zeroPointOp);
      if (!zeroPointTensor)
        return rewriter.notifyMatchFailure(
            op, "Zero point operation is not ValueTensorLiteralOp or Zero "
                "operation");
      auto zeroPointElements =
          dyn_cast<DenseIntElementsAttr>(zeroPointTensor.getValueAttr());
      if (!zeroPointElements ||
          zeroPointElements.getType().getShape().size() != 1)
        return rewriter.notifyMatchFailure(
            op, "Unsupported zero point type or size");
      for (auto val : zeroPointElements.getValues<APInt>())
        zeroPoint.push_back(val.getSExtValue());
    }
    helper.addDenseIntArrayAttr("zero_point", zeroPoint);

    return helper.replace();
  }
};

} // namespace

void torch_to_tcp::populateTcpCustomOpPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet) {

#define INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(AtenOp)                           \
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<Convert##AtenOp, AtenOp>(   \
      typeConverter, patterns, target, convertTorchOpsSet)
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(AtenGatherOp);
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(AtenIndexTensorHackedTwinOp);
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(Aten_IndexPutImplOp);
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(AtenFakeQuantizePerTensorAffineOp);
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(
      AtenFakeQuantizePerTensorAffineTensorQparamsOp);
  INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN(AtenFakeQuantizePerChannelAffineOp);
#undef INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN

  // Torch -> TOSA doesn't handle transposed convolutions; map them to
  // TCP custom_op instead.
  auto isTransposedConvOp = [](AtenConvolutionOp op) {
    bool transposed;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return false;
    return transposed;
  };

  // Torch -> TOSA supports only 2D convolutions; map the rest to
  // TCP custom_op instead.
  auto is2dConvOp = [](AtenConvolutionOp op) {
    auto inputTy =
        op.getInput().getType().cast<torch::Torch::ValueTensorType>();
    return inputTy.getSizes().size() == 4;
  };

  // Mark only regular (non-transposed) 2D convolutions as legal (in Torch
  // dialect). i.e. don't convert them to TCP custom_op and leave them in Torch,
  // to be handled by Torch -> TOSA later.
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenConvolutionOp,
                                                   AtenConvolutionOp>(
      typeConverter, patterns, target, convertTorchOpsSet,
      [&](AtenConvolutionOp op) {
        return !isTransposedConvOp(op) && is2dConvOp(op);
      });
}
