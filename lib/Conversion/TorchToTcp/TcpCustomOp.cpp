//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TcpCustomOpBuilder.h"
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
  using OpConversionPattern<AtenGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TcpCustomOpBuilder builder{op, rewriter, getTypeConverter()};

    builder.addOperand("self", adaptor.getSelf());
    builder.addOperand("index", adaptor.getIndex());
    builder.addIntAttr("axis", op.getDim());

    return builder.replace();
  }
};

class ConvertAtenIndexTensorHackedTwinOp
    : public OpConversionPattern<AtenIndexTensorHackedTwinOp> {
public:
  using OpConversionPattern<AtenIndexTensorHackedTwinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TcpCustomOpBuilder builder{op, rewriter, getTypeConverter()};

    Value input = adaptor.getSelf();
    auto inputTensorType = input.getType().dyn_cast<RankedTensorType>();
    // Check input is a tensor type.
    if (!inputTensorType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");

    builder.addOperand("self", input);
    builder.addAsMultipleTensorOperands("index_", op.getIndices());

    return builder.replace();
  }
};

class ConvertAten_IndexPutImplOp
    : public OpConversionPattern<Aten_IndexPutImplOp> {
public:
  using OpConversionPattern<Aten_IndexPutImplOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Aten_IndexPutImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    TcpCustomOpBuilder builder{op, rewriter, getTypeConverter()};
    builder.addOperand("self", adaptor.getSelf());
    builder.addAsMultipleTensorOperands("index_", adaptor.getIndices());
    builder.addOperand("values", adaptor.getValues());
    builder.addBoolAttr("accumulate", op.getAccumulate());
    builder.addBoolAttr("unsafe", op.getUnsafe());

    return builder.replace();
  }
};

class ConvertAtenConvolutionOp : public OpConversionPattern<AtenConvolutionOp> {
public:
  using OpConversionPattern<AtenConvolutionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TcpCustomOpBuilder builder{op, rewriter, getTypeConverter()};
    builder.addOperand("input", adaptor.getInput());
    builder.addOperand("weight", adaptor.getWeight());
    if (!adaptor.getBias().getType().isa<Torch::NoneType>()) {
      builder.addOperand("bias", adaptor.getBias());
    }

    builder.addListOfIntsAttr("stride", adaptor.getStride());
    builder.addListOfIntsAttr("padding", adaptor.getPadding());
    builder.addListOfIntsAttr("dilation", adaptor.getDilation());
    builder.addListOfIntsAttr("output_padding", adaptor.getOutputPadding());
    builder.addBoolAttr("transposed", op.getTransposed());
    builder.addIntAttr("groups", op.getGroups());

    return builder.replace();
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
#undef INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN

  auto isTransposedConvOp = [](AtenConvolutionOp op) {
    bool transposed;
    if (!matchPattern(op.getTransposed(), m_TorchConstantBool(&transposed)))
      return false;
    return transposed;
  };

  // Only want to convert transposed conv ops, i.e., if its not transposed,
  // its "legal", i.e., will not get converted.
  torch_to_tcp::addPatternIfOpInConvertTorchOpsSet<ConvertAtenConvolutionOp,
                                                   AtenConvolutionOp>(
      typeConverter, patterns, target, convertTorchOpsSet,
      [&](AtenConvolutionOp op) { return !isTransposedConvOp(op); });
}
