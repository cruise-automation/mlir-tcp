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
  using OpConversionPattern<AtenGatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            OpConversionPattern<AtenGatherOp>::getTypeConverter()->convertTypes(
                op->getResultTypes(), resultTypes))) {
      return failure();
    }

    int64_t dimVal;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimVal)))
      return failure();

    auto indexAttr =
        rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(dimVal));

    auto newOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(
        op, resultTypes, ValueRange{adaptor.getSelf(), adaptor.getIndex()},
        indexAttr);
    newOp.setOpName(op->getName().getStringRef());
    return success();
  }
};

class ConvertAtenIndexTensorHackedTwinOp
    : public OpConversionPattern<AtenIndexTensorHackedTwinOp> {
public:
  using OpConversionPattern<AtenIndexTensorHackedTwinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenIndexTensorHackedTwinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            OpConversionPattern<AtenIndexTensorHackedTwinOp>::getTypeConverter()
                ->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }

    SmallVector<Value> tensorOperands;
    Value input = adaptor.getSelf();
    auto inputTensorType = input.getType().dyn_cast<RankedTensorType>();
    // Check input is a tensor type.
    if (!inputTensorType)
      return rewriter.notifyMatchFailure(
          op, "Only tensor types input are currently supported");
    tensorOperands.push_back(input);

    // Deal with torch.prim.ListConstruct of non const value to get the index
    Value indexList = op.getIndices();
    SmallVector<Value> indicesTorchType;
    if (!getListConstructElements(indexList, indicesTorchType))
      return op.emitError(
          "unimplemented: the tensor list is not from list construct");
    SmallVector<Value> indexTensors = getTypeConvertedValues(
        rewriter, op->getLoc(), getTypeConverter(), indicesTorchType);

    tensorOperands.append(indexTensors.begin(), indexTensors.end());

    auto newOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(op, resultTypes,
                                                            tensorOperands);
    newOp.setOpName(op->getName().getStringRef());
    return success();
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
#undef INSERT_ATEN_TO_TCP_CUSTOM_OP_PATTERN
}
