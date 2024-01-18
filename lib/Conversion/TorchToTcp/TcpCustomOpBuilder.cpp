//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TcpCustomOpBuilder.h"

#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir::tcp {

void TcpCustomOpBuilder::addOperand(std::string opName, Value value) {
  // Should there be some error checking here that Value has a specific
  // type (for instance a RankedTensorType?)
  operandNames.push_back(opName);
  operands.push_back(value);
}

void TcpCustomOpBuilder::addAsMultipleTensorOperands(std::string opNamePrefix,
                                                     mlir::Value value) {
  mlir::SmallVector<Value> indicesTorchType;
  if (!torch::Torch::getListConstructElements(value, indicesTorchType)) {
    conversionResult = op->emitError(
        "unimplemented: the tensor list is not from list construct");
    return;
  }

  mlir::SmallVector<Value> indexTensors = torch::Torch::getTypeConvertedValues(
      rewriter, op->getLoc(), typeConverter, indicesTorchType);

  for (size_t i = 0; i < indexTensors.size(); ++i) {
    addOperand(opNamePrefix + std::to_string(i), indexTensors[i]);
  }
}

LogicalResult TcpCustomOpBuilder::replace() {
  if (conversionResult.failed()) {
    return conversionResult;
  }

  SmallVector<Type> resultTypes;
  auto result = typeConverter->convertTypes(op->getResultTypes(), resultTypes);
  if (result.failed()) {
    return result;
  }

  // TODO: Is there some better way to convert a
  // SmallVector<std::string> to SmallVector<StringRef> locally here? Note
  // that we cannot store operandNames directly as a SmallVector<StringRef>
  // because the strings passed to addOperand etc. could be computed
  // temporary objects.
  SmallVector<StringRef> operandNameRefs;
  operandNameRefs.append(operandNames.begin(), operandNames.end());

  attrs.push_back(rewriter.getNamedAttr(
      "torch_operand_names", rewriter.getStrArrayAttr(operandNameRefs)));

  auto replOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(op, resultTypes,
                                                           operands, attrs);
  replOp.setOpName(op->getName().getStringRef());
  return success();
}

void TcpCustomOpBuilder::addBoolAttr(std::string attrName, Value value) {
  if (conversionResult.failed())
    return;

  bool constVal;
  if (!matchPattern(value, torch::Torch::m_TorchConstantBool(&constVal))) {
    conversionResult = rewriter.notifyMatchFailure(
        op, std::string("non-const ") + attrName + " unsupported");
    return;
  }
  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getBoolAttr(constVal)));
}

void TcpCustomOpBuilder::addIntAttr(std::string attrName, Value value) {
  if (conversionResult.failed())
    return;

  int64_t constVal;
  if (!matchPattern(value, torch::Torch::m_TorchConstantInt(&constVal))) {
    conversionResult = rewriter.notifyMatchFailure(
        op, std::string("non-const ") + attrName + " unsupported");
    return;
  }
  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getI64IntegerAttr(constVal)));
}

void TcpCustomOpBuilder::addListOfIntsAttr(std::string attrName, Value value) {
  if (conversionResult.failed())
    return;

  SmallVector<int64_t> constVal;
  if (!matchPattern(value, torch::Torch::m_TorchListOfConstantInts(constVal))) {
    conversionResult = rewriter.notifyMatchFailure(
        op, std::string("non-const ") + attrName + " unsupported");
    return;
  }
  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getIndexArrayAttr(constVal)));
}

} // namespace mlir::tcp
