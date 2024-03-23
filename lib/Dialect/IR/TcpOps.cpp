//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

#define GET_OP_CLASSES
#include "mlir-tcp/Dialect/IR/TcpOps.cpp.inc"

namespace mlir::tcp {

LogicalResult ClampOp::verify() {
  auto inputType = getIn().getType().cast<RankedTensorType>();

  if (inputType.getElementType().isa<FloatType>()) {
    if (getMinInt() || getMaxInt())
      return emitOpError("failed to verify that int min / max attributes "
                         "must not be set when input is a float tensor");
    if (!getMinFloat() && !getMaxFloat())
      return emitOpError("failed to verify that at least one of min / max "
                         "attributes must be set");
  }

  if (inputType.getElementType().isa<IntegerType>()) {
    if (getMinFloat() || getMaxFloat())
      return emitOpError("failed to verify that float min / max attributes "
                         "must not be set when input is an int tensor");
    if (!getMinInt() && !getMaxInt())
      return emitOpError("failed to verify that at least one of min / max "
                         "attributes must be set");
  }

  return success();
}

LogicalResult BroadcastOp::verify() {
  auto compareIntAttr = [](Attribute v1, Attribute v2) {
    return v1.cast<IntegerAttr>().getInt() < v2.cast<IntegerAttr>().getInt();
  };

  auto getInt = [](IntegerAttr v) { return v.getInt(); };

  ArrayRef<int64_t> inputShape = getIn().getType().getShape();
  for (auto axis :
       llvm::map_range(getAxes().getAsRange<IntegerAttr>(), getInt)) {
    if (axis >= inputShape.size()) {
      return emitOpError(
          "failed to verify that attribute `axes` are in bounds");
    }

    if (inputShape[axis] != 1) {
      return emitOpError("failed to verify that dimensions listed in attribute "
                         "`axes` have a static size of `1`");
    }
  }

  if (!llvm::is_sorted(getAxes(), compareIntAttr))
    return emitOpError(
        "failed to verify that attribute `axes` must be in increasing order");

  if (std::adjacent_find(std::begin(getAxes()), std::end(getAxes())) !=
      std::end(getAxes()))
    return emitOpError(
        "failed to verify that attribute `axes` must not have any duplicates");

  if (getNewDimSizes().size() != getAxes().size())
    return emitOpError("failed to verify that argument `new_dim_sizes` has the "
                       "same size as the attribute `axes`");

  return success();
}

LogicalResult GroupOp::verify() {
  auto &groupBlock = getBody().front();
  if (groupBlock.empty() ||
      !groupBlock.back().mightHaveTrait<OpTrait::IsTerminator>())
    return emitOpError(
        "failed to verify that op region ends with a terminator");

  auto yieldOp = getBody().front().getTerminator();
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError("failed to verify that the number of yielded values is "
                       "same as the number of results");

  for (unsigned i = 0; i < getNumResults(); ++i) {
    if (yieldOp->getOperand(i).getType() != getResult(i).getType())
      return emitOpError()
             << "failed to verify that the type of operand #" << i
             << " of terminator matches the corresponding result type";
  }

  return success();
}

LogicalResult IsolatedGroupOp::verify() {
  auto &groupBlock = getBody().front();
  if (groupBlock.empty() ||
      !groupBlock.back().mightHaveTrait<OpTrait::IsTerminator>())
    return emitOpError(
        "failed to verify that op region ends with a terminator");

  auto yieldOp = getBody().front().getTerminator();
  if (yieldOp->getNumOperands() != getNumResults())
    return emitOpError("failed to verify that the number of yielded values is "
                       "same as the number of results");

  for (unsigned i = 0; i < getNumResults(); ++i) {
    if (yieldOp->getOperand(i).getType() != getResult(i).getType())
      return emitOpError()
             << "failed to verify that the type of operand #" << i
             << " of terminator matches the corresponding result type";
  }

  return success();
}

OpFoldResult ConstOp::fold(FoldAdaptor) { return getValueAttr(); }

LogicalResult CastOp::verify() {
  auto inputType = getIn().getType().cast<RankedTensorType>();
  auto outputType = getOut().getType().cast<RankedTensorType>();

  if (!inputType.getElementType().isIntOrFloat() ||
      !outputType.getElementType().isIntOrFloat())
    return emitOpError("Cast Op must have integer or floating-point datatype");

  if (inputType.getElementType().isa<FloatType>()) {
    if (getInIntSignedness())
      return emitOpError(
          "in_int_signedness attr should not set when input is FP");
  }

  if (inputType.getElementType().isa<IntegerType>()) {
    if (!getInIntSignedness())
      return emitOpError(
          "in_int_signedness attr must be set when input is INT");
    if (inputType.getElementType().isInteger(1) &&
        getInIntSignedness().value() != Signedness::Signless)
      return emitOpError("in_int_signedness attr must be set to "
                         "Signedness::Signless when input is i1");
  }

  if (outputType.getElementType().isa<FloatType>()) {
    if (getOutIntSignedness())
      return emitOpError(
          "out_int_signedness attr should not set when output is FP");
  }

  if (outputType.getElementType().isa<IntegerType>()) {
    if (!getOutIntSignedness())
      return emitOpError(
          "out_int_signedness attr must be set when output is INT");
    if (outputType.getElementType().isInteger(1) &&
        getOutIntSignedness().value() != Signedness::Signless)
      return emitOpError("out_int_signedness attr must be set to "
                         "Signedness::Signless when output is i1");
  }

  return success();
}

} // namespace mlir::tcp
