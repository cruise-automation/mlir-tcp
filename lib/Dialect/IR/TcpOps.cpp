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

LogicalResult ConstOp::verify() {
  if (getValueAttr().getType() != getType())
    return emitOpError("can not be used to cast types");
  return success();
}

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

LogicalResult GatherOp::verify() {
  auto inputTensor = cast<RankedTensorType>(getInput().getType());
  auto indicesTensor = cast<RankedTensorType>(getIndices().getType());
  int64_t gatherDim = getDimAttr().getValue().getSExtValue();

  if (inputTensor.getRank() != indicesTensor.getRank())
    return emitOpError(
        "requires that the input tensor and indices are the same rank");

  for (int i = 0; i < inputTensor.getRank(); i++) {
    if (inputTensor.getShape()[i] < indicesTensor.getShape()[i] &&
        !(inputTensor.getShape()[i] == ShapedType::kDynamic ||
          indicesTensor.getShape()[i] == ShapedType::kDynamic ||
          i == gatherDim)) {
      std::stringstream ss;
      ss << "indicies index " << i
         << " expected to be less than or equal to input "
         << " (" << indicesTensor.getShape()[i]
         << " <= " << inputTensor.getShape()[i] << ")";
      return emitOpError(ss.str());
    }
  }

  if (getResult().getType().getShape() != indicesTensor.getShape()) {
    return emitOpError(
        "Expect the shape of the indicies to match the output shape");
  }

  if (getResult().getType().getElementType() != inputTensor.getElementType()) {
    return emitOpError(
        "Expect the element type of the return to match the input");
  }

  return success();
}

LogicalResult GatherNDOp::verify() {
  auto inputTensor = cast<RankedTensorType>(getInput().getType());
  auto indicesTensor = cast<RankedTensorType>(getIndices().getType());

  if (indicesTensor.getRank() < 1)
    return emitError("indicies tensor should have a rank of at least one");
  if (indicesTensor.getShape()[indicesTensor.getRank() - 1] ==
      ShapedType::kDynamic)
    return emitError(
        "Last dimension of the indicies tensor can not be dynamic");
  if (indicesTensor.getShape()[indicesTensor.getRank() - 1] >
      inputTensor.getRank())
    return emitError("The last dimension of the indicies tensor should be used "
                     "to index into the input tensor.  Its shape is too large");

  SmallVector<int64_t> outputShape;
  for (int i = 0; i < indicesTensor.getRank() - 1; i++)
    outputShape.push_back(indicesTensor.getShape()[i]);
  for (int i = indicesTensor.getShape()[indicesTensor.getRank() - 1];
       i < inputTensor.getRank(); i++)
    outputShape.push_back(inputTensor.getShape()[i]);

  auto outputType =
      RankedTensorType::get(outputShape, inputTensor.getElementType());

  if (outputType != getResult().getType()) {
    std::string ss =
        "Output shape of tcp.gather_nd does not match what is expected ";
    llvm::raw_string_ostream rs(ss);
    outputType.print(rs);
    rs.flush();
    ss += " != ";
    getResult().getType().print(rs);
    rs.flush();
    return emitError(ss);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BindSymbolicShapeOp
//===----------------------------------------------------------------------===//

//
// tcp.bind_symbolic_shape %6, [%0, %1, %2], affine_map<()[s0, s1, s2] ->
// (s0, s1 * 2 + s2, 3)> : tensor<?x?x3xf32>
//

ParseResult BindSymbolicShapeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  SmallVector<OpAsmParser::UnresolvedOperand> shapeSymbols;
  AffineMapAttr shapeExpressions;
  Type operandType;

  if (parser.parseOperand(operand) || parser.parseComma() ||
      parser.parseLSquare() || parser.parseOperandList(shapeSymbols) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseAttribute(shapeExpressions, "shape_expressions",
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType)) {
    return failure();
  }

  if (parser.resolveOperand(operand, operandType, result.operands) ||
      parser.resolveOperands(shapeSymbols,
                             parser.getBuilder().getType<IntegerType>(64),
                             result.operands)) {
    return failure();
  }

  return success();
}

// Use a custom printer here to avoid the AffineMap from getting hoisted
// when printed. This makes it so the AffineMap is printed inline with the op.
void BindSymbolicShapeOp::print(OpAsmPrinter &p) {
  p << " " << getOperand() << ", [";
  llvm::interleaveComma(getShapeSymbols(), p);
  p << "], "
    << "affine_map<" << getShapeExpressions().getValue() << ">";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"shape_expressions"});
  p << " : " << getOperand().getType();
}

LogicalResult BindSymbolicShapeOp::verify() {
  if (getShapeSymbols().empty())
    return emitOpError() << "requires non-empty shapeSymbols";

  for (auto symbol : getShapeSymbols()) {
    Operation *definingOp = symbol.getDefiningOp();
    if (!isa<SymbolicIntOp>(definingOp)) {
      return emitOpError()
             << "shape symbol must be produced by a SymbolicIntOp";
    }
  }

  return success();
}

} // namespace mlir::tcp
