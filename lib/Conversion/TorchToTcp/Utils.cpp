//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;
using namespace mlir::tcp;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir::torch_to_tcp {

SignednessAttr
getTcpSignednessAttr(MLIRContext *context,
                     IntegerType::SignednessSemantics signednessInfo) {
  if (signednessInfo == IntegerType::SignednessSemantics::Signless)
    return SignednessAttr::get(context, Signedness::Signless);
  if (signednessInfo == IntegerType::SignednessSemantics::Signed)
    return SignednessAttr::get(context, Signedness::Signed);
  return SignednessAttr::get(context, Signedness::Unsigned);
}

// The parameter input is expected to be of RankedTensorType.
Value broadcastRankInLeadingDims(ConversionPatternRewriter &rewriter,
                                 Value input, int64_t rankIncrease) {
  if (rankIncrease == 0)
    return input;
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();

  SmallVector<ReassociationExprs> reassociationMap(inputType.getRank());
  if (inputType.getRank() > 0) {
    for (int64_t axis = 0; axis < rankIncrease; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    for (int64_t inputAxis = 0; inputAxis < inputType.getRank(); ++inputAxis)
      reassociationMap[inputAxis].push_back(
          rewriter.getAffineDimExpr(inputAxis + rankIncrease));
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape(rankIncrease, 1);
  resultShape.insert(resultShape.end(), inputShape.begin(), inputShape.end());
  auto resultType =
      inputType.cloneWith(ArrayRef(resultShape), inputType.getElementType());

  return rewriter.create<tensor::ExpandShapeOp>(
      input.getDefiningOp()->getLoc(), resultType, input, reassociationMap);
}

Value broadcastRank0Dor1DToND(ConversionPatternRewriter &rewriter, Value input,
                              int64_t targetRank, int64_t axisInOutput) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto inputRank = inputType.getRank();
  assert(inputRank < 2 && "Only 0D and 1D tensors are supported!");

  // Case 1: 0D -> ND
  // [] -> [1, 1, 1, 1]
  // reassociation map = [[]]
  // Case 2: 1D -> ND
  // [C] -> [1, C, 1, 1] if axisInOutput = 1
  // reassociation map = [[0, 1, 2, 3]]
  SmallVector<ReassociationExprs> reassociationMap(inputRank);
  SmallVector<int64_t> resultShape(targetRank, 1);
  if (inputRank == 1) {
    for (int64_t axis = 0; axis < targetRank; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    resultShape[axisInOutput] = inputType.getShape()[0];
  }
  Type expandResultType =
      inputType.cloneWith(ArrayRef(resultShape), inputType.getElementType());
  return rewriter.create<tensor::ExpandShapeOp>(input.getDefiningOp()->getLoc(),
                                                expandResultType, input,
                                                reassociationMap);
}

Value broadcastShapeExceptDims(ConversionPatternRewriter &rewriter, Value input,
                               Value target,
                               llvm::SmallDenseSet<int64_t> dimsToExclude) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto inputShape = inputType.getShape();

  RankedTensorType targetType = target.getType().cast<RankedTensorType>();
  auto targetShape = targetType.getShape();

  SmallVector<int64_t> axes;
  SmallVector<Value> dimSizes;
  SmallVector<int64_t> resultShape;
  // Ensure that dimsToBroadcast is sorted.
  for (int64_t axis = 0; axis < targetType.getRank(); ++axis) {
    if (dimsToExclude.contains(axis)) {
      resultShape.push_back(inputShape[axis]);
    } else {
      resultShape.push_back(targetShape[axis]);
      axes.push_back(axis);
      dimSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          input.getDefiningOp()->getLoc(), target, axis));
    }
  }
  auto axesAttr = rewriter.getI64ArrayAttr(axes);

  Type broadcastResultType =
      inputType.cloneWith(resultShape, inputType.getElementType());
  return rewriter.create<tcp::BroadcastOp>(input.getDefiningOp()->getLoc(),
                                           broadcastResultType, input, dimSizes,
                                           axesAttr);
}

// The parameters input are expected to be of RankedTensorType.
std::pair<Value, Value>
broadcastToMatchShape(ConversionPatternRewriter &rewriter, Value lhs,
                      Value rhs) {
  RankedTensorType inputAType = lhs.getType().cast<RankedTensorType>();
  RankedTensorType inputBType = rhs.getType().cast<RankedTensorType>();

  Value resultA = lhs;
  Value resultB = rhs;
  if (inputAType.getRank() > inputBType.getRank())
    resultB = broadcastRankInLeadingDims(
        rewriter, resultB, inputAType.getRank() - inputBType.getRank());
  if (inputAType.getRank() < inputBType.getRank())
    resultA = broadcastRankInLeadingDims(
        rewriter, resultA, inputBType.getRank() - inputAType.getRank());

  inputAType = resultA.getType().cast<RankedTensorType>();
  inputBType = resultB.getType().cast<RankedTensorType>();
  SmallVector<int64_t> inputAShape(inputAType.getShape().begin(),
                                   inputAType.getShape().end());
  SmallVector<int64_t> inputBShape(inputBType.getShape().begin(),
                                   inputBType.getShape().end());
  assert(inputAShape.size() == inputBShape.size());

  Operation *opA = lhs.getDefiningOp();
  Operation *opB = rhs.getDefiningOp();
  SmallVector<int64_t> axesA, axesB;
  SmallVector<Value> dimSizesA, dimSizesB;

  for (size_t curDim = 0; curDim < inputAShape.size(); curDim++) {
    if (inputAShape[curDim] == 1 && inputBShape[curDim] != 1) {
      axesA.push_back(curDim);
      dimSizesA.push_back(
          rewriter.createOrFold<tensor::DimOp>(opA->getLoc(), resultB, curDim));
      inputAShape[curDim] = inputBShape[curDim];
    }
    if (inputBShape[curDim] == 1 && inputAShape[curDim] != 1) {
      axesB.push_back(curDim);
      dimSizesB.push_back(
          rewriter.createOrFold<tensor::DimOp>(opB->getLoc(), resultA, curDim));
      inputBShape[curDim] = inputAShape[curDim];
    }
  }
  if (axesA.size() > 0) {
    auto axesAttr = rewriter.getI64ArrayAttr(axesA);
    Type resultType =
        inputAType.cloneWith(inputAShape, inputAType.getElementType());
    resultA = rewriter.create<tcp::BroadcastOp>(opA->getLoc(), resultType,
                                                resultA, dimSizesA, axesAttr);
  }
  if (axesB.size() > 0) {
    auto axesAttr = rewriter.getI64ArrayAttr(axesB);
    Type resultType =
        inputBType.cloneWith(inputBShape, inputBType.getElementType());
    resultB = rewriter.create<tcp::BroadcastOp>(opB->getLoc(), resultType,
                                                resultB, dimSizesB, axesAttr);
  }

  return std::make_pair(resultA, resultB);
}

Value broadcast0DOr1DToNDAndMatchShape(ConversionPatternRewriter &rewriter,
                                       Value input, Value target,
                                       Type resultType, int64_t axisInOutput) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  RankedTensorType targetType = target.getType().cast<RankedTensorType>();

  auto inputRank = inputType.getRank();
  auto targetRank = targetType.getRank();

  // This utility only accepts 0D and 1D inputs
  assert(inputRank < 2 && "Only 0D and 1D tensors are supported!");

  // First: Broadcast Rank
  Value result =
      broadcastRank0Dor1DToND(rewriter, input, targetRank, axisInOutput);

  // Second: Broadcast Shape
  // Case 1: 0D -> ND
  // [1, 1, 1, 1] -> [N, C, H, W]
  // Second: Broadcast Shape
  // Case 2: 1D -> ND
  // [1, C, 1, 1] -> [N, C, H, W]
  SmallVector<int64_t> axes;
  SmallVector<Value> dimSizes;
  for (int64_t axis = 0; axis < targetRank; ++axis) {
    if (inputRank == 0 || axis != axisInOutput) {
      axes.push_back(axis);
      dimSizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          result.getDefiningOp()->getLoc(), target, axis));
    }
  }
  auto axesAttr = rewriter.getI64ArrayAttr(axes);

  Type broadcastResultType =
      targetType.cloneWith(targetType.getShape(), resultType);
  result = rewriter.create<tcp::BroadcastOp>(result.getDefiningOp()->getLoc(),
                                             broadcastResultType, result,
                                             dimSizes, axesAttr);

  return result;
}

SmallVector<int64_t> getShapeFromPrimList(ArrayRef<Value> listVal) {
  SmallVector<int64_t> resultShape;
  for (Value value : listVal) {
    int64_t num;
    if (matchPattern(value, m_TorchConstantInt(&num)))
      resultShape.push_back(num);
    else
      resultShape.push_back(ShapedType::kDynamic);
  }
  return resultShape;
}

Value broadcast0DOr1DFromShape(ConversionPatternRewriter &rewriter, Value input,
                               ArrayRef<Value> targetVal,
                               SmallVector<int64_t> resultShape,
                               int64_t axisInOutput) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto inputRank = inputType.getRank();
  RankedTensorType targetType = input.getType().cast<RankedTensorType>();

  int64_t targetRank = 0;
  SmallVector<Value> dimSizes;
  for (Value value : targetVal) {
    targetRank++;
    Value newDimSize = rewriter.create<torch::TorchConversion::ToI64Op>(
        input.getDefiningOp()->getLoc(), value);
    dimSizes.push_back(rewriter.create<arith::IndexCastOp>(
        input.getDefiningOp()->getLoc(), rewriter.getIndexType(), newDimSize));
  }

  SmallVector<ReassociationExprs> reassociationMap(inputRank);
  SmallVector<int64_t> expandShape(targetRank, 1);

  if (inputRank == 1) {
    for (int64_t axis = 0; axis < targetRank; ++axis)
      reassociationMap[0].push_back(rewriter.getAffineDimExpr(axis));
    resultShape[axisInOutput] = inputType.getShape()[0];
    expandShape[axisInOutput] = inputType.getShape()[0];
  }

  Value result = input;
  auto resultType =
      targetType.cloneWith(ArrayRef(expandShape), targetType.getElementType());
  result = rewriter.create<tensor::ExpandShapeOp>(
      result.getDefiningOp()->getLoc(), resultType, input, reassociationMap);

  SmallVector<int64_t> axes;
  for (int64_t axis = 0; axis < targetRank; ++axis) {
    if (inputRank == 0 || axis != axisInOutput) {
      axes.push_back(axis);
    }
  }
  auto axesAttr = rewriter.getI64ArrayAttr(axes);
  resultType =
      targetType.cloneWith(ArrayRef(resultShape), targetType.getElementType());
  result = rewriter.create<tcp::BroadcastOp>(
      result.getDefiningOp()->getLoc(), resultType, result, dimSizes, axesAttr);

  return result;
}

Value castTensorToDtype(ConversionPatternRewriter &rewriter, Type srcType,
                        Type dstType, Value input, Type convertedType) {
  if (srcType == dstType)
    return input;

  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  auto resultType = inputType.cloneWith(inputType.getShape(), convertedType);

  SignednessAttr inputSignedness;
  SignednessAttr outputSignedness;
  if (auto inputIntType = srcType.dyn_cast<mlir::IntegerType>())
    inputSignedness = getTcpSignednessAttr(input.getDefiningOp()->getContext(),
                                           inputIntType.getSignedness());
  if (auto outputIntType = dstType.dyn_cast<mlir::IntegerType>())
    outputSignedness = getTcpSignednessAttr(input.getDefiningOp()->getContext(),
                                            outputIntType.getSignedness());
  return rewriter.create<tcp::CastOp>(input.getDefiningOp()->getLoc(),
                                      resultType, input, inputSignedness,
                                      outputSignedness);
}

// TODO: Add unit tests for all getConstTensor* functions below
template <typename T>
std::optional<Value> impl::getConstTensorUtil(PatternRewriter &rewriter,
                                              Operation *op, ArrayRef<T> vec,
                                              ArrayRef<int64_t> shape,
                                              RankedTensorType type) {
  uint64_t numTotalElements = 1;
  for (int64_t a : shape) {
    assert(a >= 0 && "getConstTensor(): Only static shapes supported");
    numTotalElements *= a;
  }

  if (vec.size() != numTotalElements) {
    op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto constAttr = DenseElementsAttr::get(type, vec);

  auto constOp = rewriter.create<tcp::ConstOp>(op->getLoc(), type, constAttr);
  return constOp.getResult();
}

template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape) {
  auto constType =
      RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));

  return impl::getConstTensorUtil<T>(rewriter, op, vec, shape, constType);
}

template std::optional<Value> getConstTensor<int8_t>(PatternRewriter &,
                                                     Operation *,
                                                     ArrayRef<int8_t> vec,
                                                     ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<int32_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int32_t> vec,
                                                      ArrayRef<int64_t> shape);

template std::optional<Value> getConstTensor<int64_t>(PatternRewriter &,
                                                      Operation *,
                                                      ArrayRef<int64_t> vec,
                                                      ArrayRef<int64_t> shape);

template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(shape, rewriter.getF32Type());

  return impl::getConstTensorUtil<float>(rewriter, op, vec, shape, constType);
}

template <>
std::optional<Value> getConstTensor<double>(PatternRewriter &rewriter,
                                            Operation *op, ArrayRef<double> vec,
                                            ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(shape, rewriter.getF64Type());

  return impl::getConstTensorUtil<double>(rewriter, op, vec, shape, constType);
}

template <>
std::optional<Value> getConstTensor<APInt>(PatternRewriter &rewriter,
                                           Operation *op, ArrayRef<APInt> vec,
                                           ArrayRef<int64_t> shape) {
  auto constType = RankedTensorType::get(
      shape, rewriter.getIntegerType(vec[0].getBitWidth()));

  return impl::getConstTensorUtil<APInt>(rewriter, op, vec, shape, constType);
}

bool getConstTensorWithType(ConversionPatternRewriter &rewriter, Operation *op,
                            Value &constOp, Type resultType, int fillVal) {
  if (resultType.isInteger(64)) {
    constOp = *getConstTensor<int64_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int64_t>(fillVal)), {});
  } else if (resultType.isInteger(32)) {
    constOp = *getConstTensor<int32_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int32_t>(fillVal)), {});
  } else if (resultType.isInteger(8)) {
    constOp = *getConstTensor<int8_t>(
        rewriter, op, llvm::ArrayRef(static_cast<int8_t>(fillVal)), {});
  } else if (resultType.isF32()) {
    constOp = *getConstTensor<float>(
        rewriter, op, llvm::ArrayRef(static_cast<float>(fillVal)), {});
  } else if (resultType.isF64()) {
    constOp = *getConstTensor<double>(
        rewriter, op, llvm::ArrayRef(static_cast<double>(fillVal)), {});
  } else {
    return false;
  }
  return true;
}

// scalarValue should be accessed by op itself, not through the adaptor
Value scalarToTcpTensor(ConversionPatternRewriter &rewriter, Operation *op,
                        Type targetType, Value scalarValue) {
  double doubleValue;
  auto isFloat = matchPattern(scalarValue, m_TorchConstantFloat(&doubleValue));
  if (isFloat) {
    return *getConstTensor<double>(rewriter, op, llvm::ArrayRef(doubleValue),
                                   {});
  }

  int64_t intValue;
  auto isInt = matchPattern(scalarValue, m_TorchConstantInt(&intValue));
  if (isInt) {
    return *getConstTensor<int64_t>(rewriter, op, llvm::ArrayRef(intValue), {});
  }

  return rewriter.create<tensor::FromElementsOp>(op->getLoc(), targetType,
                                                 ArrayRef<Value>{scalarValue});
}

void TorchToTcpCustomOpConversionHelper::addOperand(std::string opName,
                                                    Value value) {
  if (conversionResult.failed())
    return;
  operandNames.push_back(opName);
  operands.push_back(value);
}

void TorchToTcpCustomOpConversionHelper::addAsMultipleTensorOperands(
    std::string opNamePrefix, mlir::Value value) {
  if (conversionResult.failed())
    return;
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

LogicalResult TorchToTcpCustomOpConversionHelper::replace() {
  if (conversionResult.failed()) {
    return conversionResult;
  }

  SmallVector<Type> resultTypes;
  auto result = typeConverter->convertTypes(op->getResultTypes(), resultTypes);
  if (result.failed()) {
    return result;
  }

  SmallVector<StringRef> operandNameRefs;
  operandNameRefs.append(operandNames.begin(), operandNames.end());

  attrs.push_back(rewriter.getNamedAttr(
      "torch_operand_names", rewriter.getStrArrayAttr(operandNameRefs)));

  auto replOp = rewriter.replaceOpWithNewOp<tcp::CustomOp>(op, resultTypes,
                                                           operands, attrs);
  replOp.setOpName(op->getName().getStringRef());
  return success();
}

void TorchToTcpCustomOpConversionHelper::addBoolAttr(std::string attrName,
                                                     Value value) {
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

void TorchToTcpCustomOpConversionHelper::addIntAttr(std::string attrName,
                                                    Value value) {
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

void TorchToTcpCustomOpConversionHelper::addFloatAttr(std::string attrName,
                                                      Value value) {
  if (conversionResult.failed())
    return;

  double constVal;
  if (!matchPattern(value, torch::Torch::m_TorchConstantFloat(&constVal))) {
    conversionResult = rewriter.notifyMatchFailure(
        op, std::string("non-const ") + attrName + " unsupported");
    return;
  }
  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getF64FloatAttr(constVal)));
}

void TorchToTcpCustomOpConversionHelper::addListOfIntsAttr(std::string attrName,
                                                           Value value) {
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

void TorchToTcpCustomOpConversionHelper::addDenseIntArrayAttr(
    std::string attrName, ArrayRef<int64_t> values) {
  if (conversionResult.failed())
    return;

  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getDenseI64ArrayAttr(values)));
}

void TorchToTcpCustomOpConversionHelper::addDenseFloatArrayAttr(
    std::string attrName, ArrayRef<double> values) {
  if (conversionResult.failed())
    return;

  attrs.push_back(
      rewriter.getNamedAttr(attrName, rewriter.getDenseF64ArrayAttr(values)));
}

} // namespace mlir::torch_to_tcp
