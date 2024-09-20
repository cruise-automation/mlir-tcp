//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/DialectConversion.h"

#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

#include "llvm/ADT/StringSet.h"

namespace mlir {
namespace torch_to_tcp {

// Helper function to get SignednessAttr on the op from signedness
// on the type.
mlir::tcp::SignednessAttr
getTcpSignednessAttr(MLIRContext *context,
                     IntegerType::SignednessSemantics signednessInfo);

mlir::tcp::Signedness
getTcpSignedness(IntegerType::SignednessSemantics signednessInfo);

// Helper function to expand the rank of the input tensor. Works by
// adding 1-dim shape to the leading dims using `tensor::ExpandShapeOp`.
Value broadcastRankInLeadingDims(ConversionPatternRewriter &rewriter,
                                 Value input, int64_t rankIncrease);

// Broadcasts the rank of the input tensor from 0D or 1D to ND. If the input
// tensor is 1D, `axisInOutput` specifies the axis where the input axis should
// end up in the output.
Value broadcastRank0Dor1DToND(ConversionPatternRewriter &rewriter, Value input,
                              int64_t targetRank, int64_t axisInOutput);

// Broadcasts the shape of the input tensor to match the shape of the target
// tensor in all dims except the dims specified in `dimsToExclude`.
Value broadcastShapeExceptDims(ConversionPatternRewriter &rewriter, Value input,
                               Value target,
                               llvm::SmallDenseSet<int64_t> dimsToExclude);

// Helper function to do both rank and shape all-dim broadcasting
// of the inputs to match each other.
std::pair<Value, Value>
broadcastToMatchShape(ConversionPatternRewriter &rewriter, Value lhs,
                      Value rhs);

// Helper function to broadcast a 0D or 1D input tensor to match rank and shape
// of target. For the 1D case, this projects the input vector to the
// `axisInOutput` in the result.
//
// Case 1: 0D->ND
// Example: [] -> [N, C, H, W]
//   First: Broadcast Rank
//      [] -> [1, 1, 1, 1]
//   Second: Broadcast Shape
//      [1, 1, 1, 1] -> [N, C, H, W]
//
// Case 2: 1D->ND
// Example: [C] -> [N, C, H, W] (`axisInOutput = 1`)
//   First: Broadcast Rank
//      [C] -> [1, C, 1, 1]
//   Second: Broadcast Shape
//      [1, C, 1, 1] -> [N, C, H, W]
Value broadcast0DOr1DToNDAndMatchShape(ConversionPatternRewriter &rewriter,
                                       Value input, Value target,
                                       Type resultType,
                                       int64_t axisInOutput = 0);

Value broadcast0DOr1DFromShape(ConversionPatternRewriter &rewriter, Value input,
                               ArrayRef<Value> targetVal,
                               SmallVector<int64_t> resultShape,
                               int64_t axisInOutput = 0);

// Helper function to construct the shape info from a PrimListConstructOp.
// default is ShapedType::kDynamic if the element is not a constant
SmallVector<int64_t> getShapeFromPrimList(ArrayRef<Value> listVal);

// Helper function to create a Tcp tensor from a scalar value
Value scalarToTcpTensor(ConversionPatternRewriter &rewriter, Operation *op,
                        Type targetType, Value scalarValue);

// Helper function to convert a Tcp tensor to the target data type
Value castTensorToDtype(ConversionPatternRewriter &rewriter, Type srcType,
                        Type dstType, Value input, Type convertedType);

// Utility function to create a tcp.const op with given content and shape.
template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op,
                                    ArrayRef<T> vec, ArrayRef<int64_t> shape);

// Utility function to create a tcp.const op with given type info.
bool getConstTensorWithType(ConversionPatternRewriter &rewriter, Operation *op,
                            Value &constOp, Type resultType, int fillVal);

// Utility function to selectively add a torch->tcp pattern if whitelist op is
// provided
template <typename TorchToTcpPattern, typename AtenOp>
inline void addPatternIfOpInConvertTorchOpsSet(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const llvm::StringSet<> &convertTorchOpsSet,
    std::function<bool(AtenOp)> dynamicLegalityFcn = [](AtenOp) {
      return false;
    }) {
  MLIRContext *context = patterns.getContext();
  std::optional<OperationName> opName =
      TorchToTcpPattern(context).getRootKind();
  assert(opName && "All TorchToTcp patterns must target a single op");
  // When no ops are specified, convert all.
  // When ops are specified, convert those ops only.
  if (convertTorchOpsSet.empty() ||
      convertTorchOpsSet.contains(
          opName->getStringRef().ltrim(torch::Torch::kTorchOpPrefix))) {
    target.addDynamicallyLegalOp<AtenOp>(dynamicLegalityFcn);
    patterns.add<TorchToTcpPattern>(typeConverter, context);
  }
}

namespace impl {
template <typename T>
std::optional<Value>
getConstTensorUtil(PatternRewriter &rewriter, Operation *op, ArrayRef<T> vec,
                   ArrayRef<int64_t> shape, RankedTensorType type);
}

/// Helper class for converting torch ops to tcp.custom_ops. Encapsulates a
/// few common patterns which we see being used in these conversions:
///
/// 1. Some of the tensor operands to the original torch op are converted
/// into positional tensor operands. Given that a tcp.custom_op does not
/// have a strongly typed API to fetch the operands other than by index, we
/// also add an extra StringArray attribute to the tcp.custom_op called
/// "torch_operand_names" which describes what each positional argument of
/// the `tcp.custom_op` corresponds to. This also allows us to handle the
/// case of optional tensor arguments.
///
/// 2. Some of the non-tensor operands are converted into named attributes of
/// the final tcp.custom_op
///
/// See `ConvertAtenConvolutionOp` and `ConvertAten_IndexPutImplOp` for an
/// example usages.
class TorchToTcpCustomOpConversionHelper {

public:
  TorchToTcpCustomOpConversionHelper(Operation *a_op,
                                     ConversionPatternRewriter &a_rewriter,
                                     const TypeConverter *a_typeConverter)
      : op(a_op), rewriter(a_rewriter), typeConverter(a_typeConverter) {}

  /// Add a value as a named tensor operand.
  void addOperand(std::string opName, Value value);

  /// Expand the passed in value as multiple named tensor operands. Expects
  /// that value is a torch list type with tensors as elements
  void addAsMultipleTensorOperands(std::string opNamePrefix, mlir::Value value);

  // Add value as a named bool attribute
  void addBoolAttr(std::string attrName, Value value);

  // Add value as a named integer attribute
  void addIntAttr(std::string attrName, Value value);

  // Add value as a named float attribute
  void addFloatAttr(std::string attrName, Value value);

  // Add value as a named list of integers attribute
  void addListOfIntsAttr(std::string attrName, Value value);

  // Add ArrayRef as a named list of integers attribute
  void addDenseIntArrayAttr(std::string attrName, ArrayRef<int64_t> values);

  // Add ArrayRef as a named list of floats attribute
  void addDenseFloatArrayAttr(std::string attrName, ArrayRef<double> values);

  // Perform final conversion of the original op to a `tcp.custom_op`.
  LogicalResult replace();

private:
  Operation *op;
  ConversionPatternRewriter &rewriter;
  const TypeConverter *typeConverter;
  SmallVector<std::string> operandNames;
  SmallVector<Value> operands;
  SmallVector<NamedAttribute> attrs;
  LogicalResult conversionResult = LogicalResult::success();
};

} // namespace torch_to_tcp
} // namespace mlir
