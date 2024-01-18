//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace mlir::tcp {

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
/// 2. Some of the tensor operands are converted into named attributes of
/// the final tcp.custom_op
///
/// See `ConvertAtenConvolutionOp` and `ConvertAten_IndexPutImplOp` for an
/// example usages.
class TcpCustomOpBuilder {

public:
  TcpCustomOpBuilder(Operation *a_op, ConversionPatternRewriter &a_rewriter,
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

  // Add value as a named list of integers attribute
  void addListOfIntsAttr(std::string attrName, Value value);

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

} // namespace mlir::tcp
