//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_CONVERTTORCHTOTCPCUSTOMOP
#include "mlir-tcp/Conversion/Passes.h.inc"

namespace tcp {

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToTcpCustomOpPass();
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToTcpCustomOpPass(
    llvm::ArrayRef<std::string> convertTorchOps);

} // namespace tcp
} // namespace mlir
