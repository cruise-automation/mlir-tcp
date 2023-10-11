//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tcp {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpFuseElementwiseOpsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpIsolateGroupOpsPass();

/// Registers all Tcp related passes.
void registerTcpPasses();

} // namespace mlir::tcp
