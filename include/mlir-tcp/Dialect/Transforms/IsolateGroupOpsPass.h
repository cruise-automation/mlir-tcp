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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::tcp {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTcpIsolateGroupOpsPass();

// `createTcpIsolateGroupOpsPass` will clone all const operations used
// inside a `tcp.group` into the new `tcp.isolated_group` it creates. If
// you want to customize this behavior, you can use this instead to
// pass a predicate function to control when a `const-like` operation
// should be cloned into the isolated group or whether it should be added
// as an argument to the isolated group.
void populateIsolateGroupPatterns(
    RewritePatternSet &patterns,
    std::function<bool(GroupOp, Value)> shouldCopyConstPredicate);

} // namespace mlir::tcp
