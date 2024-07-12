//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_PASSES
#define TCP_PASSES

include "mlir/Pass/PassBase.td"

// \brief This is a pass that performs fusion of TCP ops.
def TcpFuseElementwiseOps : Pass<"tcp-fuse-elementwise-ops", "ModuleOp"> {
  let summary = "Performs fusion of tcp elementwise ops";
  let constructor = "mlir::tcp::createTcpFuseElementwiseOpsPass()";
}

// \brief This pass makes all TCP group ops isolated from above.
def TcpIsolateGroupOps : Pass<"tcp-isolate-group-ops", "ModuleOp"> {
  let summary = "Converts all tcp.group ops to tcp.isolated_group ops";
  let constructor = "mlir::tcp::createTcpIsolateGroupOpsPass()";
}

// \brief This pass verifies conformity to the TCP backend contract.
def VerifyTcpBackendContract : Pass<"torch-verify-tcp-backend-contract", "ModuleOp"> {
  let summary = "Verifies conformity to the tcp backend contract";
  let constructor = "mlir::tcp::createVerifyTcpBackendContractPass()";
}

// \brief This pass decomposes tensor ops into simpler tensor ops.
// For example, `tensor::ConcatOp` is decomposed into a destination (`tensor::EmptyOp`)
// and a sequence of slice inserts (`tensor::InsertSliceOp`).
def DecomposeTensorOps : Pass<"decompose-tensor-ops", "func::FuncOp"> {
  let summary = "Decomposes tensor ops into simpler tensor ops.";
  let constructor = "mlir::tcp::createDecomposeTensorOpsPass()";
}

// \brief This pass removes any unused symbolic shape ops.
// We discard remaining bind shape ops during backend lowering.
def DropSymbolicShapeOps : Pass<"drop-symbolic-shape-ops", "func::FuncOp"> {
  let summary = "Removes all remaining symbolic shape ops.";
  let constructor = "mlir::tcp::createDropSymbolicShapeOpsPass()";
}

#endif // TCP_PASSES
