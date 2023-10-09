//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "InitAll.h"

#include "Conversion/Passes.h"
#include "IR/TcpDialect.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

void mlir::tcp::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<tcp::TcpDialect>();
  registry.insert<torch::Torch::TorchDialect>();
  mlir::func::registerInlinerExtension(registry);
}

void mlir::tcp::registerAllPasses() {
  mlir::tcp::registerConversionPasses();
  mlir::tcp::registerTcpPasses();
}
