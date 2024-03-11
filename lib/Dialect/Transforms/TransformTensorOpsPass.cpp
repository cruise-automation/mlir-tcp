//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/Transforms/TransformTensorOpsPass.h"
#include "mlir-tcp/Dialect/Transforms/Passes.h"

#include "./PassDetail.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::tcp {
namespace {

class DecomposeTensorConcatOpsPass
    : public DecomposeTensorConcatOpsBase<DecomposeTensorConcatOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposeTensorConcatOpsPass() {
  return std::make_unique<DecomposeTensorConcatOpsPass>();
}

} // namespace mlir::tcp
