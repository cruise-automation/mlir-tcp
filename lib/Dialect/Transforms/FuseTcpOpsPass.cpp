//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/Transforms/FuseTcpOpsPass.h"
#include "mlir-tcp/Dialect/Transforms/FusionPatterns.h"
#include "mlir-tcp/Dialect/Transforms/Passes.h"

#include "./PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::tcp {
namespace {

class TcpFuseElementwiseOpsPass
    : public TcpFuseElementwiseOpsBase<TcpFuseElementwiseOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    auto canFuse = [](Operation *def, Operation *use) -> bool {
      return def->hasTrait<OpTrait::Elementwise>() &&
             use->hasTrait<OpTrait::Elementwise>();
    };
    patterns.add<GenericBottomUpFuser>(context, canFuse);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTcpFuseElementwiseOpsPass() {
  return std::make_unique<TcpFuseElementwiseOpsPass>();
}

} // namespace mlir::tcp
