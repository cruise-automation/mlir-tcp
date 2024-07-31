//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/Transforms/IsolateGroupOpsPass.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "mlir-tcp/Dialect/Transforms/Passes.h"

#include "./PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace mlir::tcp {

namespace {

class IsolateGroups : public OpRewritePattern<GroupOp> {
public:
  using OpRewritePattern<GroupOp>::OpRewritePattern;

  IsolateGroups(MLIRContext *context,
                std::function<bool(tcp::GroupOp, Value)> shouldInlineConst)
      : OpRewritePattern<GroupOp>(context),
        shouldInlineConst_(shouldInlineConst) {}

  LogicalResult matchAndRewrite(GroupOp groupOp,
                                PatternRewriter &rewriter) const override {
    // Collect the values used in the given GroupOp. Those will be the inputs
    // to the IsolatedGroup op. The constants used in the GroupOp are collected
    // in a separate set and cloned into the body of the IsolatedGroupOp.
    llvm::SmallVector<Value> inputs;
    llvm::SmallDenseSet<Value> addedInputs;
    llvm::SmallDenseSet<Value> consts;

    groupOp->walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        auto operandDefiningOp = operand.getDefiningOp();
        if (!operandDefiningOp) {
          operandDefiningOp = operand.getParentBlock()->getParentOp();
        }
        if (!groupOp->isProperAncestor(operandDefiningOp)) {
          if (operand.getDefiningOp() &&
              operand.getDefiningOp()->hasTrait<OpTrait::ConstantLike>() &&
              shouldInlineConst_(groupOp, operand)) {
            consts.insert(operand);
          } else if (!addedInputs.contains(operand)) {
            inputs.push_back(operand);
            addedInputs.insert(operand);
          }
        }
      }
    });

    auto isolatedGroupOp = rewriter.create<IsolatedGroupOp>(
        groupOp.getLoc(), groupOp.getResultTypes(), inputs);
    isolatedGroupOp->setAttrs(groupOp->getAttrs());

    isolatedGroupOp.getBody().takeBody(groupOp.getBody());

    auto &isolatedGroupBlock = isolatedGroupOp.getBody().front();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&isolatedGroupBlock);
      auto belongsToIsolatedGroup = [&](OpOperand &opOperand) {
        return (isolatedGroupOp->isProperAncestor(opOperand.getOwner()));
      };

      // Clone the constants at the start of the isolated group block.
      for (auto c : consts) {
        auto newConst = rewriter.clone(*c.getDefiningOp());
        rewriter.replaceUsesWithIf(c, newConst->getResult(0),
                                   belongsToIsolatedGroup);
      }

      // Add inputs as arguments to the isolated group block.
      for (size_t n = 0; n < inputs.size(); ++n) {
        isolatedGroupBlock.addArgument(inputs[n].getType(), groupOp.getLoc());
        rewriter.replaceUsesWithIf(inputs[n], isolatedGroupBlock.getArgument(n),
                                   belongsToIsolatedGroup);
      }
    }
    for (unsigned n = 0; n < groupOp.getNumResults(); ++n) {
      rewriter.replaceAllUsesWith(groupOp->getOpResult(n),
                                  isolatedGroupOp->getOpResult(n));
    }
    rewriter.eraseOp(groupOp);
    return success();
  }

  std::function<bool(tcp::GroupOp, Value)> shouldInlineConst_;
};

class DropSymbolicShapesInsideGroups
    : public OpRewritePattern<BindSymbolicShapeOp> {
  using OpRewritePattern<BindSymbolicShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BindSymbolicShapeOp shapeOp,
                                PatternRewriter &rewriter) const override {
    if (isa<tcp::GroupOp>(shapeOp->getParentOp())) {
      rewriter.eraseOp(shapeOp);
      return success();
    }
    return failure();
  }
};

class TcpIsolateGroupOpsPass
    : public TcpIsolateGroupOpsBase<TcpIsolateGroupOpsPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);

    auto shouldCopyConstPredicate = [&](tcp::GroupOp, Value) { return true; };
    populateIsolateGroupPatterns(patterns, shouldCopyConstPredicate);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTcpIsolateGroupOpsPass() {
  return std::make_unique<TcpIsolateGroupOpsPass>();
}

void populateIsolateGroupPatterns(
    RewritePatternSet &patterns,
    std::function<bool(tcp::GroupOp, Value)> shouldCopyConstPredicate) {

  patterns.add<IsolateGroups>(patterns.getContext(), shouldCopyConstPredicate);
  patterns.add<DropSymbolicShapesInsideGroups>(patterns.getContext());
}

} // namespace mlir::tcp
