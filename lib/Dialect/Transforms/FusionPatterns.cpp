//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/Transforms/FusionPatterns.h"
#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::tcp {
LogicalResult
GenericBottomUpFuser::matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const {
  Operation *use = op;
  bool opIsInsideGroup = op->getParentOfType<tcp::GroupOp>() != nullptr;
  bool isChanged = false;
  for (auto operand : op->getOperands()) {
    if (operand.getDefiningOp()) {
      Operation *def = operand.getDefiningOp();
      if (canFuse(def, use)) {
        // Currently we are only fusing ops at the top-level.
        // This is to avoid recursing inside a group and ending up with
        // nested groups that contain the same ops.
        // Since we are iterating bottom up in a block, we only need to
        // check if the def op has a func parent.
        //
        // TODO: Remove this restriction to allow fusing in nested
        // regions.
        if (!isa<func::FuncOp>(def->getParentOp())) {
          continue;
        }

        // We only support fusing def ops that have exactly one use, for
        // now. Special-case the uses of the def in
        // tcp.bind_symbolic_shape
        SmallVector<tcp::BindSymbolicShapeOp> bindSymbolicUsersOfDef;
        SmallVector<Operation *> otherUses;
        for (auto otherUserOfDef : def->getUsers()) {
          if (auto bindSymbolicShapeOp =
                  dyn_cast<tcp::BindSymbolicShapeOp>(otherUserOfDef)) {
            bindSymbolicUsersOfDef.push_back(bindSymbolicShapeOp);
          } else {
            otherUses.push_back(otherUserOfDef);
          }
        }

        bool canFuse = false;
        if (otherUses.size() > 1) {
          // If we have more than one use, either
          // 1. All those uses are used by the current op
          if (llvm::all_of(otherUses,
                           [&](Operation *userOp) { return userOp == op; }))
            canFuse = true;

          // 2. All those uses are in the same group as the current op
          if (opIsInsideGroup &&
              llvm::all_of(otherUses, [&](Operation *userOp) {
                return userOp->getParentRegion() == op->getParentRegion();
              }))
            canFuse = true;
        } else if (otherUses.size() == 1) {
          // If we have exactly one use, then we can fuse.
          canFuse = true;
        }

        if (!canFuse)
          continue;

        // Fuse the def and use ops into a group.

        // * If both the ops have the same parent region, they must be
        // part
        //   of the top-level func. So, we need to create a new group.
        // * The only other case is when the def op is part of the
        // top-level
        //   func and the use is already inside a group.
        isChanged = true;
        if (def->getParentRegion() == use->getParentRegion()) {
          auto groupOp = rewriter.create<tcp::GroupOp>(use->getLoc(),
                                                       use->getResultTypes());
          if (postFunc) {
            postFunc(groupOp, rewriter);
          }
          Block *groupBlock = new Block();
          groupOp.getBody().push_back(groupBlock);
          for (unsigned num = 0; num < use->getNumResults(); ++num) {
            rewriter.replaceAllUsesWith(use->getResult(num),
                                        groupOp->getResult(num));
          }
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(groupBlock);
            auto yieldOp =
                rewriter.create<tcp::YieldOp>(use->getLoc(), use->getResults());
            use->moveBefore(yieldOp);
            def->moveBefore(use);
          }
        } else if (auto groupOp = dyn_cast<tcp::GroupOp>(use->getParentOp())) {
          // We already know that all other uses are in the same group
          // and because we are doing this bottom up, this is the "first"
          // use of this op in this group. So its OK to move it to just
          // before this use.
          def->moveBefore(use);
        } else {
          llvm_unreachable("Unhandled case during fusion");
        }

        for (auto bindSymbolicShapeOp : bindSymbolicUsersOfDef) {
          bindSymbolicShapeOp->moveAfter(def);
        }
      }
    }
  }
  return isChanged ? success() : failure();
}
} // namespace mlir::tcp
