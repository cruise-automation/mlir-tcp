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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Debug.h"

#ifndef NDEBUG
#define DEBUG_TYPE "tcp-fusion-patterns"
#endif

namespace mlir::tcp {

LogicalResult
GenericBottomUpFuser::matchAndRewrite(Operation *op,
                                      PatternRewriter &rewriter) const {

  // Currently we are only fusing ops at the top-level.
  // This is to avoid recursing inside a group and ending up with
  // nested groups that contain the same ops.
  // Since we are iterating bottom up in a block, we only need to check
  // if the def op has a func parent.
  //
  // TODO: Remove this restriction to allow fusing in nested regions.
  if (!isa<func::FuncOp>(op->getParentOp()))
    return failure();

  if (op->use_empty())
    return failure();

  // We can only fuse a def with multiple uses if all the uses belong to the
  // same region and can be fused with the defining op
  Region *usesParentRegion = nullptr;
  SmallVector<Operation *> uses;
  llvm::DenseSet<Operation *> usesSet;
  llvm::DenseSet<tcp::BindSymbolicShapeOp> bindShapeUses;

  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op << "\n");
  for (auto &use : op->getUses()) {
    if (auto bindShapeOp = dyn_cast<tcp::BindSymbolicShapeOp>(use.getOwner())) {
      bindShapeUses.insert(bindShapeOp);
      continue;
    }

    auto parentRegion = use.getOwner()->getParentRegion();
    if (usesParentRegion && usesParentRegion != parentRegion)
      return failure();
    usesParentRegion = parentRegion;

    if (!canFuse(op, use.getOwner()))
      return failure();

    if (usesSet.insert(use.getOwner()).second)
      uses.push_back(use.getOwner());
  }

  // All its uses are tcp.bind_symbolic_shape ops.
  if (uses.empty())
    return failure();

  // Sorting by dominance ensures that the first element of this vector is
  // the first use of the def. Used below when we want to move the op into
  // an existing group.
  LLVM_DEBUG(llvm::dbgs() << "Processing op: " << *op << " with " << uses.size()
                          << " uses\n");
  DominanceInfo domInfo;
  llvm::stable_sort(uses, [&](Operation *a, Operation *b) {
    return domInfo.dominates(a, b);
  });

#ifndef NDEBUG
  for (auto use : uses) {
    LLVM_DEBUG(llvm::dbgs() << "Use: " << *use << "\n");
  }
#endif

  if (op->getParentRegion() == usesParentRegion) {
    LLVM_DEBUG(llvm::dbgs() << "Creating new group\n");
    // this case can only happen when all ops belong to the function.
    SmallVector<Type> allResultTypes;
    SmallVector<Value> allResults;
    for (auto use : uses) {
      allResultTypes.append(use->getResultTypes().begin(),
                            use->getResultTypes().end());
      allResults.append(use->getResults().begin(), use->getResults().end());
    }

    auto groupOp = rewriter.create<tcp::GroupOp>(op->getLoc(), allResultTypes);
    if (postFunc) {
      postFunc(groupOp, rewriter);
    }
    Block *groupBlock = new Block();
    groupOp.getBody().push_back(groupBlock);

    // First move all uses into the group in the dominance order
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(groupBlock);
      auto yieldOp = rewriter.create<tcp::YieldOp>(op->getLoc(), allResults);
      // This is where we are using the sorted-ness of `uses`. We are
      // guaranteed that if the users of the op themselves depend on each
      // other, then we'll move them in the correct order.
      for (auto use : uses) {
        use->moveBefore(yieldOp);
      }
      op->moveBefore(*uses.begin());
      for (auto bindShapeOp : bindShapeUses) {
        bindShapeOp->moveAfter(op);
      }
    }

    // We then replace all uses of the uses which lie outside the group
    // with the group's results. We should not replace uses inside the
    // group otherwise ops inside the group will end up depending on the
    // group's results causing dominance issues.
    size_t groupResultNum = 0;
    for (auto use : uses) {
      for (unsigned num = 0; num < use->getNumResults(); ++num) {
        auto useIsOutsideGroup = [&](OpOperand &operand) {
          return operand.getOwner()->getParentOp() != groupOp;
        };
        rewriter.replaceUsesWithIf(use->getResult(num),
                                   groupOp->getResult(groupResultNum),
                                   useIsOutsideGroup);
        groupResultNum++;
      }
    }

  } else if (auto groupOp =
                 dyn_cast<tcp::GroupOp>(usesParentRegion->getParentOp())) {
    // Given that we iterate over the funcop in a bottom up manner, when moving
    // into an existing group, we would be guaranteed that this op does not use
    // any of the ops already in the group. So we can move it to the very
    // beginning of the group. This ensures that the order of operands is
    // preserved when creating a group. For example, if we start with
    // something like:
    //
    // %0 = op1(%in1)
    // %1 = op2(%in2)
    // %2 = op3(%0, %1)
    //
    // we'll first create a %1 and %2
    //
    // %0 = op1(%in1)
    // %3 = tcp.group {
    //   %1 = op2(%in2)
    //   %2 = op3(%0, %1)
    // }
    //
    // if we try to move %0 to right before its use in the group, then we'd
    // end up with:
    //
    // %3 = tcp.group {
    //   %1 = op2(%in2)
    //   %0 = op1(%in1)
    //   %2 = op3(%0, %1)
    // }
    //
    // While this is not incorrect, it is a bit annoying that the MLIR gets
    // reordered.
    auto &firstOp = *usesParentRegion->getOps().begin();
    op->moveBefore(&firstOp);
    for (auto bindShapeOp : bindShapeUses) {
      bindShapeOp->moveBefore(&firstOp);
    }
  } else {
    op->emitError("Unhandled case during fusion");
    llvm_unreachable("Unhandled case during fusion");
  }
  LLVM_DEBUG(llvm::dbgs() << "Function after transformation:\n"
                          << op->getParentOfType<func::FuncOp>() << "\n");
  return success();
}

} // namespace mlir::tcp
