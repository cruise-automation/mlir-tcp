//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tcp {

class GenericBottomUpFuser : public RewritePattern {
public:
  using CanFuseFuncType = std::function<bool(Operation *, Operation *)>;
  using PostProcessingFuncType =
      std::function<void(Operation *, PatternRewriter &rewriter)>;

  // A class for supporting generic bottom-up fusion
  // All fused operations will be placed in a single TCP group
  // canFuseCallback checks whether two operations can be fused
  // postFuncCallback is called on the new TCP group
  GenericBottomUpFuser(MLIRContext *context, CanFuseFuncType canFuseCallback,
                       PostProcessingFuncType postFuncCallback = nullptr)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        canFuse(canFuseCallback), postFunc(postFuncCallback) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

private:
  CanFuseFuncType canFuse;
  PostProcessingFuncType postFunc;
};
} // namespace mlir::tcp
