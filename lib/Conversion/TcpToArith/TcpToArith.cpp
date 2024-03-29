//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TcpToArith/TcpToArith.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOARITH
#include "mlir-tcp/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class ConstOpConverter : public OpRewritePattern<tcp::ConstOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tcp::ConstOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

void populateTcpToArithPatternsAndLegality(RewritePatternSet &patterns,
                                           ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<tcp::ConstOp>();
  patterns.add<ConstOpConverter>(context);
}

class ConvertTcpToArith : public ConvertTcpToArithBase<ConvertTcpToArith> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(context);
    populateTcpToArithPatternsAndLegality(patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTcpToArithPass() {
  return std::make_unique<ConvertTcpToArith>();
}

} // namespace tcp
} // namespace mlir
