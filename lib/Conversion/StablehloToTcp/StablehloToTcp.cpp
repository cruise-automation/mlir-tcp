//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/StablehloToTcp/StablehloToTcp.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTCP
#include "mlir-tcp/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class TanhOpConverter : public OpRewritePattern<stablehlo::TanhOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::TanhOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<tcp::TanhOp>(op, op.getType(), op.getOperand());
    return success();
  }
};

void populateStablehloToTcpPatternsAndLegality(RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<stablehlo::TanhOp>();
  patterns.add<TanhOpConverter>(context);
}

class ConvertStablehloToTcp
    : public ConvertStablehloToTcpBase<ConvertStablehloToTcp> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tcp::TcpDialect>();

    RewritePatternSet patterns(context);
    populateStablehloToTcpPatternsAndLegality(patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertStablehloToTcpPass() {
  return std::make_unique<ConvertStablehloToTcp>();
}

} // namespace tcp
} // namespace mlir
