//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TcpToTensor/TcpToTensor.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTCPTOTENSOR
#include "mlir-tcp/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class SliceOpConverter : public OpConversionPattern<tcp::SliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tcp::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sliceOp = rewriter.create<tensor::ExtractSliceOp>(
        op.getLoc(), adaptor.getIn(), getAsOpFoldResult(adaptor.getStarts()),
        getAsOpFoldResult(adaptor.getSizes()),
        getAsOpFoldResult(adaptor.getStrides()));
    rewriter.replaceOp(op, sliceOp.getResult());
    return success();
  }
};

/**
 * TODO: the tensor.scatter is still not lowering to anything after this....
 */
class ConvertScatterNDOp : public OpConversionPattern<ScatterNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

    LogicalResult
  matchAndRewrite(ScatterNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value indicesIn = adaptor.getIndices();
    auto indicesType = cast<RankedTensorType>(indicesIn.getType());
    auto indices = rewriter.create<arith::IndexCastOp>(op.getLoc(), 
    RankedTensorType::get(indicesType.getShape(), rewriter.getIndexType()),
     adaptor.getIndices()).getResult();
    int numIndices = cast<RankedTensorType>(indicesIn.getType()).getShape().back();
    SmallVector<int64_t> scatterDims;
    for(int i = 0; i < numIndices; i++) scatterDims.push_back(i);
    rewriter.replaceOpWithNewOp<tensor::ScatterOp>(
      op, op.getType(), adaptor.getValues(), adaptor.getInput(), indices,
      scatterDims, true
    );
    return success();
  }

};

void populateTcpToTensorPatternsAndLegality(RewritePatternSet &patterns,
                                            ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<tcp::SliceOp>();
  patterns.add<SliceOpConverter>(context);
  target.addIllegalOp<tcp::ScatterNDOp>();
  patterns.add<ConvertScatterNDOp>(context);
}

class ConvertTcpToTensor : public ConvertTcpToTensorBase<ConvertTcpToTensor> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();

    RewritePatternSet patterns(context);
    populateTcpToTensorPatternsAndLegality(patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTcpToTensorPass() {
  return std::make_unique<ConvertTcpToTensor>();
}

} // namespace tcp
} // namespace mlir
