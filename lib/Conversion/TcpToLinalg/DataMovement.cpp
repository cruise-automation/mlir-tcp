//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TcpToLinalg/TcpToLinalg.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tcp;

namespace {

class ConvertGatherOp : public OpConversionPattern<GatherOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTensorType = getTypeConverter()
                                ->convertType(op.getOut().getType())
                                .cast<RankedTensorType>();

    auto inputTensor = adaptor.getInput();
    auto indicesTensor = adaptor.getIndices();
    int64_t gatherDim = adaptor.getDimAttr().getValue().getSExtValue();

    auto resultRank = resultTensorType.getRank();

    SmallVector<Value> resultDimSizes;
    for (int64_t i = 0; i < resultRank; ++i) {
      resultDimSizes.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, indicesTensor, i));
    }

    SmallVector<AffineMap, 2> indexingMaps;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);

    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, getAsOpFoldResult(resultDimSizes),
                                         resultTensorType.getElementType());

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      SmallVector<Value> extractIndices;
      for (int64_t i = 0; i < resultRank; ++i) {
        if (i == gatherDim) {
          auto indexCast = b.create<arith::IndexCastOp>(loc, b.getIndexType(),
                                                        payloadArgs[0]);
          extractIndices.push_back(indexCast);
        } else {
          auto iterIndex = b.create<linalg::IndexOp>(loc, b.getIndexType(),
                                                     b.getI64IntegerAttr(i));
          extractIndices.push_back(iterIndex);
        }
      }
      auto extract = b.create<tensor::ExtractOp>(
          loc, resultTensorType.getElementType(), inputTensor, extractIndices);
      b.create<linalg::YieldOp>(loc, extract.getResult());
    };
    Value generic =
        rewriter
            .create<linalg::GenericOp>(loc, emptyTensor.getType(),
                                       indicesTensor, emptyTensor, indexingMaps,
                                       iteratorTypes, bodyBuilder)
            .getResult(0);
    rewriter.replaceOp(op, generic);
    return success();
  }
};

/**
 * tcp.gather_nd is lowered to linalg.generic, which allows us to define every
 * element in the result tensor using a programmatic expression.  The last
 * dimension of the indicies tensor is used to index into the input tensor.
 *
 * For example, we have an indices tensor of shape 9x4x3x2 and an input
 * tensor of shape 5x6x7x8, then the resulting tensor will be of shape
 * 9x4x3x7x8.  Where the first three dimensions of the resulting tensor are used
 * to index into the indicies tensor.  Then the last dimension of the index
 * tensor (the 2 sized dimension) is used to index into the input tensor.
 */
class ConvertGatherNDOp : public OpConversionPattern<GatherNDOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatherNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTensorType = getTypeConverter()
                                ->convertType(op.getOut().getType())
                                .cast<RankedTensorType>();

    auto inputTensor = adaptor.getInput();
    auto indicesTensor = adaptor.getIndices();
    auto indicesType = cast<RankedTensorType>(indicesTensor.getType());
    auto inputType = cast<RankedTensorType>(inputTensor.getType());
    int numGatherAxes = indicesType.getShape().back();

    SmallVector<Value> resultDimSizes;
    for (int i = 0; i < indicesType.getRank() - 1; i++) {
      resultDimSizes.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, indicesTensor, i));
    }
    for (int i = numGatherAxes; i < inputType.getRank(); i++) {
      resultDimSizes.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, inputTensor, i));
    }

    assert(resultDimSizes.size() == resultTensorType.getRank());

    Value emptyTensor =
        rewriter.create<tensor::EmptyOp>(loc, getAsOpFoldResult(resultDimSizes),
                                         resultTensorType.getElementType());

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
      SmallVector<Value> valueIndices, gatherIndices;
      for (int i = 0; i < indicesType.getRank() - 1; i++) {
        auto idx = b.create<linalg::IndexOp>(loc, b.getIndexType(),
                                             b.getI64IntegerAttr(i));
        gatherIndices.push_back(idx);
      }
      for (int i = 0; i < numGatherAxes; i++) {
        SmallVector<Value> gi = gatherIndices;
        auto gidx = b.create<arith::ConstantOp>(loc, b.getIndexAttr(i));
        gi.push_back(gidx);
        assert(gi.size() == indicesType.getRank());
        auto idxExtract = b.create<tensor::ExtractOp>(
            loc, indicesType.getElementType(), indicesTensor, gi);
        auto idxCast =
            b.create<arith::IndexCastOp>(loc, b.getIndexType(), idxExtract);
        valueIndices.push_back(idxCast);
      }
      for (int i = indicesType.getRank() - 1; i < resultTensorType.getRank();
           i++) {
        auto idx = b.create<linalg::IndexOp>(loc, b.getIndexType(),
                                             b.getI64IntegerAttr(i));
        valueIndices.push_back(idx);
      }
      assert(valueIndices.size() == inputType.getRank());
      auto extract =
          b.create<tensor::ExtractOp>(loc, resultTensorType.getElementType(),
                                      inputTensor, valueIndices)
              .getResult();

      b.create<linalg::YieldOp>(loc, extract);
    };

    SmallVector<Value> empty;
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(
        rewriter.getMultiDimIdentityMap(resultTensorType.getRank()));
    SmallVector<utils::IteratorType> iteratorTypes(
        resultTensorType.getRank(), utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, resultTensorType, empty, emptyTensor, indexingMaps, iteratorTypes,
        bodyBuilder);

    rewriter.replaceOp(op, generic.getResult(0));

    return success();
  }
};

} // namespace

void mlir::TcpToLinalg::populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();

  target.addIllegalOp<GatherOp>();
  patterns.add<ConvertGatherOp>(typeConverter, context);
  target.addIllegalOp<GatherNDOp>();
  patterns.add<ConvertGatherNDOp>(typeConverter, context);
}
