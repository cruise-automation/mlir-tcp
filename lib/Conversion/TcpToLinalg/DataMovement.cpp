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

  LogicalResult matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op->getLoc();
    auto resultTensorType = getTypeConverter()
                                ->convertType(op->getResult(0).getType())
                                .cast<RankedTensorType>();

    auto inputTensor = op->getOperands()[0];
    auto indicesTensor = op->getOperands()[1];

    int64_t gatherDim = op.getDimAttr().getValue().getSExtValue();

    auto resultRank = resultTensorType.getRank();

    SmallVector<Value> resultDimSizes;
    for (int64_t i = 0; i < resultRank; ++i) {
      resultDimSizes.push_back(
          b.createOrFold<tensor::DimOp>(loc, indicesTensor, i));
    }

    SmallVector<AffineMap, 2> indexingMaps;
    indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));
    indexingMaps.push_back(b.getMultiDimIdentityMap(resultRank));

    SmallVector<utils::IteratorType> iteratorTypes(
        resultRank, utils::IteratorType::parallel);

    Value emptyTensor =
        b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(resultDimSizes),
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
    Value generic = b.create<linalg::GenericOp>(
                         loc, emptyTensor.getType(), indicesTensor, emptyTensor,
                         indexingMaps, iteratorTypes, bodyBuilder)
                        .getResult(0);
    b.replaceOp(op, generic);
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
}
