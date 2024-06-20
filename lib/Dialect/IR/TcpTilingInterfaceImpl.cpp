#include "mlir-tcp/Dialect/IR/TcpTilingInterfaceImpl.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/Support/Debug.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tcp;

namespace {

struct SliceOpTiling
    : public TilingInterface::ExternalModel<SliceOpTiling, tcp::SliceOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    // The iterator types describe the type of the loops in the
    // loop-nest for the eventual tiled op. For the slice op, these are
    // just parallel loops.
    auto sliceOp = cast<tcp::SliceOp>(op);
    return SmallVector(sliceOp.getType().getRank(),
                       utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    // The iteration domain describes the loop bounds for the tiled
    // operation.
    auto sliceOp = cast<tcp::SliceOp>(op);
    OpFoldResult zero = b.getIndexAttr(0);
    OpFoldResult one = b.getIndexAttr(1);
    SmallVector<Range> loopRanges(sliceOp.getType().getRank(),
                                  {zero, one, one});
    for (auto [idx, size] : llvm::enumerate(sliceOp.getSizes())) {
      loopRanges[idx].offset = zero;
      loopRanges[idx].size = size;
    }
    return loopRanges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    // getTiledImplementation and generateResultTileValue are the same
    // because:
    // 1. `getIterationDomain` returns a loop with zero offsets
    //     and the result slice length
    // 2. `getResultTilePosition` returns the same offsets and sizes as
    //     the input offsets and slices.
    // In other words, the body of the tiled implementation is
    // computing the output at exactly offsets/sizes.
    return generateResultTileValue(op, b, 0, offsets, sizes);
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {

    auto sliceOp = cast<tcp::SliceOp>(op);
    auto loc = op->getLoc();

    // TODO: How to handle strides from the sliceOp?
    // Do we need to error out when strides != 1?
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);

    auto fold = [&](OpFoldResult ofr) {
      return getValueOrCreateConstantIndexOp(b, loc, ofr);
    };
    auto add = [&](OpFoldResult val1, OpFoldResult val2) {
      return b.createOrFold<arith::AddIOp>(loc, fold(val1), fold(val2));
    };
    auto mul = [&](OpFoldResult val1, OpFoldResult val2) {
      return b.createOrFold<arith::MulIOp>(loc, fold(val1), fold(val2));
    };

    auto sliceStart = sliceOp.getStarts();
    auto sliceStrides = sliceOp.getStrides();
    SmallVector<OpFoldResult> newOffsets;
    for (auto [offset, start, stride] :
         llvm::zip(offsets, sliceStart, sliceStrides)) {
      newOffsets.push_back(add(start, mul(offset, stride)));
    }

    auto extractOp = b.create<tensor::ExtractSliceOp>(
        loc, sliceOp.getIn(), newOffsets, sizes, strides);

    // This extra redundant `tensor.extract_slice` is needed because the tiling
    // algorithms expect that `generateResultTileValue` will return an
    // op whose operands operate on slices of the original
    // inputs. Without this, we'd need to make a small modification to
    // the core tile and fuse algorithm in MLIR.
    SmallVector<OpFoldResult> zeroOffsets(offsets.size(),
                                          b.getI64IntegerAttr(0));
    auto returnSliceOp = b.create<tensor::ExtractSliceOp>(
        loc, extractOp.getResult(), zeroOffsets, sizes, strides);

    return TilingResult{{returnSliceOp},
                        SmallVector<Value>(returnSliceOp->getResults())};
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
};

} // namespace

void mlir::tcp::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, TcpDialect *dialect) {
    tcp::SliceOp::attachInterface<SliceOpTiling>(*ctx);
  });
}
