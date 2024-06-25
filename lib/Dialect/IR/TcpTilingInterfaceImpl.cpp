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

Value getOpFoldResultAsValue(OpFoldResult ofr, OpBuilder &b,
                             mlir::Location loc) {
  // Case 1: Check for Value.
  if (auto val = llvm::dyn_cast_if_present<Value>(ofr)) {
    return val;
  }
  // Case 2: Check for IntegerAttr.
  Attribute attr = llvm::dyn_cast_if_present<Attribute>(ofr);
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  assert(intAttr && "Expected to find an integer in OpFoldResult");
  return b.create<arith::ConstantIndexOp>(loc,
                                          intAttr.getValue().getSExtValue());
}

SmallVector<Value> getOpFoldResultsAsValues(ArrayRef<OpFoldResult> ofrs,
                                            OpBuilder &b, mlir::Location loc) {
  SmallVector<Value> vals = llvm::map_to_vector(ofrs, [&](OpFoldResult ofr) {
    return getOpFoldResultAsValue(ofr, b, loc);
  });
  return vals;
}

struct SliceOpTiling
    : public TilingInterface::ExternalModel<SliceOpTiling, tcp::SliceOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto sliceOp = cast<tcp::SliceOp>(op);
    return SmallVector(sliceOp.getType().getRank(),
                       utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    // The iteration domain describes the loop bounds for the tiled operation.
    // For a slice op, we keep the loop bounds same as the output of the slice.
    // We can compute the indices of the input tensor from the iteration indices
    // of the output.
    //
    // So, the iteration domain is always [0, size, 1] for every dimension.
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
    return generateResultTileValue(op, b, 0, offsets, sizes);
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {

    auto sliceOp = cast<tcp::SliceOp>(op);
    auto loc = op->getLoc();

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
      // Offset on the input of slice is computed by:
      //   start + offset_on_output * stride
      newOffsets.push_back(add(start, mul(offset, stride)));
    }

    auto extractOp = b.create<tensor::ExtractSliceOp>(
        loc, sliceOp.getIn(), newOffsets, sizes,
        getAsOpFoldResult(sliceOp.getStrides()));

    // Add a `tcp.slice` op on the tile.
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> zeroOffsets(offsets.size(), zero.getResult());
    auto returnSliceOp = b.create<tcp::SliceOp>(
        loc, extractOp.getType(), extractOp.getResult(), zeroOffsets,
        getOpFoldResultsAsValues(sizes, b, loc), sliceOp.getStrides());

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
