//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
namespace tcp {
using IndexTy = long;

// An MLIR function with type
//
//   (tensor<?xf32>, tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?x?xf32>)
//
// Will have the ABI:
//
//   Output func(DECL_RANK_1_MEMREF_ABI(float), DECL_RANK_2_MEMREF_ABI(float))
//
// where Output is defined as:
//
// struct Output {
//   RankedMemref<float, 2> A;
//   RankedMemref<float, 3> B;
// };

template <typename DataType, int Rank> struct RankedMemref {
  // Described at https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
  DataType *AllocatedPointer;
  DataType *AlignedPointer;
  IndexTy Offset;
  IndexTy Sizes[Rank];
  IndexTy Strides[Rank];
};

#define DECL_RANK_2_MEMREF_ABI(data_type)                                      \
  data_type *, data_type *, IndexTy, IndexTy, IndexTy, IndexTy, IndexTy
#define DECL_RANK_1_MEMREF_ABI(data_type)                                      \
  data_type *, data_type *, IndexTy, IndexTy, IndexTy
#define DECL_RANK_0_MEMREF_ABI(data_type) data_type *, data_type *, IndexTy

// Helper macros that unpack a memref into a sequence of arguments suitable for
// passing to an AOT compiled function.

#define PASS_RANK_2_MEMREF(memref)                                             \
  (memref).AllocatedPointer, (memref).AlignedPointer, (memref).Offset,         \
      (memref).Sizes[0], (memref).Sizes[1], (memref).Strides[0],               \
      (memref).Strides[1]
#define PASS_RANK_1_MEMREF(memref)                                             \
  (memref).AllocatedPointer, (memref).AlignedPointer, (memref).Offset,         \
      (memref).Sizes[0], (memref).Strides[0],
#define PASS_RANK_0_MEMREF(memref)                                             \
  (memref).AllocatedPointer, (memref).AlignedPointer, (memref).Offset

} // namespace tcp
} // namespace mlir
