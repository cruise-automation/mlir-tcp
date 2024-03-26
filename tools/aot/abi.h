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
//   StridedMemRefType<float, 2> A;
//   StridedMemRefType<float, 3> B;
// };
//
// and StridedMemRefType is defined as:
// https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
// https://sourcegraph.com/github.com/llvm/llvm-project@b5048700fc31f3bf6dd32ace7730815d4cfef411/-/blob/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h?L131
//
// template <typename T, int N>
// struct StridedMemRefType {
//   T *basePtr;
//   T *data;
//   int64_t offset;
//   int64_t sizes[N];
//   int64_t strides[N];
//   ...
// };

#define DECL_RANK_3_MEMREF_ABI(data_type)                                      \
  data_type *, data_type *, IndexTy, IndexTy, IndexTy, IndexTy, IndexTy,       \
      IndexTy, IndexTy
#define DECL_RANK_2_MEMREF_ABI(data_type)                                      \
  data_type *, data_type *, IndexTy, IndexTy, IndexTy, IndexTy, IndexTy
#define DECL_RANK_1_MEMREF_ABI(data_type)                                      \
  data_type *, data_type *, IndexTy, IndexTy, IndexTy
#define DECL_RANK_0_MEMREF_ABI(data_type) data_type *, data_type *, IndexTy

// Helper macros that unpack a memref into a sequence of arguments suitable for
// passing to an AOT compiled function.

#define PASS_RANK_3_MEMREF(memref)                                             \
  (memref).basePtr, (memref).data, (memref).offset, (memref).sizes[0],         \
      (memref).sizes[1], (memref).sizes[2], (memref).strides[0],               \
      (memref).strides[1], (memref).strides[2]
#define PASS_RANK_2_MEMREF(memref)                                             \
  (memref).basePtr, (memref).data, (memref).offset, (memref).sizes[0],         \
      (memref).sizes[1], (memref).strides[0], (memref).strides[1]
#define PASS_RANK_1_MEMREF(memref)                                             \
  (memref).basePtr, (memref).data, (memref).offset, (memref).sizes[0],         \
      (memref).strides[0]
#define PASS_RANK_0_MEMREF(memref)                                             \
  (memref).basePtr, (memref).data, (memref).offset

} // namespace tcp
} // namespace mlir
