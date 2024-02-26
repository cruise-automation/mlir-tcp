//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "tools/aot/abi.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::tcp;

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" StridedMemRefType<float, 2> func_1(DECL_RANK_2_MEMREF_ABI(float),
                                              DECL_RANK_2_MEMREF_ABI(float),
                                              DECL_RANK_2_MEMREF_ABI(float));

static StridedMemRefType<float, 2> CreateRank2Memref(float *Ptr) {
  StridedMemRefType<float, 2> Result;
  Result.basePtr = Ptr;
  Result.data = Ptr;
  Result.offset = 0;
  Result.sizes[0] = 2;
  Result.sizes[1] = 3;
  Result.strides[0] = 3;
  Result.strides[1] = 1;

  return Result;
}

TEST(AotCompiled, SingleOutput) {
  float Arr1[2][3];
  float Arr2[2][3];
  float Arr3[2][3];

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      Arr1[i][j] = 5;
      Arr2[i][j] = 2 + i;
      Arr3[i][j] = 3 + j;
    }

  StridedMemRefType<float, 2> Input1 = CreateRank2Memref(&Arr1[0][0]);
  StridedMemRefType<float, 2> Input2 = CreateRank2Memref(&Arr2[0][0]);
  StridedMemRefType<float, 2> Input3 = CreateRank2Memref(&Arr3[0][0]);

  StridedMemRefType<float, 2> Result =
      func_1(PASS_RANK_2_MEMREF(Input1), PASS_RANK_2_MEMREF(Input2),
             PASS_RANK_2_MEMREF(Input3));

  ASSERT_EQ(Result.sizes[0], 2);
  ASSERT_EQ(Result.sizes[1], 3);
  ASSERT_EQ(Result.strides[0], 3);
  ASSERT_EQ(Result.strides[1], 1);

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      float Expected = (5 + (2 + i)) * (3 + j);
      EXPECT_EQ(Result.data[3 * i + j], Expected);
    }

  free(Result.basePtr);
}

struct TwoMemRefs {
  StridedMemRefType<float, 2> A;
  StridedMemRefType<float, 2> B;
};

extern "C" TwoMemRefs func_2(DECL_RANK_2_MEMREF_ABI(float),
                             DECL_RANK_2_MEMREF_ABI(float),
                             DECL_RANK_2_MEMREF_ABI(float));

TEST(AotCompiled, MultiOutput) {
  float Arr1[2][3];
  float Arr2[2][3];
  float Arr3[2][3];

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      Arr1[i][j] = 5;
      Arr2[i][j] = 2 + i;
      Arr3[i][j] = 3 + j;
    }

  StridedMemRefType<float, 2> Input1 = CreateRank2Memref(&Arr1[0][0]);
  StridedMemRefType<float, 2> Input2 = CreateRank2Memref(&Arr2[0][0]);
  StridedMemRefType<float, 2> Input3 = CreateRank2Memref(&Arr3[0][0]);

  TwoMemRefs Result =
      func_2(PASS_RANK_2_MEMREF(Input1), PASS_RANK_2_MEMREF(Input2),
             PASS_RANK_2_MEMREF(Input3));

  ASSERT_EQ(Result.A.sizes[0], 2);
  ASSERT_EQ(Result.A.sizes[1], 3);
  ASSERT_EQ(Result.A.strides[0], 3);
  ASSERT_EQ(Result.A.strides[1], 1);

  ASSERT_EQ(Result.B.sizes[0], 2);
  ASSERT_EQ(Result.B.sizes[1], 3);
  ASSERT_EQ(Result.B.strides[0], 3);
  ASSERT_EQ(Result.B.strides[1], 1);

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      float ExpectedA = 5 + (2 + i);
      EXPECT_EQ(Result.A.data[3 * i + j], ExpectedA);

      float ExpectedB = ExpectedA * (3 + j);
      EXPECT_EQ(Result.B.data[3 * i + j], ExpectedB);
    }

  free(Result.A.basePtr);
  free(Result.B.basePtr);
}

extern "C" StridedMemRefType<float, 1> func_3(DECL_RANK_0_MEMREF_ABI(float),
                                              DECL_RANK_1_MEMREF_ABI(float));

TEST(AotCompiled, MixedRanks) {
  float Arr0 = 10.0;
  float Arr1[2] = {1.0, 2.0};

  StridedMemRefType<float, 1> Result =
      func_3(&Arr0, &Arr0, 0, Arr1, Arr1, 0, 2, 1);

  EXPECT_EQ(Result.sizes[0], 2);
  EXPECT_EQ(Result.strides[0], 1);
  EXPECT_EQ(Result.data[0], 11.0);
  EXPECT_EQ(Result.data[1], 12.0);

  free(Result.basePtr);
}
