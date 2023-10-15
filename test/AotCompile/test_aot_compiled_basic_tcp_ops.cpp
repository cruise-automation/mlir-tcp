//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "tools/aot/abi.h"

#include "gtest/gtest.h"

using namespace mlir::tcp;

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" RankedMemref<float, 2> func_1(DECL_RANK_2_MEMREF_ABI(float),
                                         DECL_RANK_2_MEMREF_ABI(float),
                                         DECL_RANK_2_MEMREF_ABI(float));

static RankedMemref<float, 2> CreateRank2Memref(float *Ptr) {
  RankedMemref<float, 2> Result;
  Result.AllocatedPointer = Ptr;
  Result.AlignedPointer = Ptr;
  Result.Offset = 0;
  Result.Sizes[0] = 2;
  Result.Sizes[1] = 3;
  Result.Strides[0] = 3;
  Result.Strides[1] = 1;

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

  RankedMemref<float, 2> Input1 = CreateRank2Memref(&Arr1[0][0]);
  RankedMemref<float, 2> Input2 = CreateRank2Memref(&Arr2[0][0]);
  RankedMemref<float, 2> Input3 = CreateRank2Memref(&Arr3[0][0]);

  RankedMemref<float, 2> Result =
      func_1(PASS_RANK_2_MEMREF(Input1), PASS_RANK_2_MEMREF(Input2),
             PASS_RANK_2_MEMREF(Input3));

  ASSERT_EQ(Result.Sizes[0], 2);
  ASSERT_EQ(Result.Sizes[1], 3);
  ASSERT_EQ(Result.Strides[0], 3);
  ASSERT_EQ(Result.Strides[1], 1);

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      float Expected = (5 + (2 + i)) * (3 + j);
      EXPECT_EQ(Result.AlignedPointer[3 * i + j], Expected);
    }

  free(Result.AllocatedPointer);
}

struct TwoMemRefs {
  RankedMemref<float, 2> A;
  RankedMemref<float, 2> B;
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

  RankedMemref<float, 2> Input1 = CreateRank2Memref(&Arr1[0][0]);
  RankedMemref<float, 2> Input2 = CreateRank2Memref(&Arr2[0][0]);
  RankedMemref<float, 2> Input3 = CreateRank2Memref(&Arr3[0][0]);

  TwoMemRefs Result =
      func_2(PASS_RANK_2_MEMREF(Input1), PASS_RANK_2_MEMREF(Input2),
             PASS_RANK_2_MEMREF(Input3));

  ASSERT_EQ(Result.A.Sizes[0], 2);
  ASSERT_EQ(Result.A.Sizes[1], 3);
  ASSERT_EQ(Result.A.Strides[0], 3);
  ASSERT_EQ(Result.A.Strides[1], 1);

  ASSERT_EQ(Result.B.Sizes[0], 2);
  ASSERT_EQ(Result.B.Sizes[1], 3);
  ASSERT_EQ(Result.B.Strides[0], 3);
  ASSERT_EQ(Result.B.Strides[1], 1);

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 3; j++) {
      float ExpectedA = 5 + (2 + i);
      EXPECT_EQ(Result.A.AlignedPointer[3 * i + j], ExpectedA);

      float ExpectedB = ExpectedA * (3 + j);
      EXPECT_EQ(Result.B.AlignedPointer[3 * i + j], ExpectedB);
    }

  free(Result.A.AllocatedPointer);
  free(Result.B.AllocatedPointer);
}

extern "C" RankedMemref<float, 1> func_3(DECL_RANK_0_MEMREF_ABI(float),
                                         DECL_RANK_1_MEMREF_ABI(float));

TEST(AotCompiled, MixedRanks) {
  float Arr0 = 10.0;
  float Arr1[2] = {1.0, 2.0};

  RankedMemref<float, 1> Result = func_3(&Arr0, &Arr0, 0, Arr1, Arr1, 0, 2, 1);

  EXPECT_EQ(Result.Sizes[0], 2);
  EXPECT_EQ(Result.Strides[0], 1);
  EXPECT_EQ(Result.AlignedPointer[0], 11.0);
  EXPECT_EQ(Result.AlignedPointer[1], 12.0);
}
