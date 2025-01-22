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

#include "cnpy.h"
#include "gtest/gtest.h"

using namespace mlir::tcp;

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

template <typename DataType, int Rank>
static StridedMemRefType<DataType, Rank>
CreateMemRefFromNpyArray(cnpy::NpyArray &arr) {
  StridedMemRefType<DataType, Rank> Result;
  Result.basePtr = arr.data<DataType>();
  Result.data = arr.data<DataType>();
  Result.offset = 0;

  // Check if the Rank matches
  if (arr.shape.size() != Rank) {
    std::cerr << "Error: Rank mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Check if the DataType matches
  if (arr.word_size != sizeof(DataType)) {
    std::cerr << "Error: Data type mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Set sizes and strides based on the shape of the numpy array
  int stride = 1;
  for (int i = Rank - 1; i >= 0; --i) {
    Result.sizes[i] = arr.shape[i];
    Result.strides[i] = stride;
    stride *= arr.shape[i];
  }

  return Result;
}

// CreateMemRefFromNpyArray function specialized for rank 0
template <typename DataType>
static StridedMemRefType<DataType, 0>
CreateMemRefFromNpyArray(cnpy::NpyArray &arr) {
  StridedMemRefType<DataType, 0> Result;
  Result.basePtr = arr.data<DataType>();
  Result.data = arr.data<DataType>();
  Result.offset = 0;

  // Check if the Rank matches
  if (!arr.shape.empty()) {
    std::cerr << "Error: Rank mismatch. Expected rank-0 array." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Check if the DataType matches
  if (arr.word_size != sizeof(DataType)) {
    std::cerr << "Error: Data type mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  return Result;
}

// ### DO NOT MODIFY ### //
// This template file is pre-processed by `aot_compile` bazel macro
// to materialize the templated parameters based on the inputs
// passed by the callsite where the macro is instantiated.

struct OutputMemRefDescriptor {
  // ##OUTPUT_MEMREF_VARIABLE_DECLARATIONS##//
  //  StridedMemRefType<float, 2> Output0;
};

extern "C" OutputMemRefDescriptor func_main(
    // ##INPUT_MEMREF_ABI_DECLARATIONS##//
    //  DECL_RANK_2_MEMREF_ABI(float),
    //  DECL_RANK_2_MEMREF_ABI(float),
    //  DECL_RANK_2_MEMREF_ABI(float)
);

TEST(AotCompiled, ExecuteTest) {

  cnpy::npz_t reference_tensors = cnpy::npz_load(
      "//##REFERENCE_TENSORS_PATH##//"
      // "test/AotCompile/_internal_add_mul_single_output_reference_tensors.npz"
  );

  // ##READ_REFERENCE_TENSORS_INTO_NPY_ARRAY##//
  //  cnpy::NpyArray refInput0 = reference_tensors["Input0"];
  //  cnpy::NpyArray refInput1 = reference_tensors["Input1"];
  //  cnpy::NpyArray refInput2 = reference_tensors["Input2"];
  //  cnpy::NpyArray refOutput0 = reference_tensors["Output0"];

  // ##CREATE_MEMREF_FROM_NPY_ARRAY##//
  //  StridedMemRefType<float, 2> Input0 =
  //      CreateMemRefFromNpyArray<float, 2>(refInput0);
  //  StridedMemRefType<float, 2> Input1 =
  //      CreateMemRefFromNpyArray<float, 2>(refInput1);
  //  StridedMemRefType<float, 2> Input2 =
  //      CreateMemRefFromNpyArray<float, 2>(refInput2);

  OutputMemRefDescriptor Result = func_main(
      // ##PASS_INPUT_MEMREF_ARGUMENTS##//
      //  PASS_RANK_2_MEMREF(Input0),
      //  PASS_RANK_2_MEMREF(Input1),
      //  PASS_RANK_2_MEMREF(Input2)
  );

  // ##ASSERT_RESULT_SHAPE_MATCHES_REFERENCE##//
  //  ASSERT_EQ(Result.Output0.sizes[0], refOutput0.shape[0]);
  //  ASSERT_EQ(Result.Output0.sizes[1], refOutput0.shape[1]);

  // ##EXPECT_RESULT_DATA_MATCHES_REFERENCE##//
  //  for (int i = 0; i < refOutput0.num_vals; i++)
  //    EXPECT_FLOAT_EQ(Result.Output0.data[i], refOutput0.data<float>()[i]);

  // ##DEALLOCATE_RESULT_MEMREF##//
  //  free(Result.Output0.basePtr);
}
