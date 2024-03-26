# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import re
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Test generator for AOT compiled programs using a test template"
)
parser.add_argument(
    "--test_template_path",
    required=True,
    help="Path to the execute_test.template.cpp file",
)
parser.add_argument(
    "--reference_tensors_path",
    required=True,
    help="Path to the file containing the reference inputs and outputs (.npz)",
)

NUMPY_TO_MEMREF_DTYPE_MAP = {
    # Add more mappings as needed
    "int8": "int8_t",
    "uint8": "uint8_t",
    "int16": "int16_t",
    "uint16": "uint16_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
}

MEMREF_DTYPE_TO_GTEST_ASSERT_MAP = {
    "int8_t": "EXPECT_EQ",
    "uint8_t": "EXPECT_EQ",
    "int16_t": "EXPECT_EQ",
    "uint16_t": "EXPECT_EQ",
    "int32_t": "EXPECT_EQ",
    "uint32_t": "EXPECT_EQ",
    "int64_t": "EXPECT_EQ",
    "uint64_t": "EXPECT_EQ",
    "float": "EXPECT_FLOAT_EQ",
    "double": "EXPECT_DOUBLE_EQ",
}


def main():
    args = parser.parse_args()

    # Track string substitutions
    input_memref_abi_declarations = []
    pass_input_memref_arguments = []
    create_memref_from_npy_array_str = ""
    output_memref_variable_declarations_str = ""
    assert_result_shape_matches_reference_str = ""
    expect_result_data_matches_reference_str = ""
    deallocate_result_memref_str = ""
    read_reference_tensors_into_npy_array_str = ""
    reference_tensors_path_str = args.reference_tensors_path.removeprefix(
        "bazel-out/k8-fastbuild/bin/"
    )

    # Interpret function signature (num_args, rank, dtype, num_returns)
    # from the saved reference tensors and build string substitutions
    reference_tensors = np.load(args.reference_tensors_path)
    for key in reference_tensors.keys():
        tensor = reference_tensors[key]
        rank = tensor.ndim
        dtype = NUMPY_TO_MEMREF_DTYPE_MAP[str(tensor.dtype)]

        if "Input" in key:
            input_memref_abi_declarations.append(
                f"""
    DECL_RANK_{rank}_MEMREF_ABI({dtype})"""
            )
            pass_input_memref_arguments.append(
                f"""
      PASS_RANK_{rank}_MEMREF({key})"""
            )
            if rank == 0:
                create_memref_from_npy_array_str += f"""
  StridedMemRefType<{dtype}, {rank}> {key} =
      CreateMemRefFromNpyArray<{dtype}>(ref{key});"""
            else:
                create_memref_from_npy_array_str += f"""
  StridedMemRefType<{dtype}, {rank}> {key} =
      CreateMemRefFromNpyArray<{dtype}, {rank}>(ref{key});"""

        if "Output" in key:
            output_memref_variable_declarations_str += f"""
  StridedMemRefType<{dtype}, {rank}> {key};"""
            for n in range(rank):
                assert_result_shape_matches_reference_str += f"""
  ASSERT_EQ(Result.{key}.sizes[{n}], ref{key}.shape[{n}]);"""
            expect_result_data_matches_reference_str += f"""
  for (int i = 0; i < ref{key}.num_vals; i++)
    {MEMREF_DTYPE_TO_GTEST_ASSERT_MAP[dtype]}(Result.{key}.data[i], ref{key}.data<{dtype}>()[i]);"""
            deallocate_result_memref_str += f"""
  free(Result.{key}.basePtr);"""

        read_reference_tensors_into_npy_array_str += f"""
  cnpy::NpyArray ref{key} = reference_tensors["{key}"];"""

    # Comma separated, except at the end.
    input_memref_abi_declarations_str = ",".join(input_memref_abi_declarations)
    pass_input_memref_arguments_str = ",".join(pass_input_memref_arguments)

    substitutions = {
        r"//##OUTPUT_MEMREF_VARIABLE_DECLARATIONS##//": output_memref_variable_declarations_str,
        r"//##INPUT_MEMREF_ABI_DECLARATIONS##//": input_memref_abi_declarations_str,
        r"//##REFERENCE_TENSORS_PATH##//": reference_tensors_path_str,
        r"//##PASS_INPUT_MEMREF_ARGUMENTS##//": pass_input_memref_arguments_str,
        r"//##READ_REFERENCE_TENSORS_INTO_NPY_ARRAY##//": read_reference_tensors_into_npy_array_str,
        r"//##CREATE_MEMREF_FROM_NPY_ARRAY##//": create_memref_from_npy_array_str,
        r"//##ASSERT_RESULT_SHAPE_MATCHES_REFERENCE##//": assert_result_shape_matches_reference_str,
        r"//##EXPECT_RESULT_DATA_MATCHES_REFERENCE##//": expect_result_data_matches_reference_str,
        r"//##DEALLOCATE_RESULT_MEMREF##//": deallocate_result_memref_str,
    }

    # Open template test
    with open(args.test_template_path, "r") as test_template_file:
        test_template_source = test_template_file.read()

    # Perform the regex search and replace for each pattern
    for pattern, replacement_string in substitutions.items():
        test_template_source = re.sub(pattern, replacement_string, test_template_source)

    # Important: This print is needed to pipe outputs in aot_compile's genrule
    print(test_template_source)


if __name__ == "__main__":
    main()
