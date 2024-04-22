# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("//tools/aot:aot_compile.bzl", "aot_compile")
load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
# No need to depend on hermetic python for internal repo.
# load("@pip_deps//:requirements.bzl", "requirement")

AOT_TEST_SUITE = [
    # (test_name, skip_ci)
    ("add_mul_single_output", False),
    ("add_mul_multi_output", False),
    ("add_tensor_mixed_ranks", False),
    ("add_tensor_with_alpha", False),
    ("sub_tensor_with_alpha", False),
    ("div_tensor_mixed_ranks", False),
    ("add_sub_mul_div_scalar", False),
    ("sigmoid", False),
    ("tanh", False),
    ("clamp", False),
    ("relu", False),
    ("round_even", False),
    ("sqrt_float", False),
    ("sqrt_int", False),
    ("concat_float_tensors", False),
    ("concat_int_tensors", False),
    ("slice_tensor", False),
    ("broadcast_unit_dim_to_static_with_explicit_dim_static", False),
    ("broadcast_unit_dim_to_static_with_unchanged_dim_static", False),
    ("broadcast_unit_dim_to_static_with_unchanged_dim_dynamic", False),
    ("broadcast_unit_dim_to_dynamic_with_unchanged_dim_static", False),
    ("broadcast_unit_dim_to_dynamic_with_unchanged_dim_dynamic", False),
    ("broadcast_unit_dim_to_static_with_rank_increase", False),
    ("broadcast_unit_dim_to_dynamic_with_rank_increase", False),
]

py_library(
    name = "model_loader_lib",
    srcs = ["model_loader_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        # No need to depend on hermetic python for internal repo.
        # requirement("torch"),
        "//tools/aot:torch_loader_utils",
    ],
)

# Disable aot_compile tests for internal repo.
# [
#     aot_compile(
#         name = test_name,
#         skip_ci = skip_ci,
#         torch_loader_lib = ":model_loader_lib",
#         torch_loader_path = "test.AotCompile.model_loader_lib.%s_loader" % test_name,
#     )
#     for test_name, skip_ci in AOT_TEST_SUITE
# ]

aot_compile(
    name = "basic_tcp_ops",
    tcp_source = "basic_tcp_ops.mlir",
)

cc_test(
    name = "basic_tcp_ops_compile_execute_test",
    srcs = ["basic_tcp_ops_compile_execute_test.cpp"],
    tags = ["aot_tests"],
    deps = [
        ":aot_compiled_basic_tcp_ops",
        "//tools/aot:abi",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:mlir_c_runner_utils_hdrs",
    ],
)

test_suite(
    name = "aot_tests",
    tags = ["aot_tests"],
)
