# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("//tools/aot:aot_compile.bzl", "aot_compile")
load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

AOT_TEST_SUITE = [
    "add_mul_single_output",
    "add_mul_multi_output",
    "add_tensor_mixed_ranks",
    "add_tensor_with_alpha",
    "sub_tensor_with_alpha",
    "div_tensor_mixed_ranks",
    "add_sub_mul_div_scalar",
    "sigmoid",
    "tanh",
    "clamp",
    "relu",
    "sqrt_float",
    "sqrt_int",
    "concat_float_tensors",
    "concat_int_tensors",
    "slice_tensor",
]

py_library(
    name = "model_loader_lib",
    srcs = ["model_loader_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        "//tools/aot:torch_loader_utils",
    ],
)

[
    aot_compile(
        name = test,
        torch_loader_lib = ":model_loader_lib",
        torch_loader_path = "test.AotCompile.model_loader_lib.%s_loader" % test,
    )
    for test in AOT_TEST_SUITE
]

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
