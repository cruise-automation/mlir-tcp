# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("//tools/aot:aot_compile.bzl", "aot_compile")
load("@rules_cc//cc:defs.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "model_loader_lib",
    srcs = ["model_loader_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        "//tools/aot:torch_loader_utils",
    ],
)

aot_compile(
    name = "add_mul_single_output",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.add_mul_single_output_loader",
)

aot_compile(
    name = "add_mul_multi_output",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.add_mul_multi_output_loader",
)

aot_compile(
    name = "broadcast_add_mixed_ranks",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.broadcast_add_mixed_ranks_loader",
)

aot_compile(
    name = "sigmoid",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.sigmoid_loader",
)

aot_compile(
    name = "concat_float_tensors",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.concat_float_tensors_loader",
)

aot_compile(
    name = "concat_int_tensors",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.concat_int_tensors_loader",
)

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
