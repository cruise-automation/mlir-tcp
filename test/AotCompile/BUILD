# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("//tools/aot:aot_compile.bzl", "aot_compile")
load("@rules_cc//cc:defs.bzl", "cc_test")

aot_compile(
    name = "basic_tcp_ops",
    tcp_source = "basic_tcp_ops.mlir",
)

cc_test(
    name = "test_aot_compiled_basic_tcp_ops",
    srcs = ["test_aot_compiled_basic_tcp_ops.cpp"],
    tags = ["aot_tests"],
    deps = [
        ":aot_compiled_basic_tcp_ops",
        "//tools/aot:abi",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "aot_tests",
    tags = ["aot_tests"],
)
