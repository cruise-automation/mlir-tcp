# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

# Used by `aot_compile` bazel macro
exports_files([
    "torch_exporter_harness.py",
    "execute_test_generator.py",
    "execute_test.template.cpp",
])

py_library(
    name = "torch_loader_utils",
    srcs = ["torch_loader_utils.py"],
    deps = [requirement("torch")],
)

cc_library(
    name = "abi",
    hdrs = ["abi.h"],
)

# Dummy target for clangd compilation database purposes only.
# This specific target is not used by the `aot_compile` bazel
# macro, but an equivalent target is.
cc_test(
    name = "execute_test_template",
    srcs = ["execute_test.template.cpp"],
    tags = ["manual"],
    deps = [
        ":abi",
        "@cnpy",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)
