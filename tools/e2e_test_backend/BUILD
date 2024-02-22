# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_python//python:defs.bzl", "py_library")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "tcp_backend",
    srcs = [
        "abc.py",
        "refbackend.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch_mlir"),
    ],
)
