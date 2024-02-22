# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_python//python:defs.bzl", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "basic_test",
    srcs = ["fx_import/basic_test.py"],
    tags = ["python_tests"],
    deps = [
        requirement("torch"),
        requirement("torch_mlir"),
    ],
)

py_test(
    name = "torchscript_resnet18_test",
    srcs = ["refbackend/torchscript_resnet18_test.py"],
    tags = ["python_tests"],
    deps = [
        requirement("torch"),
        requirement("torch_mlir"),
        requirement("torchvision"),
        requirement("pillow"),
        requirement("requests"),
    ],
)

test_suite(
    name = "python_tests",
    tags = ["python_tests"],
)
