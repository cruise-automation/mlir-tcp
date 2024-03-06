# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_python//python:defs.bzl", "py_test")
# No need to depend on hermetic python for internal repo.
# load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "basic_test",
    srcs = ["fx_import/basic_test.py"],
    # Skip python tests for internal repo.
    tags = ["python_tests", "manual"],
    # No need to depend on hermetic python for internal repo.
    # deps = [
        # requirement("torch"),
        # requirement("torch_mlir"),
    # ],
)

test_suite(
    name = "python_tests",
    tags = ["python_tests"],
)
