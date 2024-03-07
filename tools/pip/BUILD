# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_python//python:pip.bzl", "compile_pip_requirements")

# bazel run //tools/pip:requirements.update
compile_pip_requirements(
    name = "requirements",
    requirements_in = "//:requirements.txt",
    requirements_txt = "//:requirements_lock.txt",
    tags = ["manual"],
)
