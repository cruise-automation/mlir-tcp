# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

licenses(["notice"])  # MIT

package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "cnpy",
    srcs = ["cnpy.cpp"],
    hdrs = ["cnpy.h"],
    copts = [
        "-Wno-unused-variable",
    ],
    deps = ["@llvm_zlib//:zlib"],
)

cc_test(
    name = "test_cnpy",
    srcs = ["example1.cpp"],
    deps = [":cnpy"],
)
