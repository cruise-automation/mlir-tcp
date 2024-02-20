# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Keep this list updated by running the following query:
    #   bazel query 'kind("(cc.*) rule", //...)'
    targets = [
        "//:Pipeline",
        "//:StablehloToTcp",
        "//:TcpConversionPasses",
        "//:TcpConversionPassesIncGen",
        "//:TcpDialect",
        "//:TcpDialectPasses",
        "//:TcpDialectPassesIncGen",
        "//:TcpInitAll",
        "//:TcpOpsIncGen",
        "//:TcpToArith",
        "//:TcpToLinalg",
        "//:TcpTypesIncGen",
        "//:TorchToTcp",
        "//:tcp-opt",
    ],
)
