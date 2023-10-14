# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

td_library(
    name = "TcpTdFiles",
    srcs = [
        "include/Dialect/IR/TcpBase.td",
        "include/Dialect/IR/TcpEnums.td",
        "include/Dialect/IR/TcpOps.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TcpEnumsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "include/Dialect/IR/TcpEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/Dialect/IR/TcpEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/IR/TcpOps.td",
    deps = [
        ":TcpTdFiles",
    ],
)

gentbl_cc_library(
    name = "TcpAttrsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "include/Dialect/IR/TcpAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/Dialect/IR/TcpAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/IR/TcpOps.td",
    deps = [
        ":TcpTdFiles",
    ],
)

gentbl_cc_library(
    name = "TcpOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/Dialect/IR/TcpOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/Dialect/IR/TcpOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tcp",
            ],
            "include/Dialect/IR/TcpDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tcp",
            ],
            "include/Dialect/IR/TcpDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/IR/TcpOps.td",
    deps = [
        ":TcpTdFiles",
    ],
)

cc_library(
    name = "TcpDialect",
    srcs = [
        "lib/Dialect/IR/TcpDialect.cpp",
        "lib/Dialect/IR/TcpOps.cpp",
    ],
    hdrs = [
        "include/Dialect/IR/TcpDialect.h",
        "include/Dialect/IR/TcpOps.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpAttrsIncGen",
        ":TcpEnumsIncGen",
        ":TcpOpsIncGen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:QuantOps",
    ],
)

td_library(
    name = "TcpTransformsPassesTdFiles",
    srcs = [
        "include/Dialect/Transforms/Passes.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "TcpTransformsPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/Dialect/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/Transforms/Passes.td",
    deps = [
        ":TcpTransformsPassesTdFiles",
    ],
)

cc_library(
    name = "TcpPasses",
    srcs = [
        "lib/Dialect/Transforms/FuseTcpOpsPass.cpp",
        "lib/Dialect/Transforms/IsolateGroupOpsPass.cpp",
        "lib/Dialect/Transforms/PassDetail.h",
        "lib/Dialect/Transforms/Passes.cpp",
    ],
    hdrs = [
        "include/Dialect/Transforms/FuseTcpOpsPass.h",
        "include/Dialect/Transforms/IsolateGroupOpsPass.h",
        "include/Dialect/Transforms/Passes.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpDialect",
        ":TcpTransformsPassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

td_library(
    name = "TcpConversionPassesTdFiles",
    srcs = [
        "include/Conversion/Passes.td",
    ],
    includes = ["include"],
)

gentbl_cc_library(
    name = "TcpConversionPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
            ],
            "include/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Conversion/Passes.td",
    deps = [
        ":TcpConversionPassesTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TcpConversionPasses",
    srcs = [
        "lib/Conversion/Passes.cpp",
    ],
    hdrs = [
        "include/Conversion/Passes.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":StablehloToTcp",
        ":TcpToArith",
        ":TcpToLinalg",
        ":TorchToTcp",
    ],
)

cc_library(
    name = "TorchToTcp",
    srcs = glob([
        "lib/Conversion/*.h",
        "lib/Conversion/TorchToTcp/*.h",
        "lib/Conversion/TorchToTcp/*.cpp",
    ]),
    hdrs = glob(["include/Conversion/TorchToTcp/*.h"]),
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPassesIncGen",
        ":TcpDialect",
        "@llvm-project//mlir:Dialect",
        "@torch-mlir//:TorchMLIRConversionUtils",
        "@torch-mlir//:TorchMLIRTorchBackendTypeConversion",
        "@torch-mlir//:TorchMLIRTorchConversionDialect",
    ],
)

cc_library(
    name = "StablehloToTcp",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/StablehloToTcp/StablehloToTcp.cpp",
    ],
    hdrs = [
        "include/Conversion/StablehloToTcp/StablehloToTcp.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPassesIncGen",
        ":TcpDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "TcpToLinalg",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/TcpToLinalg/Elementwise.cpp",
        "lib/Conversion/TcpToLinalg/Misc.cpp",
        "lib/Conversion/TcpToLinalg/PopulatePatterns.h",
        "lib/Conversion/TcpToLinalg/TcpToLinalg.cpp",
    ],
    hdrs = [
        "include/Conversion/TcpToLinalg/TcpToLinalg.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPassesIncGen",
        ":TcpDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TcpToArith",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/TcpToArith/TcpToArith.cpp",
    ],
    hdrs = [
        "include/Conversion/TcpToArith/TcpToArith.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPassesIncGen",
        ":TcpDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "TcpInitAll",
    srcs = [
        "lib/InitAll.cpp",
    ],
    hdrs = [
        "include/InitAll.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPasses",
        ":TcpPasses",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@torch-mlir//:TorchMLIRTorchDialect",
    ],
)

cc_library(
    name = "Pipeline",
    srcs = [
        "lib/Pipeline/Pipeline.cpp",
    ],
    hdrs = [
        "include/Pipeline/Pipeline.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPasses",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:Pass",
    ],
)

cc_binary(
    name = "tcp-opt",
    srcs = [
        "tools/tcp-opt/tcp-opt.cpp",
    ],
    deps = [
        ":Pipeline",
        ":TcpDialect",
        ":TcpInitAll",
        ":TcpPasses",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@stablehlo//:register",
    ],
)

load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

buildifier(
    name = "buildifier",
)
