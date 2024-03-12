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

exports_files([
    "requirements.txt",
    "requirements_lock.txt",
])

td_library(
    name = "TcpDialectTdFiles",
    srcs = [
        "include/mlir-tcp/Dialect/IR/TcpBase.td",
        "include/mlir-tcp/Dialect/IR/TcpEnums.td",
        "include/mlir-tcp/Dialect/IR/TcpOps.td",
        "include/mlir-tcp/Dialect/IR/TcpTypes.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "TcpOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/mlir-tcp/Dialect/IR/TcpOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/mlir-tcp/Dialect/IR/TcpOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tcp",
            ],
            "include/mlir-tcp/Dialect/IR/TcpDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tcp",
            ],
            "include/mlir-tcp/Dialect/IR/TcpDialect.cpp.inc",
        ),
        (
            ["-gen-attrdef-decls"],
            "include/mlir-tcp/Dialect/IR/TcpAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/mlir-tcp/Dialect/IR/TcpAttrs.cpp.inc",
        ),
        (
            ["-gen-enum-decls"],
            "include/mlir-tcp/Dialect/IR/TcpEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/mlir-tcp/Dialect/IR/TcpEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tcp/Dialect/IR/TcpOps.td",
    deps = [":TcpDialectTdFiles"],
)

gentbl_cc_library(
    name = "TcpTypesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "include/mlir-tcp/Dialect/IR/TcpTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/mlir-tcp/Dialect/IR/TcpTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tcp/Dialect/IR/TcpTypes.td",
    deps = [":TcpDialectTdFiles"],
)

cc_library(
    name = "TcpDialect",
    srcs = [
        "lib/Dialect/IR/TcpDialect.cpp",
        "lib/Dialect/IR/TcpOps.cpp",
    ],
    hdrs = [
        "include/mlir-tcp/Dialect/IR/TcpDialect.h",
        "include/mlir-tcp/Dialect/IR/TcpOps.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpOpsIncGen",
        ":TcpTypesIncGen",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:QuantOps",
    ],
)

gentbl_cc_library(
    name = "TcpDialectPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/mlir-tcp/Dialect/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tcp/Dialect/Transforms/Passes.td",
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "TcpDialectPasses",
    srcs = [
        "lib/Dialect/Transforms/FuseTcpOpsPass.cpp",
        "lib/Dialect/Transforms/FusionPatterns.cpp",
        "lib/Dialect/Transforms/IsolateGroupOpsPass.cpp",
        "lib/Dialect/Transforms/PassDetail.h",
        "lib/Dialect/Transforms/Passes.cpp",
        "lib/Dialect/Transforms/TransformTensorOps.cpp",
        "lib/Dialect/Transforms/VerifyTcpBackendContractPass.cpp",
    ],
    hdrs = [
        "include/mlir-tcp/Dialect/Transforms/FuseTcpOpsPass.h",
        "include/mlir-tcp/Dialect/Transforms/FusionPatterns.h",
        "include/mlir-tcp/Dialect/Transforms/IsolateGroupOpsPass.h",
        "include/mlir-tcp/Dialect/Transforms/Passes.h",
        "include/mlir-tcp/Dialect/Transforms/TransformTensorOps.h",
        "include/mlir-tcp/Dialect/Transforms/VerifyTcpBackendContractPass.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpDialect",
        ":TcpDialectPassesIncGen",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "TcpConversionPassesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/mlir-tcp/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/mlir-tcp/Conversion/Passes.td",
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "TcpConversionPasses",
    srcs = ["lib/Conversion/Passes.cpp"],
    hdrs = ["include/mlir-tcp/Conversion/Passes.h"],
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
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/TorchToTcp/DataMovement.cpp",
        "lib/Conversion/TorchToTcp/Elementwise.cpp",
        "lib/Conversion/TorchToTcp/Misc.cpp",
        "lib/Conversion/TorchToTcp/PopulatePatterns.h",
        "lib/Conversion/TorchToTcp/TcpCustomOp.cpp",
        "lib/Conversion/TorchToTcp/TorchToTcp.cpp",
        "lib/Conversion/TorchToTcp/TorchToTcpCustomOp.cpp",
        "lib/Conversion/TorchToTcp/Utils.cpp",
        "lib/Conversion/TorchToTcp/Utils.h",
    ],
    hdrs = [
        "include/mlir-tcp/Conversion/TorchToTcp/TorchToTcp.h",
        "include/mlir-tcp/Conversion/TorchToTcp/TorchToTcpCustomOp.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPassesIncGen",
        ":TcpDialect",
        "@llvm-project//mlir:Dialect",
        "@torch-mlir//:TorchMLIRConversionUtils",
        "@torch-mlir//:TorchMLIRTorchBackendTypeConversion",
        "@torch-mlir//:TorchMLIRTorchConversionDialect",
        "@torch-mlir//:TorchMLIRTorchPasses",
        "@torch-mlir//:TorchMLIRTorchToLinalg",
    ],
)

cc_library(
    name = "StablehloToTcp",
    srcs = [
        "lib/Conversion/PassDetail.h",
        "lib/Conversion/StablehloToTcp/StablehloToTcp.cpp",
    ],
    hdrs = ["include/mlir-tcp/Conversion/StablehloToTcp/StablehloToTcp.h"],
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
    hdrs = ["include/mlir-tcp/Conversion/TcpToLinalg/TcpToLinalg.h"],
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
    hdrs = ["include/mlir-tcp/Conversion/TcpToArith/TcpToArith.h"],
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
    srcs = ["lib/InitAll.cpp"],
    hdrs = ["include/mlir-tcp/InitAll.h"],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPasses",
        ":TcpDialectPasses",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
        "@torch-mlir//:TorchMLIRTorchDialect",
    ],
)

cc_library(
    name = "Pipeline",
    srcs = ["lib/Pipeline/Pipeline.cpp"],
    hdrs = ["include/mlir-tcp/Pipeline/Pipeline.h"],
    strip_include_prefix = "include",
    deps = [
        ":TcpConversionPasses",
        ":TcpDialectPasses",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:Pass",
        "@torch-mlir//:TorchMLIRTorchConversionPasses",
    ],
)

cc_binary(
    name = "tcp-opt",
    srcs = ["tools/tcp-opt/tcp-opt.cpp"],
    deps = [
        ":Pipeline",
        ":TcpDialect",
        ":TcpDialectPasses",
        ":TcpInitAll",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
        "@stablehlo//:register",
    ],
)
