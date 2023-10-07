load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

td_library(
    name = "TcpTdFiles",
    srcs = [
        "include/Dialect/Tcp/IR/TcpBase.td",
        "include/Dialect/Tcp/IR/TcpEnums.td",
        "include/Dialect/Tcp/IR/TcpOps.td",
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
            "include/Dialect/Tcp/IR/TcpEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "include/Dialect/Tcp/IR/TcpEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/Tcp/IR/TcpOps.td",
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
            "include/Dialect/Tcp/IR/TcpAttrs.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "include/Dialect/Tcp/IR/TcpAttrs.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/Tcp/IR/TcpOps.td",
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
            "include/Dialect/Tcp/IR/TcpOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/Dialect/Tcp/IR/TcpOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=tcp",
            ],
            "include/Dialect/Tcp/IR/TcpDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=tcp",
            ],
            "include/Dialect/Tcp/IR/TcpDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/Tcp/IR/TcpOps.td",
    deps = [
        ":TcpTdFiles",
    ],
)

cc_library(
    name = "TcpDialect",
    srcs = [
        "lib/Dialect/Tcp/IR/TcpDialect.cpp",
        "lib/Dialect/Tcp/IR/TcpOps.cpp",
    ],
    hdrs = [
        "include/Dialect/Tcp/IR/TcpDialect.h",
        "include/Dialect/Tcp/IR/TcpOps.h",
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
        "include/Dialect/Tcp/Transforms/Passes.td",
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
            "include/Dialect/Tcp/Transforms/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/Dialect/Tcp/Transforms/Passes.td",
    deps = [
        ":TcpTransformsPassesTdFiles",
    ],
)

cc_library(
    name = "TcpPasses",
    srcs = [
        "lib/Dialect/Tcp/Transforms/FuseTcpOpsPass.cpp",
        "lib/Dialect/Tcp/Transforms/IsolateGroupOpsPass.cpp",
        "lib/Dialect/Tcp/Transforms/PassDetail.h",
        "lib/Dialect/Tcp/Transforms/Passes.cpp",
    ],
    hdrs = [
        "include/Dialect/Tcp/Transforms/FuseTcpOpsPass.h",
        "include/Dialect/Tcp/Transforms/IsolateGroupOpsPass.h",
        "include/Dialect/Tcp/Transforms/Passes.h",
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
        ":TcpToLinalg",
        ":TcpToArith",
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
        "@torch-mlir//:TorchMLIRConversionUtils",
        ":TcpDialect",
        "@torch-mlir//:TorchMLIRTorchBackendTypeConversion",
        "@torch-mlir//:TorchMLIRTorchConversionDialect",
        "@llvm-project//mlir:Dialect",
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
        "@torch-mlir//:TorchMLIRTorchDialect",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:DialectUtils",
        "@llvm-project//mlir:IR",
    ],
)

cc_binary(
    name = "tcp-opt",
    srcs = [
        "tools/tcp-opt/tcp-opt.cpp",
    ],
    deps = [
        ":TcpInitAll",
        ":TcpDialect",
        ":TcpPasses",
        "@stablehlo//:register",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:QuantOps",
    ],
)
