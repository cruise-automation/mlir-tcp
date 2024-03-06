# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

def aot_compile(
        name,
        tcp_source = None,
        torch_loader_lib = None,
        torch_loader_path = "",
        skip_ci = False):
    """
    AOT compile Torch or TCP programs to a CPU library and execute it to
    validate functional correctness of the compiled code against PyTorch
    semantics.

    Exposes a target named `aot_compiled_${name}` that has one global function
    for every function in the TCP program (when `tcp_source` is specified), or
    one global function corresponding to the PyTorch program's forward function
    (when `torch_loader_lib` is specified).

    The functions in Torch or TCP sources must always consume and return tensors.
    The ABI of the generated code is exposed in `abi.h`.

    Parameters
    ----------
    name
        Name of the program to be AOT compiled.
    tcp_source
        Path to the "*.mlir" source containing the TCP program.
    torch_loader_lib
        Label of the `py_library` target for the torch_loader module containing
        the PyTorch program.
    torch_loader_path
        Full python import path (dot separated) to the torch_loader function.
    skip_ci
        When `True`, skip execute tests from CI (and `bazel test //...` expansions).

    Generated Targets
    -----------------
    An invocation of `aot_compile(name="foo", ...)` generates the following targets:
        aot_compiled_foo:
            cc_library wrapper around the AOT compiled assembly source targeting CPU.
            This has one global function for every function in the TCP program
            (when `tcp_source` is specified), or one global function corresponding to
            the PyTorch program's forward function (when `torch_loader_lib` is specified).
            When built, generates a shared object that can by dynamically linked into
            an executable at runtime.
        foo_compile_execute_test:
            cc_test that executes the compiled code on CPU using reference inputs,
            and validates the outputs against PyTorch.
        foo_torch_exporter:
            py_binary that runs the torch_loader function to get the `TorchLoaderOutput`
            (containing the PyTorch program and inputs), then calls the upstream
            `fx.export_and_import` API to generate Torch dialect, and finally runs the
            PyTorch program on reference inputs and saves the reference outputs (as .npz)
            which will eventually be used for validation of the AOT compiled code.
        foo_execute_test_generator:
            py_binary that reads the reference tensors to infer the function signature
            (rank, element type for each input/output tensor) and then materializes the
            templatized parameters in `execute_test.template.cpp`.
        gen_foo_mlir_torch:
            genrule that invokes `foo_torch_exporter` and saves the torch dialect program
            (*_torch.mlir).
        gen_foo_mlir_tcp:
            genrule that invokes `tcp-opt` to convert the torch dialect program to the
            tcp dialect program (*_tcp.mlir) using `-torch-backend-to-tcp-backend-pipeline`.
        gen_foo_mlir_llvm:
            genrule that invokes `tcp-opt` to convert the tcp dialect program to the
            llvm dialect program (*_llvm.mlir) using `-tcp-to-llvm-pipeline`.
        gen_foo_llvm_ir:
            genrule that invokes `mlir-translate` to convert the llvm dialect program to
            the llvm assembly (*.ll) using `-mlir-to-llvmir`.
        gen_foo_host_asm:
            genrule that invokes `llc` on the llvm assembly to generate assembly source
            (*.S) for the host architecture (CPU).
        gen_foo_reference_tensors:
            genrule that invokes `foo_torch_exporter` and saves the reference tensors
            to a numpy archive (*.npz).
        gen_foo_execute_test:
            genrule that invokes `foo_execute_test_generator` to generate a materialized
            execute_test.cpp for foo.

    The set of auto-generated targets can be obtained by running the following query:
        bazel query 'attr(name, "foo", //test/AotCompile/...)'

    """
    if not tcp_source and not torch_loader_lib:
        fail("aot_compile macro requires either `tcp_source` or `torch_loader_lib` " +
             "to be specified.")
    if tcp_source and torch_loader_lib:
        fail("aot_compile macro cannot accept both `tcp_source` and `torch_loader_lib`. " +
             "Please specify either one.")
    if torch_loader_lib != None and torch_loader_path == "":
        fail("aot_compile macro requires `torch_loader_path` to be specified along with " +
             "`torch_loader_lib`.")
    if tcp_source and torch_loader_path != "":
        fail("aot_compile macro cannot accept `torch_loader_path` when `tcp_source` " +
             "is specified.")

    _name = "_internal_" + name

    # Use torch_export based compilation if tcp_source is not specified
    if not tcp_source:
        torch_exporter = name + "_torch_exporter"
        reference_tensors_file = _name + "_reference_tensors.npz"

        py_binary(
            name = torch_exporter,
            srcs = ["//tools/aot:torch_exporter_harness.py"],
            main = "torch_exporter_harness.py",
            deps = [
                torch_loader_lib,
                requirement("numpy"),
                requirement("torch"),
                requirement("torch-mlir"),
                "//tools/aot:torch_loader_utils",
            ],
            # This is needed for testing the binary standalone
            args = ["--torch_loader_path=" + torch_loader_path],
        )

        native.genrule(
            name = "gen_" + name + "_reference_tensors",
            srcs = [],
            outs = [reference_tensors_file],
            cmd = "./$(location " + torch_exporter + ")" +
                  " --torch_loader_path=" + torch_loader_path +
                  " --reference_tensors_path=$(location " + reference_tensors_file + ")",
            tools = [torch_exporter],
        )

        native.genrule(
            name = "gen_" + name + "_mlir_torch",
            srcs = [],
            outs = [_name + "_torch.mlir"],
            cmd = "./$(location " + torch_exporter + ")" +
                  " --torch_loader_path=" + torch_loader_path +
                  " > $(OUTS)",
            tools = [torch_exporter],
        )

        native.genrule(
            name = "gen_" + name + "_mlir_tcp",
            srcs = [_name + "_torch.mlir"],
            outs = [_name + "_tcp.mlir"],
            cmd = "./$(location //:tcp-opt)" +
                  " -torch-backend-to-tcp-backend-pipeline $(SRCS)" +
                  " > $(OUTS)",
            tools = ["//:tcp-opt"],
        )

    native.genrule(
        name = "gen_" + name + "_mlir_llvm",
        # When tcp_source is provided, prefer that as the start for aot_compile;
        # else continue using genrule generated *_tcp.mlir (torch_export workflow)
        srcs = [tcp_source or (_name + "_tcp.mlir")],
        outs = [_name + "_llvm.mlir"],
        cmd = "./$(location //:tcp-opt)" +
              " -tcp-to-llvm-pipeline $(SRCS)" +
              " > $(OUTS)",
        tools = ["//:tcp-opt"],
    )

    native.genrule(
        name = "gen_" + name + "_llvm_ir",
        srcs = [_name + "_llvm.mlir"],
        outs = [_name + ".ll"],
        cmd = "./$(location @llvm-project//mlir:mlir-translate)" +
              " -mlir-to-llvmir $(SRCS)" +
              " > $(OUTS)",
        tools = ["@llvm-project//mlir:mlir-translate"],
    )

    # TODO: Replace llc with clang for `.o` generation
    native.genrule(
        name = "gen_" + name + "_host_asm",
        srcs = [_name + ".ll"],
        outs = [_name + ".S"],
        cmd = "./$(location @llvm-project//llvm:llc) -O3 < $(SRCS)" +
              " > $(OUTS)",
        tools = ["@llvm-project//llvm:llc"],
    )

    cc_library(
        name = "aot_compiled_" + name,
        srcs = [_name + ".S"],
        # Can only be consumed (depended on) by test targets.
        # Prevents inadvertent use in a production usecase.
        testonly = True,
    )

    # Can't use auto-generated tests for tcp_source based compilations due to
    # lack of reference inputs/outputs for comparisons; write tests manually.
    if not tcp_source:
        execute_test_generator = name + "_execute_test_generator"
        test_template_file = "//tools/aot:execute_test.template.cpp"

        py_binary(
            name = execute_test_generator,
            srcs = ["//tools/aot:execute_test_generator.py"],
            main = "execute_test_generator.py",
            deps = [requirement("numpy")],
            # This is needed for testing the binary standalone
            args = [
                "--test_template_path=$(location " + test_template_file + ")",
                "--reference_tensors_path=$(location " + reference_tensors_file + ")",
            ],
            data = [
                test_template_file,
                reference_tensors_file,
            ],
        )

        native.genrule(
            name = "gen_" + name + "_execute_test",
            srcs = [
                test_template_file,
                reference_tensors_file,
            ],
            outs = [_name + "_execute_test.cpp"],
            cmd = "./$(location " + execute_test_generator + ")" +
                  " --test_template_path=$(location " + test_template_file + ")" +
                  " --reference_tensors_path=$(location " + reference_tensors_file + ")" +
                  " > $(OUTS)",
            tools = [execute_test_generator],
        )

        cc_test(
            name = name + "_compile_execute_test",
            srcs = [_name + "_execute_test.cpp"],
            tags = [
                "aot_tests",
                "manual" if skip_ci else "",
            ],
            deps = [
                ":aot_compiled_" + name,
                "//tools/aot:abi",
                "@cnpy//:cnpy",
                "@com_google_googletest//:gtest_main",
                "@llvm-project//mlir:mlir_c_runner_utils_hdrs",
            ],
            data = [reference_tensors_file],
        )
