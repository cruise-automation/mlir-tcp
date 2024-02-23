# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

workspace(name = "mlir-tcp")

load("//:deps.bzl", "third_party_deps")

third_party_deps()

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    targets = [
        "X86",
        "NVPTX",
        "AArch64",
    ],
)

load("@torch-mlir-raw//utils/bazel:configure.bzl", "torch_mlir_configure")

torch_mlir_configure(name = "torch-mlir")

# -------------------------- #
#    Hermetic Python Setup   #
# -------------------------- #

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python_3_10",
    python_version = "3.10",
)

load("@python_3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    python_interpreter_target = interpreter,
    requirements_lock = "//:requirements_lock.txt",
)

load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

# --------------------------- #
#    Buildifier Dependencies  #
# --------------------------- #

# https://github.com/bazelbuild/buildtools/blob/master/buildifier/README.md

load("@io_bazel_rules_go//go:deps.bzl", "go_rules_dependencies")

go_rules_dependencies()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains")

go_register_toolchains(version = "1.20.3")

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# ------------------------------------ #
#    Bazel Compile Commands Extractor  #
#    for clangd                        #
# ------------------------------------ #

# https://github.com/hedronvision/bazel-compile-commands-extractor/blob/main/README.md

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()
