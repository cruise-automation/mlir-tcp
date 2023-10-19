# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Modify this file to get bazel to use a local source checkout for the 3p deps
# (e.g. to quickly test out local changes).
#
# To avoid accidentally checking in modifications to this file, you can ask git
# to ignore changes to this file by running the following:
#
# $ git update-index --assume-unchanged local_repos.bzl

def use_local_llvm_repo():
    # Change this to return True to have mlir-tcp use the source tree at
    # `local_llvm_repo_path()`
    return False

def use_local_torch_mlir_repo():
    # Change this to return True to have mlir-tcp use the source tree at
    # `local_torch_mlir_repo_path()`
    return False

def use_local_stablehlo_repo():
    # Change this to return True to have mlir-tcp use the source tree at
    # `local_stablehlo_repo_path()`
    return False

def local_llvm_repo_path():
    return "./third_party/llvm-project"

def local_torch_mlir_repo_path():
    return "./third_party/torch-mlir"

def local_stablehlo_repo_path():
    return "./third_party/stablehlo"
