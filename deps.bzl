# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(
    ":local_repos.bzl",
    "local_llvm_repo_path",
    "local_stablehlo_repo_path",
    "local_torch_mlir_repo_path",
    "use_local_llvm_repo",
    "use_local_stablehlo_repo",
    "use_local_torch_mlir_repo",
)

def third_party_deps():
    if use_local_llvm_repo():
        native.new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = local_llvm_repo_path(),
        )
    else:
        LLVM_COMMIT = "f9031f00f2c90bc0af274b45ec3e169b5250a688"
        LLVM_SHA256 = "695896a5c3b4011043c27a43ac17079ee725e2d72e3b664868f09626a187daf3"
        http_archive(
            name = "llvm-raw",
            build_file_content = "# empty",
            sha256 = LLVM_SHA256,
            strip_prefix = "llvm-project-" + LLVM_COMMIT,
            urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
        )

    if use_local_torch_mlir_repo():
        native.new_local_repository(
            name = "torch-mlir-raw",
            build_file_content = "# empty",
            path = local_torch_mlir_repo_path(),
        )
    else:
        TORCH_MLIR_COMMIT = "98e08023bbf71e00ab81e980eac9f7c96f1f24b4"
        TORCH_MLIR_SHA256 = "b3bac5863c73dc97bde06a032a928fe0fe586c44c8005bdbb3edd9def056ef0d"
        http_archive(
            name = "torch-mlir-raw",
            build_file_content = "# empty",
            sha256 = TORCH_MLIR_SHA256,
            strip_prefix = "torch-mlir-" + TORCH_MLIR_COMMIT,
            urls = ["https://github.com/llvm/torch-mlir/archive/{commit}.tar.gz".format(commit = TORCH_MLIR_COMMIT)],
        )

    if use_local_stablehlo_repo():
        native.local_repository(
            name = "stablehlo",
            path = local_stablehlo_repo_path(),
        )
    else:
        STABLEHLO_COMMIT = "c28d55e91b4a5daaff18a33ce7e9bbd0f171256a"
        STABLEHLO_SHA256 = "b7c18132c092ec60604e8869685a745a2b31fc1edfc64caf5b19ddce0f1cee60"
        http_archive(
            name = "stablehlo",
            sha256 = STABLEHLO_SHA256,
            strip_prefix = "stablehlo-" + STABLEHLO_COMMIT,
            urls = ["https://github.com/openxla/stablehlo/archive/{commit}.tar.gz".format(commit = STABLEHLO_COMMIT)],
        )

    SKYLIB_VERSION = "1.3.0"
    http_archive(
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
            "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        ],
    )

    http_archive(
        name = "llvm_zstd",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = [
            "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
        ],
    )

    http_archive(
        name = "llvm_zlib",
        build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = [
            "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
        ],
    )

    http_archive(
        name = "com_google_googletest",
        sha256 = "b976cf4fd57b318afdb1bdb27fc708904b3e4bed482859eb94ba2b4bdd077fe2",
        urls = ["https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip"],
        strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
    )

    RULES_PYTHON_VERSION = "0.29.0"
    RULES_PYTHON_SHA256 = "d71d2c67e0bce986e1c5a7731b4693226867c45bfe0b7c5e0067228a536fc580"
    http_archive(
        name = "rules_python",
        sha256 = RULES_PYTHON_SHA256,
        strip_prefix = "rules_python-{}".format(RULES_PYTHON_VERSION),
        url = "https://github.com/bazelbuild/rules_python/releases/download/{}/rules_python-{}.tar.gz".format(RULES_PYTHON_VERSION, RULES_PYTHON_VERSION),
    )

    http_archive(
        name = "io_bazel_rules_go",
        sha256 = "6dc2da7ab4cf5d7bfc7c949776b1b7c733f05e56edc4bcd9022bb249d2e2a996",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
            "https://github.com/bazelbuild/rules_go/releases/download/v0.39.1/rules_go-v0.39.1.zip",
        ],
    )

    http_archive(
        name = "bazel_gazelle",
        sha256 = "727f3e4edd96ea20c29e8c2ca9e8d2af724d8c7778e7923a854b2c80952bc405",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.30.0/bazel-gazelle-v0.30.0.tar.gz",
        ],
    )

    http_archive(
        name = "com_google_protobuf",
        sha256 = "3bd7828aa5af4b13b99c191e8b1e884ebfa9ad371b0ce264605d347f135d2568",
        strip_prefix = "protobuf-3.19.4",
        urls = [
            "https://github.com/protocolbuffers/protobuf/archive/v3.19.4.tar.gz",
        ],
    )

    http_archive(
        name = "com_github_bazelbuild_buildtools",
        sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
        strip_prefix = "buildtools-4.2.2",
        urls = [
            "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
        ],
    )

    http_archive(
        name = "hedron_compile_commands",
        sha256 = "2188c3cd3a16404a6b20136151b37e7afb5a320e150453750c15080de5ba3058",
        strip_prefix = "bazel-compile-commands-extractor-6d58fa6bf39f612304e55566fa628fd160b38177",
        url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/6d58fa6bf39f612304e55566fa628fd160b38177.tar.gz",
    )

    http_archive(
        name = "cnpy",
        build_file = "//third_party:cnpy.BUILD",
        sha256 = "5120abc54a564efa92c642cc0199cc4fd3f345901157de9fbbdcedbb34d28d8a",
        strip_prefix = "cnpy-4e8810b1a8637695171ed346ce68f6984e585ef4",
        urls = ["https://github.com/rogersce/cnpy/archive/4e8810b1a8637695171ed346ce68f6984e585ef4.tar.gz"],
    )
