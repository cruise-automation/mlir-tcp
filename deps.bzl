load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def third_party_deps():
    LLVM_COMMIT = "28b27c1b10ae8d1f5b4fb9df691e8cf0da9be3f6"
    LLVM_SHA256 = "1f7a7ca5983801d671901644659c32d028e5e7316418fabcb6159454249aefa3"
    http_archive(
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )

    TORCH_MLIR_COMMIT = "52abae1526e51ae8c415ca98ce4a56b00782b68b"
    TORCH_MLIR_SHA256 = "f9973f3519b4ba98475917eb700f447b65fee88e9dd60c61f174ce38335ccb3b"
    http_archive(
        name = "torch-mlir-raw",
        sha256 = TORCH_MLIR_SHA256,
        build_file_content = "# empty",
        strip_prefix = "torch-mlir-" + TORCH_MLIR_COMMIT,
        urls = ["https://github.com/llvm/torch-mlir/archive/{commit}.tar.gz".format(commit = TORCH_MLIR_COMMIT)],
    )

    STABLEHLO_COMMIT = "5a8bb985f50a679721292b14f97f270344ac64a3"
    STABLEHLO_SHA256 = "abda3e8e029c1409b53b1eea080e5cfb4c4ef6705064d7cd954d8272d059567a"
    http_archive(
        name = "stablehlo",
        sha256 = STABLEHLO_SHA256,
        strip_prefix = "stablehlo-" + STABLEHLO_COMMIT,
        urls = ["https://github.com/openxla/stablehlo/archive/{commit}.tar.gz".format(commit = STABLEHLO_COMMIT)],
        # This patch allows testing stablehlo from mlir-tcp
        patches = ["@//:stablehlo.patch"],
        patch_args = ["-p1"],
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
        urls = ["https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip"],
        strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
    )
