load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def third_party_deps():
    LLVM_COMMIT = "4acc3ffbb0af5631bc7916aeff3570f448899647"
    LLVM_SHA256 = "7c5a640383e220dcf16e41a717b5e7d589c29598d31ae304ebc81b73b3be5fd2"
    http_archive(
        name = "llvm-raw",
        build_file_content = "# empty",
        sha256 = LLVM_SHA256,
        strip_prefix = "llvm-project-" + LLVM_COMMIT,
        urls = ["https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT)],
    )

    TORCH_MLIR_COMMIT = "3d974ed9883eac4c3651ac7799a49da5ad9c597b"
    TORCH_MLIR_SHA256 = "71aec7c30d72604325ffe275f36bef8df2476dd11d7bf502aaf14f72011ea7f9"
    http_archive(
        name = "torch-mlir-raw",
        sha256 = TORCH_MLIR_SHA256,
        build_file_content = "# empty",
        strip_prefix = "torch-mlir-" + TORCH_MLIR_COMMIT,
        urls = ["https://github.com/llvm/torch-mlir/archive/{commit}.tar.gz".format(commit = TORCH_MLIR_COMMIT)],
    )

    STABLEHLO_COMMIT = "77a59815a82b34f7b08ed2d42a711d9920682d0e"
    STABLEHLO_SHA256 = "367ac567bc9a543ec3c9bbf16e1304a174b1d42bdb7bdeab2ce8b20134ed68d2"
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
        urls = ["https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip"],
        strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
    )
