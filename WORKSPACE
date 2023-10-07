workspace(name = "tensor_compute_primitives_ws")

load("//:deps.bzl", "third_party_deps")

third_party_deps()

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
load("@torch-mlir-raw//utils/bazel:configure.bzl", "torch_mlir_configure")

llvm_configure(
    name = "llvm-project",
    targets = [
        "X86",
        "NVPTX",
        "AArch64",
    ],
)

torch_mlir_configure(name = "torch-mlir")
