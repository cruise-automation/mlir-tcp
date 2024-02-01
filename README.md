Tensor Compute Primitives
=========================

Mid-level intermediate representation for machine learning programs.

[![Bazel Build and Test (mlir-tcp)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestTcp.yml/badge.svg)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestTcp.yml)

:construction: **This project is under active development (WIP).**

## Project Communication

- For general discussion use `#mlir-tcp` channel on the [LLVM Discord](https://discord.gg/xS7Z362)
- For feature request or bug report file a detailed [issue on GitHub](https://github.com/cruise-automation/mlir-tcp/issues)

## Developer Guide

To build TCP using Bazel, follow these steps:

1. (Optional) For a quick start, launch an interactive docker container with clang (and lld) pre-installed:
```shell
./docker/run_docker.sh
```

2. You can now build `tcp-opt` by running:
```shell
bazel build //:tcp-opt
```

3. To run TCP lit and aot compile tests:
```shell
bazel test //test/...
```

We welcome contributions to `mlir-tcp`. If you do contribute, please finalize your PR with clang-format and bazel buildifier to ensure the C++ sources and BUILD files are formatted consistently:
```shell
# clang-format
find . -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# buildifer
bazel run //tools/buildifier:buildifier
```

To enable clangd (for code completion, navigation and insights), generate the compilation database using [bazel-compile-commands-extractor](https://github.com/hedronvision/bazel-compile-commands-extractor):
```shell
bazel build //...

bazel run //tools/clangd:refresh_compile_commands
```
When run successfully, a `compile_commands.json` is generated at the workspace root (and refreshed upon re-runs). If you're using VSCode, just hit CMD+SHIFT+P and select `clangd: Restart language server` to start clangd. Note that this only works for non-docker builds at the moment.

When bumping upstream dependencies (LLVM, Torch-MLIR, StableHLO), you may validate the set of "green commits" by running the corresponding third-party tests:
```shell
bazel test @llvm-project//mlir/...
bazel test @torch-mlir//...
bazel test @stablehlo//...
```

The following CI workflows are automatically triggered anytime upstream dependencies (`deps.bzl`) are updated:
- [![Bazel Build and Test (llvm-project)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestLlvm.yml/badge.svg)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestLlvm.yml)
- [![Bazel Build and Test (torch-mlir)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestTorchmlir.yml/badge.svg)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestTorchmlir.yml)
- [![Bazel Build and Test (stablehlo)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestStablehlo.yml/badge.svg)](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestStablehlo.yml)


## Debugging Guide

Below are some standard techniques for debugging your compilation process, assuming you've reduced it to a form that can be reproduced with `tcp-opt`. For MLIR-specific debugging tips, refer [here](https://mlir.llvm.org/getting_started/Debugging/).

### `printf` debugging

Printing to stdout/stderr works as usual:
```C++
op.emitWarning() << "HERE: " << myVariable;      // preferred for op/loc diagnostics

llvm::errs() << "HERE: " << myVariable << "\n";  // alternative
```

You can also hook into the [LLVM_DEBUG](https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option) macro:
```C++
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "foo"
LLVM_DEBUG(llvm::dbgs() << "This only shows up when -debug or -debug-only=foo is provided.\n");
#undef DEBUG_TYPE

#define DEBUG_TYPE "bar"
LLVM_DEBUG(llvm::dbgs() << "This only shows up when -debug or -debug-only=bar is provided.\n");
#undef DEBUG_TYPE
```

Then run with the `-debug-only=foo,bar` flag to cuts out messages that aren't associated with the passed `DEBUG_TYPE`s.
```shell
bazel run //:tcp-opt -- --some-pass `pwd`/test.mlir -debug-only=foo,bar
```

### `gdb` debugging

To debug `tcp-opt` with [gdb](https://www.sourceware.org/gdb/):
```shell
bazel build --config=gdb //:tcp-opt

gdb --args bazel-bin/tcp-opt -h
```

For help with gdb commands please refer to [gdb cheat sheet](https://gist.github.com/rkubik/b96c23bd8ed58333de37f2b8cd052c30).

### Enable `llvm-symbolizer`

If you get a stack dump without any symbol names:
```shell
Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
0  tcp-opt   0x000055ac1c9c0c1d
1  tcp-opt   0x000055ac1c9c110b
2  tcp-opt   0x000055ac1c9be846
3  tcp-opt   0x000055ac1c9c1855
4  libc.so.6 0x00007f7011c6a520
...
```

Do this and re-run:
```shell
bazel build @llvm-project//llvm:llvm-symbolizer
export LLVM_SYMBOLIZER_PATH=`pwd`/bazel-bin/external/llvm-project/llvm/llvm-symbolizer
```
