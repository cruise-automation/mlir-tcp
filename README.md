Tensor Compute Primitives
=========================

Mid-level intermediate representation for machine learning programs.

![Bazel Build](https://github.com/cruise-automation/mlir-tcp/actions/workflows/bazelBuildAndTestTcp.yml/badge.svg)

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
bazel build --config=clang_linux //:tcp-opt
```
(replace `linux` with `osx` for Mac)

3. To run TCP lit and aot compile tests:
```shell
bazel test --config=clang_linux //test/...
```

We welcome contributions to `mlir-tcp`. If you do contribute, please finalize your PR with clang-format and bazel buildifier to ensure the C++ sources and BUILD files are formatted consistently:
```shell
# clang-format
find . -type f -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# buildifer
bazel run --config=clang_linux //:buildifier
```
