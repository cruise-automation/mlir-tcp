AOT Compile (Developer Guide)
=============================

The [`aot_compile`](https://github.com/cruise-automation/mlir-tcp/blob/main/tools/aot/aot_compile.bzl) bazel macro implements an end-to-end framework to compile PyTorch (or TCP) programs to a CPU library, execute it and test for functional correctness of the generated code. It comprises starting with TorchDynamo export of PyTorch programs, conversion and lowerings through {Torch, TCP, Linalg, LLVM} MLIR dialects, translation to LLVM assembly, compilation to assembly source for the host architecture (CPU), and lastly generation of shared object that could be dynamically linked into an executable/test at runtime. It leverages a series of genrules to stitch the compilation pipeline together, and an unsophisticated meta-programming trick for auto-generating C++ tests (specialized to the input program's function signature) that execute the compiled code and validate its numerics against reference PyTorch.

When authoring new TCP ops with dialect conversions from/to Torch and Linalg, adding an `aot_compile` target is a fast, automated and standardized way to test the e2e compilation and validate that the op lowerings are implemented consistent with PyTorch semantics.

Caveat: The AOT compile framework's primary objective is to serve as an end-to-end `compile -> execute -> test` harness for functional correctness, and *not* as an optimizing compiler for production usecases. In the future we might be interested in reusing pieces of infrastructure here to construct an optimizing compiler, but it entails more work to get there (such as a runtime and performance benchmark apparatus).

## Compile PyTorch programs

Onboarding to the `aot_compile` macro is quite easy (examples [here](https://github.com/cruise-automation/mlir-tcp/blob/main/test/AotCompile/BUILD)). Start by adding the following line to the `BUILD` to load the macro:
```starlark
load("//tools/aot:aot_compile.bzl", "aot_compile")
```

Then call the macro like this:
```starlark
aot_compile(
    name = "broadcast_add_mixed_ranks",
    torch_loader_lib = ":model_loader_lib",
    torch_loader_path = "test.AotCompile.model_loader_lib.broadcast_add_mixed_ranks_loader",
)
```

Here, `torch_loader_lib` expects a `py_library` target for the module that defines the PyTorch program to be AOT compiled, and `torch_loader_path` is the full python import path (dot separated) to the loader function.
```starlark
py_library(
    name = "model_loader_lib",
    srcs = ["model_loader_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("torch"),
        "//tools/aot:torch_loader_utils",
    ],
)
```

The loader function can be called anything really, but it should define the PyTorch program, sample inputs and dynamic dim constraints (if any), and always return a `TorchLoaderOutput` object. The PyTorch program's forward function must always consume and return tensors, like so:
```python
import torch
from torch.export import dynamic_dim

from tools.aot.torch_loader_utils import TorchLoaderOutput


def broadcast_add_mixed_ranks_loader() -> TorchLoaderOutput:
    class BroadcastAddMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.add(x, y)
            return add

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2)

    # Dynamic dim constraints
    constraints = [dynamic_dim(y, 0)]

    return TorchLoaderOutput(
        model=BroadcastAddMixedRanks(),
        inputs=[x, y],
        constraints=constraints,
    )
```

An invocation of `aot_compile(name="foo", ...)` generates a bunch of targets (see [here](https://github.com/cruise-automation/mlir-tcp/blob/main/tools/aot/aot_compile.bzl#L43) for the list) that can be helpful in debugging the intermediate steps in the compilation process.

To get the full list of `aot_compile` macro generated targets for `broadcast_add_mixed_ranks`, run the query:
```shell
$ bazel query 'attr(name, "broadcast_add_mixed_ranks", //test/AotCompile/...)'

//test/AotCompile:aot_compiled_broadcast_add_mixed_ranks
//test/AotCompile:broadcast_add_mixed_ranks_compile_execute_test
//test/AotCompile:broadcast_add_mixed_ranks_execute_test_generator
//test/AotCompile:broadcast_add_mixed_ranks_torch_exporter
//test/AotCompile:gen_broadcast_add_mixed_ranks_execute_test
//test/AotCompile:gen_broadcast_add_mixed_ranks_host_asm
//test/AotCompile:gen_broadcast_add_mixed_ranks_llvm_ir
//test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_llvm
//test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_tcp
//test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_torch
//test/AotCompile:gen_broadcast_add_mixed_ranks_reference_tensors
```

### Debugging e2e compilation pipeline

Lets walk through a series of steps involved in debugging an e2e compilation pipeline. Note that these steps are not required to be manually run one at a time (although they can be). Bazel automatically identifies the DAG of dependencies and executes just what is needed to build the specified target.

#### 1. Inspect the Torch dialect (`*_torch.mlir`) exported from the PyTorch program:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_torch

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_torch (61 packages loaded, 16582 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_torch up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_torch.mlir
INFO: Elapsed time: 6.085s, Critical Path: 0.69s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```ll
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_torch.mlir

module {
  func.func @func_main(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> {
    %int1 = torch.constant.int 1
    %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[],f32>, !torch.vtensor<[?],f32>, !torch.int -> !torch.vtensor<[?],f32>
    return %0 : !torch.vtensor<[?],f32>
  }
}
```

#### 2. Inspect the TCP dialect (`*_tcp.mlir`) lowered from the Torch dialect:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_tcp

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_tcp (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_tcp up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_tcp.mlir
INFO: Elapsed time: 0.572s, Critical Path: 0.03s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```ll
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_tcp.mlir

module {
  func.func @func_main(%arg0: tensor<f32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %expanded = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %0 = tcp.broadcast %expanded, %dim {axes = [0]} : tensor<1xf32>, index -> tensor<?xf32>
    %1 = tcp.add %0, %arg1 : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}
```

#### 3. Inspect the LLVM dialect (`*_llvm.mlir`) lowered from the TCP dialect:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_llvm

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_llvm (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_mlir_llvm up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_llvm.mlir
INFO: Elapsed time: 0.305s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```ll
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_llvm.mlir

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @func_main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: !llvm.ptr, %arg5: i64, %arg6: i64, %arg7: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64)>
.
.
.
    %57 = llvm.load %56 : !llvm.ptr -> f32
    %58 = llvm.fadd %54, %57  : f32
    %59 = llvm.getelementptr %44[%51] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %59 : f32, !llvm.ptr
    %60 = llvm.add %51, %14  : i64
    llvm.br ^bb4(%60 : i64)
  ^bb6:  // pred: ^bb4
    llvm.return %50 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}
```

#### 4. Inspect the LLVM assembly (`*.ll`) translated from the LLVM dialect:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_llvm_ir

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_llvm_ir (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_llvm_ir up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks.ll
INFO: Elapsed time: 0.312s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```ll
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks.ll

; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @func_main(ptr %0, ptr %1, i64 %2, ptr %3, ptr %4, i64 %5, i64 %6, i64 %7) {
  %9 = insertvalue { ptr, ptr, i64 } undef, ptr %0, 0
  %10 = insertvalue { ptr, ptr, i64 } %9, ptr %1, 1
  %11 = insertvalue { ptr, ptr, i64 } %10, i64 %2, 2
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %3, 0
.
.
.
63:                                               ; preds = %51
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %50
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

#### 5. Inspect the assembly source (`*.S`) compiled for the host architecture (CPU):
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_host_asm

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_host_asm (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_host_asm up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks.S
INFO: Elapsed time: 0.360s, Critical Path: 0.03s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```ll
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks.S

        .text
        .file   "LLVMDialectModule"
        .globl  func_main                       # -- Begin function func_main
        .p2align        4, 0x90
        .type   func_main,@function
func_main:                              # @func_main
        .cfi_startproc
# %bb.0:
        pushq   %rbp
        .cfi_def_cfa_offset 16
.
.
.
        popq    %r14
        .cfi_def_cfa_offset 24
        popq    %r15
        .cfi_def_cfa_offset 16
        popq    %rbp
        .cfi_def_cfa_offset 8
        retq
.Lfunc_end0:
        .size   func_main, .Lfunc_end0-func_main
        .cfi_endproc
                                        # -- End function
        .section        ".note.GNU-stack","",@progbits
```

#### 6. Build the shared object (`*.so`) from the host assembly that can be dynamically linked into an executable/test at runtime:
```shell
$ bazel build //test/AotCompile:aot_compiled_broadcast_add_mixed_ranks

INFO: Analyzed target //test/AotCompile:aot_compiled_broadcast_add_mixed_ranks (8 packages loaded, 8403 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:aot_compiled_broadcast_add_mixed_ranks up-to-date:
  bazel-bin/test/AotCompile/libaot_compiled_broadcast_add_mixed_ranks.a
  bazel-bin/test/AotCompile/libaot_compiled_broadcast_add_mixed_ranks.so
INFO: Elapsed time: 2.264s, Critical Path: 0.12s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```

Note that this `cc_library` target (called `aot_compiled_*`) is marked `testonly`, which only allows it to be added as a dependency for test targets. The goal is to avoid any inadvertent use of the compiled artifacts in production usecases.

#### 7. Save the reference input and output tensors needed for validation of the compiled code:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_reference_tensors

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_reference_tensors (0 packages loaded, 5 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:gen_broadcast_add_mixed_ranks_reference_tensors up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_reference_tensors.npz
INFO: Elapsed time: 0.743s, Critical Path: 0.15s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```

#### 8. Inspect the C++ test (`*_execute_test.cpp`) auto-generated from the template:
```shell
$ bazel build //test/AotCompile:gen_broadcast_add_mixed_ranks_execute_test

INFO: Analyzed target //test/AotCompile:gen_broadcast_add_mixed_ranks_execute_test (22 packages loaded, 91 targets configured).
INFO: Found 1 target...Target //test/AotCompile:gen_broadcast_add_mixed_ranks_execute_test up-to-date:
  bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_execute_test.cpp
INFO: Elapsed time: 0.329s, Critical Path: 0.02s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
```
```cpp
$ cat bazel-bin/test/AotCompile/_internal_broadcast_add_mixed_ranks_execute_test.cpp

//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "tools/aot/abi.h"

#include "cnpy.h"
#include "gtest/gtest.h"

using namespace mlir::tcp;

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

template <typename DataType, int Rank>
static StridedMemRefType<DataType, Rank>
CreateMemRefFromNpyArray(cnpy::NpyArray &arr) {
  StridedMemRefType<DataType, Rank> Result;
  Result.basePtr = arr.data<DataType>();
  Result.data = arr.data<DataType>();
  Result.offset = 0;

  // Check if the Rank matches
  if (arr.shape.size() != Rank) {
    std::cerr << "Error: Rank mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Check if the DataType matches
  if (arr.word_size != sizeof(DataType)) {
    std::cerr << "Error: Data type mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Set sizes and strides based on the shape of the numpy array
  int stride = 1;
  for (int i = Rank - 1; i >= 0; --i) {
    Result.sizes[i] = arr.shape[i];
    Result.strides[i] = stride;
    stride *= arr.shape[i];
  }

  return Result;
}

// CreateMemRefFromNpyArray function specialized for rank 0
template <typename DataType>
static StridedMemRefType<DataType, 0>
CreateMemRefFromNpyArray(cnpy::NpyArray &arr) {
  StridedMemRefType<DataType, 0> Result;
  Result.basePtr = arr.data<DataType>();
  Result.data = arr.data<DataType>();
  Result.offset = 0;

  // Check if the Rank matches
  if (!arr.shape.empty()) {
    std::cerr << "Error: Rank mismatch. Expected rank-0 array." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  // Check if the DataType matches
  if (arr.word_size != sizeof(DataType)) {
    std::cerr << "Error: Data type mismatch." << std::endl;
    // Return an uninitialized memref
    return Result;
  }

  return Result;
}

// ### DO NOT MODIFY ### //
// This template file is pre-processed by `aot_compile` bazel macro
// to materialize the templated parameters based on the inputs
// passed by the callsite where the macro is instantiated.

struct OutputMemRefDescriptor {
  StridedMemRefType<float, 1> Output0;
};

extern "C" OutputMemRefDescriptor func_main(
    DECL_RANK_0_MEMREF_ABI(float),
    DECL_RANK_1_MEMREF_ABI(float)
);

TEST(AotCompiled, ExecuteTest) {
  cnpy::npz_t reference_tensors = cnpy::npz_load(
      "test/AotCompile/_internal_broadcast_add_mixed_ranks_reference_tensors.npz"
  );

  cnpy::NpyArray refInput0 = reference_tensors["Input0"];
  cnpy::NpyArray refInput1 = reference_tensors["Input1"];
  cnpy::NpyArray refOutput0 = reference_tensors["Output0"];

  StridedMemRefType<float, 0> Input0 =
      CreateMemRefFromNpyArray<float>(refInput0);
  StridedMemRefType<float, 1> Input1 =
      CreateMemRefFromNpyArray<float, 1>(refInput1);

  OutputMemRefDescriptor Result = func_main(
      PASS_RANK_0_MEMREF(Input0),
      PASS_RANK_1_MEMREF(Input1)
  );

  ASSERT_EQ(Result.Output0.sizes[0], refOutput0.shape[0]);

  for (int i = 0; i < refOutput0.num_vals; i++)
    EXPECT_EQ(Result.Output0.data[i], refOutput0.data<float>()[i]);

  free(Result.Output0.basePtr);
}
```

#### 9. Run the C++ test to execute the generated code and validate functional correctness against reference PyTorch
```shell
$ bazel run //test/AotCompile:broadcast_add_mixed_ranks_compile_execute_test

INFO: Analyzed target //test/AotCompile:broadcast_add_mixed_ranks_compile_execute_test (0 packages loaded, 0 targets configured).
INFO: Found 1 target...
Target //test/AotCompile:broadcast_add_mixed_ranks_compile_execute_test up-to-date:
  bazel-bin/test/AotCompile/broadcast_add_mixed_ranks_compile_execute_test
INFO: Elapsed time: 0.215s, Critical Path: 0.00s
INFO: 1 process: 1 internal.
INFO: Build completed successfully, 1 total action
INFO: Running command line: external/bazel_tools/tools/test/test-setup.sh test/AotCompile/broadcast_add_mixed_ranks_compile_execute_test
exec ${PAGER:-/usr/bin/less} "$0" || exit 1
Executing tests from //test/AotCompile:broadcast_add_mixed_ranks_compile_execute_test
-----------------------------------------------------------------------------
Running main() from gmock_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from AotCompiled
[ RUN      ] AotCompiled.ExecuteTest
[       OK ] AotCompiled.ExecuteTest (1 ms)
[----------] 1 test from AotCompiled (1 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1 ms total)
[  PASSED  ] 1 test.
```

## Compile TCP programs

The `aot_compile` macro also accepts TCP dialect programs as inputs (instead of PyTorch programs). This is useful to maintain framework neutrality by allowing alternate ingress pathways (like Stablehlo, JAX, TensorFlow, ONNX etc.) into the TCP dialect. When `tcp_source` is specified, the generated `aot_compiled_foo` CPU library has one global function for every function in the TCP program. Let's look at an example.

```starlark
aot_compile(
    name = "basic_tcp_ops",
    tcp_source = "basic_tcp_ops.mlir",
)
```

Here, `tcp_source` expects a `.mlir` file containing TCP programs, like so:
```mlir
// basic_tcp_ops.mlir

func.func @func_1(%arg0: tensor<?x?xf32>,
                  %arg1: tensor<?x?xf32>,
                  %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.mul %0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func.func @func_2(%arg0: tensor<?x?xf32>,
                  %arg1: tensor<?x?xf32>,
                  %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.mul %0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

func.func @func_3(%arg0: tensor<f32>,
                  %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
  %arg0_ex = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
  %arg0_bcast = tcp.broadcast %arg0_ex, %dim {axes = [0]} : tensor<1xf32>, index -> tensor<?xf32>
  %0 = tcp.add %arg0_bcast, %arg1 : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
```

Now run the query to get all the relevant targets created.
```shell
$ bazel query 'attr(name, "basic_tcp_ops", //test/AotCompile/...)'

//test/AotCompile:aot_compiled_basic_tcp_ops
//test/AotCompile:gen_basic_tcp_ops_host_asm
//test/AotCompile:gen_basic_tcp_ops_llvm_ir
//test/AotCompile:gen_basic_tcp_ops_mlir_llvm
```

Note we're missing the `//test/AotCompile:basic_tcp_ops_compile_execute_test` target. As there is no access to PyTorch reference implementation, the `aot_compile` macro does not auto-generate C++ execute tests but they can be manually written (example [here](https://github.com/cruise-automation/mlir-tcp/blob/main/test/AotCompile/test_aot_compiled_basic_tcp_ops.cpp)). These tests should include `extern "C"` function declarations with the same name and for every function in the input TCP source.

The rest of the steps to debug the e2e compilation pipeline are pretty much the same.
