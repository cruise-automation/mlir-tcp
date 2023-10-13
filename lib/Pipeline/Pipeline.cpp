//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Pipeline/Pipeline.h"

#include "Conversion/TcpToLinalg/TcpToLinalg.h"
#include "Conversion/TcpToArith/TcpToArith.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static void tcpToLlvmPipelineBuilder(OpPassManager &pm) {
  pm.addPass(tcp::createConvertTcpToLinalgPass());
  pm.addPass(tcp::createConvertTcpToArithPass());
  pm.addPass(func::createFuncBufferizePass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addNestedPass<func::FuncOp>(tensor::createTensorBufferizePass());
  pm.addNestedPass<func::FuncOp>(bufferization::createBufferizationBufferizePass());
  pm.addPass(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToCFPass());

  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void tcp::registerTcpPipelines() {
  PassPipelineRegistration<>(
    "lower-tcp-to-llvm", "Lowers TCP to LLVM", tcpToLlvmPipelineBuilder);
}
