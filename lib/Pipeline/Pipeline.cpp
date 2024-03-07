//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Pipeline/Pipeline.h"

#include "mlir-tcp/Dialect/Transforms/VerifyTcpBackendContractPass.h"

#include "mlir-tcp/Conversion/TcpToArith/TcpToArith.h"
#include "mlir-tcp/Conversion/TcpToLinalg/TcpToLinalg.h"
#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcp.h"
#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcpCustomOp.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;

static void createTorchBackendToTcpBackendPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(tcp::createConvertTorchToTcpPass());
  pm.addNestedPass<func::FuncOp>(tcp::createConvertTorchToTcpCustomOpPass());

  // Clean up any non-canonical code introduced above.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // TCP backend contract.
  pm.addPass(torch::TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      torch::TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that TCP backend expects.
  // This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(tcp::createVerifyTcpBackendContractPass());
}

static void createTcpToLlvmPipeline(OpPassManager &pm) {
  pm.addPass(tcp::createConvertTcpToLinalgPass());
  pm.addPass(tcp::createConvertTcpToArithPass());
  pm.addPass(func::createFuncBufferizePass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addNestedPass<func::FuncOp>(tensor::createTensorBufferizePass());
  pm.addNestedPass<func::FuncOp>(
      bufferization::createBufferizationBufferizePass());
  pm.addPass(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToCFPass());

  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void tcp::registerTcpPipelines() {
  PassPipelineRegistration<>(
      "torch-backend-to-tcp-backend-pipeline",
      "Pipeline lowering torch backend contract to TCP backend contract.",
      createTorchBackendToTcpBackendPipeline);

  PassPipelineRegistration<>("tcp-to-llvm-pipeline", "Lowers TCP to LLVM",
                             createTcpToLlvmPipeline);
}
