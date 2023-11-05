//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcpCustomOp.h"

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace mlir {

#define GEN_PASS_DEF_CONVERTTORCHTOTCPCUSTOMOP
#include "mlir-tcp/Conversion/Passes.h.inc"

namespace tcp {

namespace {

class ConvertTorchToTcpCustomOp
    : public ConvertTorchToTcpCustomOpBase<ConvertTorchToTcpCustomOp> {
private:
  llvm::StringSet<> convertTorchOpsSet;

public:
  ConvertTorchToTcpCustomOp() = default;
  ConvertTorchToTcpCustomOp(ArrayRef<std::string> convertTorchOps) {
    this->convertTorchOps = convertTorchOps;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tcp::TcpDialect>();
    registry.insert<tensor::TensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Usually the default constructor is called which means `convertTorchOps`
    // is usually unset. Doing this here allows the initialization of
    // `convertTorchOpsSet` to be be delayed to when `runOnOperation` is called.
    convertTorchOpsSet.clear();
    convertTorchOpsSet.insert(convertTorchOps.begin(), convertTorchOps.end());

    ConversionTarget target(*context);
    target.addLegalDialect<tcp::TcpDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    torch_to_tcp::populateTcpCustomOpPatternsAndLegality(
        typeConverter, patterns, target, convertTorchOpsSet);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToTcpCustomOpPass() {
  llvm::ArrayRef<std::string> emptyArrayRef;
  return std::make_unique<ConvertTorchToTcpCustomOp>(
      /*convertTorchOps=*/emptyArrayRef);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToTcpCustomOpPass(
    llvm::ArrayRef<std::string> convertTorchOps) {
  return std::make_unique<ConvertTorchToTcpCustomOp>(convertTorchOps);
}

} // namespace tcp
} // namespace mlir
