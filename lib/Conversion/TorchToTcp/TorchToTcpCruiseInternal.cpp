//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcpCruiseInternal.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertTorchToTcpCruiseInternal
    : public ConvertTorchToTcpCruiseInternalBase<
          ConvertTorchToTcpCruiseInternal> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tcp::TcpDialect>();
    registry.insert<tensor::TensorDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<tcp::TcpDialect, tensor::TensorDialect,
                           arith::ArithDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([&](Torch::IndexArrayType) {
      return tcp::IndexArrayType::get(context);
    });
    typeConverter.addConversion(
        [&](Torch::TorchIndexType) { return ::mlir::IndexType::get(context); });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    torch_to_tcp::cruise::populateCruiseInternalPatternsAndLegality(
        typeConverter, patterns, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tcp::createConvertTorchToTcpCruiseInternalPass() {
  return std::make_unique<ConvertTorchToTcpCruiseInternal>();
}
