//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Conversion/Passes.h"

#include "Conversion/StablehloToTcp/StablehloToTcp.h"
#include "Conversion/TcpToArith/TcpToArith.h"
#include "Conversion/TcpToLinalg/TcpToLinalg.h"
#include "Conversion/TorchToTcp/TorchToTcp.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"
} // end namespace

void mlir::tcp::registerConversionPasses() {
  ::registerPasses();
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::tcp::createConvertStablehloToTcpPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::tcp::createConvertTcpToLinalgPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::tcp::createConvertTcpToArithPass();
  });
}
