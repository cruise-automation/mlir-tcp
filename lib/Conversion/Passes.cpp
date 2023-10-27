//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Conversion/Passes.h"

#include "mlir-tcp/Conversion/StablehloToTcp/StablehloToTcp.h"
#include "mlir-tcp/Conversion/TcpToArith/TcpToArith.h"
#include "mlir-tcp/Conversion/TcpToLinalg/TcpToLinalg.h"
#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcp.h"
#include "mlir-tcp/Conversion/TorchToTcp/TorchToTcpCruiseInternal.h"

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "mlir-tcp/Conversion/Passes.h.inc"
} // end namespace

void mlir::tcp::registerTcpConversionPasses() { ::registerPasses(); }
