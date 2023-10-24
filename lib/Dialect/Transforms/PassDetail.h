//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir-tcp/Dialect/IR/TcpOps.h"

namespace mlir::tcp {

using namespace mlir;
#define GEN_PASS_CLASSES
#include "mlir-tcp/Dialect/Transforms/Passes.h.inc"

} // end namespace mlir::tcp
