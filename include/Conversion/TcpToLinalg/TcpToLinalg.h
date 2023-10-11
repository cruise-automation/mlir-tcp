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

namespace mlir {

#define GEN_PASS_DECL_CONVERTTCPTOLINALG
#include "Conversion/Passes.h.inc"

namespace tcp {

std::unique_ptr<Pass> createConvertTcpToLinalgPass();

} // namespace tcp
} // namespace mlir
