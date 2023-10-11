//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "Transforms/Passes.h"
#include "Transforms/FuseTcpOpsPass.h"
#include "Transforms/IsolateGroupOpsPass.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace {
#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"
} // end namespace

void mlir::tcp::registerTcpPasses() { ::registerPasses(); }
