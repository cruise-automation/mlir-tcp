//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir-tcp/Dialect/IR/TcpEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir-tcp/Dialect/IR/TcpTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-tcp/Dialect/IR/TcpAttrs.h.inc"

#define GET_OP_CLASSES
#include "mlir-tcp/Dialect/IR/TcpOps.h.inc"
