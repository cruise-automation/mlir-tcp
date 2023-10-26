//===------------------------------------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-tcp/Dialect/IR/TcpDialect.h"
#include "mlir-tcp/Dialect/IR/TcpOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::tcp;

void TcpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-tcp/Dialect/IR/TcpOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tcp/Dialect/IR/TcpTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-tcp/Dialect/IR/TcpAttrs.cpp.inc"
      >();
}

#include "mlir-tcp/Dialect/IR/TcpEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir-tcp/Dialect/IR/TcpTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir-tcp/Dialect/IR/TcpAttrs.cpp.inc"

#include "mlir-tcp/Dialect/IR/TcpDialect.cpp.inc"

Attribute TcpDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef attrKind;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &attrKind, type, attr);
  if (result.has_value())
    return attr;

  parser.emitError(parser.getNameLoc(), "unknown Tcp attribute");
  return Attribute();
}

void TcpDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
  llvm_unreachable("unhandled Tcp attribute kind");
}
