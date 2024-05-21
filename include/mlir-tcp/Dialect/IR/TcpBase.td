//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_BASE
#define TCP_BASE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Tcp Dialect Base
//===----------------------------------------------------------------------===//

def Tcp_Dialect : Dialect {
  let name = "tcp";
  let cppNamespace = "::mlir::tcp";
  let description = [{
    Tensor Compute Primitives (TCP) dialect.

    TCP is a mid-level transformation oriented IR for deep learning & similar
    applications.
  }];
  // Dialect attribute parsing hook.
  let extraClassDeclaration = [{        
    Attribute parseAttribute(
        DialectAsmParser &parser, Type type) const override;
    void printAttribute(
        Attribute attr, DialectAsmPrinter &printer) const override;
  }]; 
}

//===----------------------------------------------------------------------===//
// Tcp Ops Base
//===----------------------------------------------------------------------===//

class Tcp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tcp_Dialect, mnemonic, traits> {
}

class Tcp_UnaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Tcp_Op<mnemonic, !listconcat(traits, [
        Pure,
        Elementwise,
        SameOperandsAndResultShape])> {
}

class Tcp_BinaryElementwiseOp<string mnemonic, list<Trait> traits = []> :
    Tcp_Op<mnemonic, !listconcat(traits, [
        NoMemoryEffect,
        Elementwise,
        SameOperandsAndResultShape])> {
}

#endif // TCP_BASE
