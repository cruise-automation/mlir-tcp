//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_ENUMS
#define TCP_ENUMS

include "mlir/IR/EnumAttr.td"

include "mlir-tcp/Dialect/IR/TcpBase.td"

// TCP Signedness enum (mirror the mlir::IntegerType::SignednessSemantics enum)
def Tcp_Signedness_Signless : I32EnumAttrCase<"Signless", 0>;
def Tcp_Signedness_Signed   : I32EnumAttrCase<"Signed", 1>;
def Tcp_Signedness_Unsigned : I32EnumAttrCase<"Unsigned", 2>;

def Tcp_Signedness : I32EnumAttr<"Signedness",
    "Signedness info for integer type element",
    [
      Tcp_Signedness_Signless,
      Tcp_Signedness_Signed,
      Tcp_Signedness_Unsigned
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tcp";
}

def Tcp_SignednessAttr : EnumAttr<Tcp_Dialect, Tcp_Signedness, "signedness">;

#endif // TCP_ENUMS
