//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_TYPES
#define TCP_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

include "mlir-tcp/Dialect/IR/TcpBase.td"
include "mlir-tcp/Dialect/IR/TcpTypesCruiseInternal.td"

//===----------------------------------------------------------------------===//
// Tcp Quantized Types
//===----------------------------------------------------------------------===//
// The base class of a quantized type.
// Param tuple is: [bitwidth, zero_pt, s_mantissa, s_exp, low_end, high_end].
// Where low and high ends are 0,255 when unsigned, -128,127 when signed, for
// the 8-bit case.
class Tcp_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"$_self.isa<mlir::quant::QuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "int", "uint") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr = !interleave(params, ", ") #
                          !if(signed, ", true", ", false");
}
//===----------------------------------------------------------------------===//
// Name    Symmetry           Sign
//===----------------------------------------------------------------------===//
// q8ua  : asymmetric         unsigned
// q8sa  : asymmetric         signed
// q8ss  : symmetric          signed
// q16ss : symmetric          signed
//===----------------------------------------------------------------------===//
def Tcp_QuantizedInt	: AnyTypeOf<[Tcp_QuantizedType<"q8ua", [8], 0>,
                                     Tcp_QuantizedType<"q8sa", [8], 1>,
                                     Tcp_QuantizedType<"q8ss", [8, 0], 1>,
                                     Tcp_QuantizedType<"q16ss", [16, 0], 1>]>;


//===----------------------------------------------------------------------===//
// Tcp Basic Types
//===----------------------------------------------------------------------===//

def Tcp_Scalar : AnyTypeOf<[AnyFloat, AnySignlessInteger, Tcp_QuantizedInt]>;
def Tcp_Tensor : RankedTensorOf<[Tcp_Scalar]>;
def Tcp_TensorOrScalar : AnyTypeOf<[Tcp_Tensor, Tcp_Scalar]>;

def Tcp_FloatTensor : RankedTensorOf<[AnyFloat]>;
def Tcp_IntTensor : RankedTensorOf<[AnySignlessInteger]>;
def Tcp_FloatOrIntTensor : RankedTensorOf<[AnyFloat, AnySignlessInteger]>;

#endif // TCP_TYPES
