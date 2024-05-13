//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_OPS_CRUISE_INTERNAL
#define TCP_OPS_CRUISE_INTERNAL

def Tcp_CreateIndexArrayOp : Tcp_Op<"create_index_array", []> {
    let summary = "Creates index_arry type given variadic number of input indices";

    let arguments = (ins
        Variadic<Index>:$inputs
    );

    let results = (outs
        Tcp_IndexArrayType : $outputs
    );
}

def Tcp_BindTensorShape : Tcp_Op<"bind_tensor_shape", []> {
    let summary = "Binds a tensor to its index_array shape type.";

    let arguments = (ins
        AnyType:$tensor,
        Tcp_IndexArrayType : $inputs
    );
}

// caspr CasprCreateTensorFromIndexOp : used for encapsulating result of tensor_dim_op in tensor
def Tcp_CasprCreateTensorFromIndexOp : Tcp_Op<"caspr_create_tensor_from_index"> {
    let arguments = (ins
      Index:$index
    );

    let results = (outs
      AnyType:$result
    );

  let assemblyFormat = [{
    `(` operands `)` `:` attr-dict-with-keyword functional-type(operands, results)
  }];
}

def Tcp_CasprIndexFromTensorOp : Tcp_Op<"caspr_index_from_tensor"> {
    let arguments = (ins
      AnyType:$input
    );

    let results = (outs
      Index:$index
    );

  let assemblyFormat = [{
    `(` operands `)` `:` attr-dict-with-keyword functional-type(operands, results)
  }];
}

#endif // TCP_OPS_CRUISE_INTERNAL
