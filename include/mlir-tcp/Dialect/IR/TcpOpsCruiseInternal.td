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

def Tcp_CustomOp : Tcp_Op<"custom_op", []> {
    let summary = "Custom TCP opqaue operation";

    let description = [{
        Custom opaque operations fill the gap in lowering operations that have tensor inputs and outputs,
        but lack a clear operational semantics or have a an operational semantic that isn't define by TCP.
        e.g user define custom operations.
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs,
        StrAttr:$op_name
    );

    let results = (outs
        Variadic<AnyType>:$outputs
    );

    let assemblyFormat = "`(`$op_name`)` $inputs attr-dict `:` type($inputs) `->` type($outputs)";
}


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


#endif // TCP_OPS_CRUISE_INTERNAL
