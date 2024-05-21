//===-------------------------------------------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_TYPES_CRUISE_INTERNAL
#define TCP_TYPES_CRUISE_INTERNAL

//===----------------------------------------------------------------------===//
// Tcp Custom Types
//===----------------------------------------------------------------------===//

def Tcp_IndexArrayType : Tcp_Type<"IndexArray", "index_array"> {
    let summary = "IndexArray TCP type, to holds a list of index builtin type to represent shape";

    let parameters = (ins
        "int":$elements
    );

    let assemblyFormat = "`<` `[` $elements `]` `>`";
}

#endif // TCP_TYPES_CRUISE_INTERNAL
