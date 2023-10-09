//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TCP_CONVERSION_PASSES
#define TCP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTorchToTcp : Pass<"convert-torch-to-tcp", "func::FuncOp"> {
  let summary = "Convert Torch ops to Tcp ops";
  let description = [{
    Convert Torch ops to Tcp ops.
  }];
  let constructor = "mlir::torch::createConvertTorchToTcpPass()";
}

//===----------------------------------------------------------------------===//
// StablehloToTcp
//===----------------------------------------------------------------------===//

def ConvertStablehloToTcp
      : Pass<"convert-stablehlo-to-tcp", "func::FuncOp"> {
  let summary = "Lower StableHLO to TCP";
  let description = [{
    Pass that converts StableHLO operations to equivalent operations in TCP.
  }];

  let constructor = "mlir::tcp::createConvertStablehloToTcpPass()";
  let dependentDialects = [
    "mlir::tcp::TcpDialect",
  ];
}

//===----------------------------------------------------------------------===//
// TcpToLinalg
//===----------------------------------------------------------------------===//

def ConvertTcpToLinalg
      : Pass<"convert-tcp-to-linalg", "func::FuncOp"> {
  let summary = "Lower TCP to Linalg";
  let description = [{
    Pass that converts TCP operations to equivalent operations in Linalg.
  }];

  let constructor = "mlir::tcp::createConvertTcpToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

//===----------------------------------------------------------------------===//
// TcpToArith
//===----------------------------------------------------------------------===//

def ConvertTcpToArith
      : Pass<"convert-tcp-to-arith", "func::FuncOp"> {
  let summary = "Lower TCP to Arith";
  let description = [{
    Pass that converts TCP operations to equivalent operations in Arith.
  }];

  let constructor = "mlir::tcp::createConvertTcpToArithPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
}

#endif // TCP_CONVERSION_PASSES
