// RUN: tcp-opt %s -tcp-to-llvm-pipeline | FileCheck %s

// CHECK-LABEL: llvm.func @main
// CHECK:         llvm.mlir.constant
// CHECK:         llvm.mlir.undef
// CHECK:         llvm.insertvalue
// CHECK:         llvm.extractvalue
// CHECK:         llvm.alloca
// CHECK:         llvm.store
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.mul
// CHECK:         llvm.ptrtoint
// CHECK:         llvm.add
// CHECK:         llvm.call
// CHECK:         llvm.sub
// CHECK:         llvm.urem
// CHECK:         llvm.inttoptr
// CHECK:         llvm.br
// CHECK:         llvm.icmp
// CHECK:         llvm.cond_br
// CHECK:         llvm.load
// CHECK:         llvm.mul
// CHECK:         llvm.add
// CHECK:         llvm.fadd
// CHECK:         llvm.store
// CHECK:         llvm.fmul
// CHECK:       llvm.return
func.func @main(%arg0: tensor<?x?xf32>,
                  %arg1: tensor<?x?xf32>,
                  %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.mul %0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
