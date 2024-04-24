// RUN: tcp-opt %s -convert-tcp-to-linalg -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @gather
// CHECK-SAME:        %[[ARG0:.+]]: tensor<1x4x3xf32>,
// CHECK-SAME:        %[[ARG1:.+]]: tensor<1x4x2xi64>) -> tensor<1x4x2xf32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x4x2xf32>
// CHECK:         %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:                          ins(%[[ARG1]] : tensor<1x4x2xi64>) outs(%[[EMPTY]] : tensor<1x4x2xf32>)
// CHECK:           ^bb0(%[[IN:.+]]: i64, %[[OUT:.+]]: f32):
// CHECK:           %[[I0:.+]] = linalg.index 0 : index
// CHECK:           %[[I1:.+]] = linalg.index 1 : index
// CHECK:           %[[I2:.+]] = arith.index_cast %[[IN]] : i64 to index
// CHECK:           %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[I0]], %[[I1]], %[[I2]]] : tensor<1x4x3xf32>
// CHECK:           linalg.yield %[[EXTRACT]] : f32
// CHECK:         } -> tensor<1x4x2xf32>
// CHECK:         return %[[GENERIC]] : tensor<1x4x2xf32>
func.func @gather(%arg0 : tensor<1x4x3xf32>, %arg1 : tensor<1x4x2xi64>) -> tensor<1x4x2xf32> {
  %0 = "tcp.gather"(%arg0, %arg1) {dim = 2 : index} : (tensor<1x4x3xf32>, tensor<1x4x2xi64>) -> tensor<1x4x2xf32>
  return %0 : tensor<1x4x2xf32>
}
