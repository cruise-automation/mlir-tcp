// RUN: tcp-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @test_gather(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<2x?xi64>) -> tensor<2x?xf32>
// CHECK:         %[[GATHER:.*]] = tcp.gather %[[ARG0]], %[[ARG1]] {dim = 0 : index} : tensor<?x?xf32>, tensor<2x?xi64> -> tensor<2x?xf32>
// CHECK:         return %[[GATHER]] : tensor<2x?xf32>
func.func @test_gather(%arg0 : tensor<?x?xf32>, %arg1 : tensor<2x?xi64>) -> tensor<2x?xf32> {
  %0 = tcp.gather %arg0, %arg1 { dim = 0 : index } :
                tensor<?x?xf32>, tensor<2x?xi64> -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}
