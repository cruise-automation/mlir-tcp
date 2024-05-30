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

// -----

// CHECK-LABEL: func.func @test_slice(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<1x56x?x?xf32>) -> tensor<1x28x?x?xf32>
// CHECK:           %[[D1:.*]] = tensor.dim %[[ARG0]]
// CHECK:           %[[D2:.*]] = tensor.dim %[[ARG0]]
// CHECK:           %[[SLICE:.*]] = tcp.slice %[[ARG0]](%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) (%{{.*}}, %{{.*}}, %[[D1]], %[[D2]])
// CHECK-SAME:                                (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : tensor<1x56x?x?xf32> -> tensor<1x28x?x?xf32>
// CHECK:           return %[[SLICE]] : tensor<1x28x?x?xf32>
func.func @test_slice(%arg0: tensor<1x56x?x?xf32>) -> tensor<1x28x?x?xf32> {
  %c28 = arith.constant 28 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c2 : tensor<1x56x?x?xf32>
  %dim_0 = tensor.dim %arg0, %c3 : tensor<1x56x?x?xf32>
  %1 = tcp.slice %arg0 ( %c0, %c0, %c0, %c0 ) ( %c1, %c28, %dim, %dim_0 ) ( %c1, %c2, %c1, %c1 ) : tensor<1x56x?x?xf32> -> tensor<1x28x?x?xf32>
  return %1 : tensor<1x28x?x?xf32>
}
