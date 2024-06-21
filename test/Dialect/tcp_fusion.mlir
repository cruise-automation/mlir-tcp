// RUN: tcp-opt %s -split-input-file -tcp-fuse-elementwise-ops | FileCheck %s

// CHECK-LABEL: func.func @test_basic_fusion(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[GROUP:.*]] = tcp.group {
// CHECK:           %[[TANH:.*]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           %[[ADD:.*]] = tcp.add %[[TANH]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           tcp.yield %[[ADD]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>
// CHECK:         return %[[GROUP]] : tensor<?x?xf32>
func.func @test_basic_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.add %0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_multiple_fusions(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<1x?xf32>, %[[ARG3:.*]]: tensor<1x?xf32>) -> tensor<?x?xf32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[GROUP1:.*]] = tcp.group {
// CHECK:           %[[TANH1:.*]] = tcp.tanh %[[ARG2]] : tensor<1x?xf32> -> tensor<1x?xf32>
// CHECK:           %[[TANH2:.*]] = tcp.tanh %[[ARG3]] : tensor<1x?xf32> -> tensor<1x?xf32>
// CHECK:           %[[MUL:.*]] = tcp.mul %[[TANH1]], %[[TANH2]] : tensor<1x?xf32>, tensor<1x?xf32> -> tensor<1x?xf32>
// CHECK:           tcp.yield %[[MUL]] : tensor<1x?xf32>
// CHECK:         } : tensor<1x?xf32>
// CHECK:         %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
// CHECK:         %[[BCAST:.*]] = tcp.broadcast %[[GROUP1]], %[[DIM]] {axes = [0]} : tensor<1x?xf32>, index -> tensor<?x?xf32>
// CHECK:         %[[GROUP2:.*]] = tcp.group {
// CHECK:           %[[CLAMP:.*]] = tcp.clamp %[[BCAST]] {min_float = 0.000000e+00 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           %[[ADD:.*]] = tcp.add %[[ARG0]], %[[CLAMP]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           %[[SUB:.*]] = tcp.sub %[[ARG1]], %[[ADD]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:           tcp.yield %[[SUB]] : tensor<?x?xf32>
// CHECK:         } : tensor<?x?xf32>
// CHECK:         return %[[GROUP2]] : tensor<?x?xf32>
func.func @test_multiple_fusions(%arg0 : tensor<?x?xf32>,
                                 %arg1 : tensor<?x?xf32>,
                                 %arg2 : tensor<1x?xf32>,
                                 %arg3 : tensor<1x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg2 : tensor<1x?xf32> -> tensor<1x?xf32>
  %1 = tcp.tanh %arg3 : tensor<1x?xf32> -> tensor<1x?xf32>
  %2 = tcp.mul %0, %1 : tensor<1x?xf32>, tensor<1x?xf32> -> tensor<1x?xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %3 = tcp.broadcast %2, %dim {axes = [0]} : tensor<1x?xf32>, index -> tensor<?x?xf32>
  %4 = tcp.clamp %3 {min_float = 0.0 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
  %5 = tcp.add %arg0, %4 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %6 = tcp.sub %arg1, %5 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}

// -----

// CHECK:   func.func @test_multi_use_fusion(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:     %[[V0:.+]] = tcp.group {
// CHECK:       %[[V1:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V2:.+]] = tcp.add %[[V1]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V3:.+]] = tcp.sub %[[V2]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V4:.+]] = tcp.mul %[[V2]], %[[V3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V4]] : tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>
// CHECK:     return %[[V0]] : tensor<?x?xf32>
// CHECK:   }
func.func @test_multi_use_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.add %0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = tcp.sub %1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %3 = tcp.mul %1, %2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK:   func.func @test_multi_use_fusion(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK:     %[[V0:.+]]:2 = tcp.group {
// CHECK:       %[[V1:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V2:.+]] = tcp.add %[[V1]], %[[V1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V3:.+]] = tcp.mul %[[V2]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V4:.+]] = tcp.sub %[[V2]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V3]], %[[V4]] : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK:     return %[[V0]]#1, %[[V0]]#0 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK:   }
func.func @test_multi_use_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)  {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.add %0, %0 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = tcp.sub %1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %3 = tcp.mul %1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  "func.return" (%2, %3) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
}
