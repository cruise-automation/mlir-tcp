// RUN: tcp-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_constant_folding() -> tensor<f32>
// CHECK:         %[[CONST0:.*]] = tcp.const {value = dense<2.500000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[MUL:.*]] = tcp.mul %[[CONST0]], %[[CONST0]] : tensor<f32>, tensor<f32> -> tensor<f32>
// CHECK:         return %[[MUL]] : tensor<f32>
func.func @test_constant_folding() -> tensor<f32> {
  %0 = tcp.const {value = dense<2.5> : tensor<f32>} : tensor<f32>
  %1 = tcp.const {value = dense<2.5> : tensor<f32>} : tensor<f32>
  %2 = tcp.mul %0, %1 : tensor<f32>, tensor<f32> -> tensor<f32>
  return %2 : tensor<f32>
}

// -----

// CHECK-LABEL:  func.func @test_tcp_symbolic_int$canonicalize(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:         %[[S0:.*]] = tcp.symbolic_int "s0" {min_val = 3, max_val = 6} : i64
// CHECK-NOT:     %[[S1:.*]] = tcp.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : i64
// CHECK:         tcp.bind_symbolic_shape %[[ARG0]], [%[[S0]]], affine_map<()[s0] -> (s0)> : tensor<?xf32>
// CHECK:         tcp.bind_symbolic_shape %[[ARG1]], [%[[S0]]], affine_map<()[s0] -> (s0 + 1)> : tensor<?xf32>
// CHECK:         return %[[ARG0]] : tensor<?xf32>
func.func @test_tcp_symbolic_int$canonicalize(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcp.symbolic_int "s0" {min_val = 3, max_val = 6} : i64
  %1 = tcp.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : i64
  tcp.bind_symbolic_shape %arg0, [%0], affine_map<()[s0] -> (s0)> : tensor<?xf32>
  tcp.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (s0 + 1)> : tensor<?xf32>
  return %arg0 : tensor<?xf32>
}
