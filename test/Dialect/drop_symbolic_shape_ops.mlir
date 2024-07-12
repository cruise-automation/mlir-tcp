// RUN: tcp-opt %s -drop-symbolic-shape-ops | FileCheck %s

// CHECK-LABEL:  func.func @test_drop_symbolic_shape_ops(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK-NOT:     %[[S0:.*]] = tcp.symbolic_int "s0" {min_val = 3, max_val = 6} : i64
// CHECK-NOT:     %[[S1:.*]] = tcp.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : i64
// CHECK-NOT:     tcp.bind_symbolic_shape %[[ARG0]], [%{{.*}}], affine_map<()[s0] -> (s0)> : tensor<?xf32>
// CHECK-NOT:     tcp.bind_symbolic_shape %[[ARG1]], [%{{.*}}], affine_map<()[s0] -> (s0 + 1)> : tensor<?xf32>
// CHECK:         return %[[ARG0]] : tensor<?xf32>
func.func @test_drop_symbolic_shape_ops(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = tcp.symbolic_int "s0" {min_val = 3, max_val = 6} : i64
  %1 = tcp.symbolic_int "s0 + 1" {min_val = 4, max_val = 7} : i64
  tcp.bind_symbolic_shape %arg0, [%0], affine_map<()[s0] -> (s0)> : tensor<?xf32>
  tcp.bind_symbolic_shape %arg1, [%0], affine_map<()[s0] -> (s0 + 1)> : tensor<?xf32>
  return %arg0 : tensor<?xf32>
}
