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

// This and the previous test used to create a single fused group in
// earlier versions of the fusion algorithm. However, that algorithm had a
// bug causing us to revert to a simpler algo which does not create a
// single group for this sequence.

// CHECK:   func.func @test_multi_use_fusion(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK:     %[[V0:.+]] = tcp.group {
// CHECK:       %[[V3:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V4:.+]] = tcp.add %[[V3]], %[[V3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V4]] : tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>
// CHECK:     %[[V1:.+]] = tcp.sub %[[V0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V2:.+]] = tcp.mul %[[V0]], %[[V1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     return %[[V1]], %[[V2]] : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK:   }
func.func @test_multi_use_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)  {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.add %0, %0 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = tcp.sub %1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %3 = tcp.mul %1, %2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  "func.return" (%2, %3) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
}

// -----

// CHECK:   func.func @test_fusion_with_symbolic_shape(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:     %[[V0:.+]] = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
// CHECK:     %[[V1:.+]] = tcp.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775806} : i64
// CHECK:     tcp.bind_symbolic_shape %[[ARG0]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     %[[V2:.+]] = tcp.group {
// CHECK:       %[[V3:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.bind_symbolic_shape %[[V3]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:       %[[V4:.+]] = tcp.tanh %[[V3]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.bind_symbolic_shape %[[V4]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:       %[[V5:.+]] = tcp.add %[[V4]], %[[V4]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V5]] : tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>
// CHECK:     tcp.bind_symbolic_shape %[[V2]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     return %[[V2]] : tensor<?x?xf32>
// CHECK:   }
func.func @test_fusion_with_symbolic_shape(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %s0 = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
  %s1 = tcp.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775806} : i64
  
  tcp.bind_symbolic_shape %arg0, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>

  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %0, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>

  %1 = tcp.tanh %0 : tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %1, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
  
  %2 = tcp.add %1, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %2, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>

  return %2 : tensor<?x?xf32>
}

// -----

// CHECK:   func.func @test_multi_use_fusion_with_sym_shapes(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK:     %[[V0:.+]] = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
// CHECK:     %[[V1:.+]] = tcp.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775806} : i64
// CHECK:     tcp.bind_symbolic_shape %[[ARG0]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     %[[V2:.+]] = tcp.group {
// CHECK:       %[[V5:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.bind_symbolic_shape %[[V5]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:       %[[V6:.+]] = tcp.add %[[V5]], %[[V5]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V6]] : tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>
// CHECK:     tcp.bind_symbolic_shape %[[V2]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     %[[V3:.+]] = tcp.sub %[[V2]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     tcp.bind_symbolic_shape %[[V3]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     %[[V4:.+]] = tcp.mul %[[V2]], %[[V3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     tcp.bind_symbolic_shape %[[V4]], [%[[V0]], %[[V1]]], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
// CHECK:     return %[[V3]], %[[V4]] : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK:   }
func.func @test_multi_use_fusion_with_sym_shapes(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>)  {
  %s0 = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
  %s1 = tcp.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775806} : i64
  tcp.bind_symbolic_shape %arg0, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>

  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %0, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
  %1 = tcp.add %0, %0 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %1, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
  %2 = tcp.sub %1, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %2, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
  %3 = tcp.mul %1, %2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  tcp.bind_symbolic_shape %3, [%s0, %s1], affine_map<()[s0, s1] -> (s0, s1)> : tensor<?x?xf32>
  "func.return" (%2, %3) : (tensor<?x?xf32>, tensor<?x?xf32>) -> ()
}


// -----

// This test shows why iterating over all the users of an op and then
// fusing them together might lead to bugs. In this case, %0 is only used
// by %2 and %5 and they are all element-wise ops. However, if we create a
// tcp.group for them, there's no correct place to put the newly created
// tcp.group without violating dominance for the other operands and uses of
// %2 and %5.
//
// This change shows the need to start from a op and only look at its
// operands to start a fusion operation.


// CHECK:   func.func @buggy_tcp_fusion(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:     %[[V0:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V1:.+]] = tcp.custom_op("test.op") %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V2:.+]] = tcp.add %[[V0]], %[[V1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V3:.+]] = tcp.custom_op("test.op") %[[V2]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V4:.+]] = tcp.custom_op("test.op") %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V5:.+]] = tcp.mul %[[V0]], %[[V4]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     %[[V6:.+]] = tcp.custom_op("test.op") %[[V5]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:     return %[[V2]] : tensor<?x?xf32>
// CHECK:   }
func.func @buggy_tcp_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>)  {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  
  %1 = tcp.custom_op("test.op") %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %2 = tcp.add %0, %1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %3 = tcp.custom_op("test.op") %2 : tensor<?x?xf32> -> tensor<?x?xf32>

  %4 = tcp.custom_op("test.op") %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %5 = tcp.mul %0, %4 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %6 = tcp.custom_op("test.op") %5 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// Make sure that things do not break if a value is used twice by the same
// op.

// CHECK:   func.func @test_multi_use_fusion_same_op_uses(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:     %[[V0:.+]] = tcp.group {
// CHECK:       %[[V1:.+]] = tcp.tanh %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       %[[V2:.+]] = tcp.mul %[[V1]], %[[V1]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:       tcp.yield %[[V2]] : tensor<?x?xf32>
// CHECK:     } : tensor<?x?xf32>
// CHECK:     return %[[V0]] : tensor<?x?xf32>
// CHECK:   }
func.func @test_multi_use_fusion_same_op_uses(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  %3 = tcp.mul %0, %0 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
