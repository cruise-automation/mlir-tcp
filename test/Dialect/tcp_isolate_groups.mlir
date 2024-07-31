// RUN: tcp-opt %s -split-input-file -tcp-isolate-group-ops --mlir-print-ir-after-all | FileCheck %s

// CHECK-LABEL: func.func @test_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: tensor<?x?xf32>):
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[ARG3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
func.func @test_group(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
   return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_bigger_group(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>, %[[ARG1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG2:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]], %[[ARG2]] {
// CHECK:            ^bb0(%[[ARG3:.*]]: tensor<?x?xf32>, %[[ARG4:.*]]: tensor<?x?xf32>, %[[ARG5:.*]]: tensor<?x?xf32>):
// CHECK:              %[[CLAMP:.*]] = tcp.clamp %[[ARG3]] {min_float = 0.000000e+00 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[SUB:.*]] = tcp.sub %[[ARG4]], %[[CLAMP]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ARG5]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[ADD:.*]] = tcp.add %[[TANH]], %[[SUB]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[ADD]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
func.func @test_bigger_group(%arg0 : tensor<?x?xf32>,
                             %arg1 : tensor<?x?xf32>,
                             %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %4 = tcp.clamp %arg0 {min_float = 0.0 : f32} : tensor<?x?xf32> -> tensor<?x?xf32>
      %5 = tcp.sub %arg1, %4 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %6 = tcp.tanh %arg2 : tensor<?x?xf32> -> tensor<?x?xf32>
      %7 = tcp.add %6, %5 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %7 : tensor<?x?xf32>
  }) : () -> tensor<?x?xf32>
   return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_propagate_attrs(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]], %[[ARG1]] attributes {test_attr} {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<?x?xf32>, %[[ARG3:.*]]: tensor<?x?xf32>):
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[ARG3]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              %[[TANH:.*]] = tcp.tanh %[[ADD]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:              tcp.yield %[[TANH]] : tensor<?x?xf32>
// CHECK:          } : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:          return %[[GROUP]] : tensor<?x?xf32>
// CHECK:        }
func.func @test_propagate_attrs(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
      %3 = tcp.tanh %2 : tensor<?x?xf32> -> tensor<?x?xf32>
      tcp.yield %3 : tensor<?x?xf32>
  }) { "test_attr" } : () -> tensor<?x?xf32>
   return %10 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @test_const(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<5xi32>) -> tensor<5xi32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<5xi32>):
// CHECK:              %[[CONST:.*]] = tcp.const
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              %[[MUL:.*]] = tcp.mul %[[ADD]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              tcp.yield %[[MUL]] : tensor<5xi32>
// CHECK:          } : tensor<5xi32> -> tensor<5xi32>
// CHECK:          return %[[GROUP]] : tensor<5xi32>
// CHECK:        }
func.func @test_const(%arg0 : tensor<5xi32>) -> tensor<5xi32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %1 = tcp.const {value = dense<[3, 6, 10, 15, 23]> : tensor<5xi32>} : tensor<5xi32>
      %2 = tcp.add %arg0, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      %3 = tcp.mul %2, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      tcp.yield %3 : tensor<5xi32>
  }) : () -> tensor<5xi32>
   return %10 : tensor<5xi32>
}

// -----

// CHECK-LABEL: func.func @test_inputs_with_multiple_uses(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<5xi32>) -> tensor<5xi32>
// CHECK:          %[[GROUP:.*]] = tcp.isolated_group %[[ARG0]] {
// CHECK:            ^bb0(%[[ARG2:.*]]: tensor<5xi32>):
// CHECK:              %[[CONST:.*]] = tcp.const
// CHECK:              %[[ADD:.*]] = tcp.add %[[ARG2]], %[[CONST]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              %[[MUL:.*]] = tcp.mul %[[ADD]], %[[ARG2]] : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
// CHECK:              tcp.yield %[[MUL]] : tensor<5xi32>
// CHECK:          } : tensor<5xi32> -> tensor<5xi32>
// CHECK:          return %[[GROUP]] : tensor<5xi32>
// CHECK:        }
func.func @test_inputs_with_multiple_uses(%arg0 : tensor<5xi32>) -> tensor<5xi32> {
  %10 = "tcp.group" () ({
    ^bb0() :
      %1 = tcp.const {value = dense<[3, 6, 10, 15, 23]> : tensor<5xi32>} : tensor<5xi32>
      %2 = tcp.add %arg0, %1 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      %3 = tcp.mul %2, %arg0 : tensor<5xi32>, tensor<5xi32> -> tensor<5xi32>
      tcp.yield %3 : tensor<5xi32>
  }) : () -> tensor<5xi32>
   return %10 : tensor<5xi32>
}


// -----

// isolate tcp.group ops in the presence of nested regions.

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @forward(%[[ARG0:.+]]: tensor<?x4096xf32>, %[[ARG1:.+]]: tensor<?x4096xf32>, %[[ARG2:.+]]: tensor<?x4096xf32>) -> tensor<?x4096xf32> {
// CHECK:     %[[C0:.+]] = arith.constant 0 : index
// CHECK:     %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x4096xf32>
// CHECK:     %[[V0:.+]] = tcp.isolated_group %[[DIM]], %[[ARG0]], %[[ARG1]] attributes {group_type = "codegen_group"} {
// CHECK:     ^bb0(%[[ARG3:.+]]: index, %[[ARG4:.+]]: tensor<?x4096xf32>, %[[ARG5:.+]]: tensor<?x4096xf32>):
// CHECK:       %[[V1:.+]] = tensor.empty(%[[ARG3]]) : tensor<?x4096xf32>
// CHECK:       %[[V2:.+]] = scf.forall (%[[ARG6:.+]], %[[ARG7:.+]]) in (%[[ARG3]], 4096) shared_outs(%[[ARG8:.+]] = %[[V1]]) -> (tensor<?x4096xf32>) {
// CHECK:         %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG4]][%[[ARG6]], %[[ARG7]]] [1, 1] [1, 1] : tensor<?x4096xf32> to tensor<1x1xf32>
// CHECK:         %[[EXTRACTED_SLICE_0:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG6]], %[[ARG7]]] [1, 1] [1, 1] : tensor<?x4096xf32> to tensor<1x1xf32>
// CHECK:         %[[V3:.+]] = tensor.empty() : tensor<1x1xf32>
// CHECK:         %[[V4:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[EXTRACTED_SLICE]], %[[EXTRACTED_SLICE_0]] : tensor<1x1xf32>, tensor<1x1xf32>) outs(%[[V3]] : tensor<1x1xf32>) {
// CHECK:         ^bb0(%[[IN:.+]]: f32, %[[IN_1:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK:           %[[V5:.+]] = arith.mulf %[[IN]], %[[IN_1]] : f32
// CHECK:           linalg.yield %[[V5]] : f32
// CHECK:         } -> tensor<1x1xf32>
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[V4]] into %[[ARG8]][%[[ARG6]], %[[ARG7]]] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<?x4096xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       tcp.yield %[[V2]] : tensor<?x4096xf32>
// CHECK:     } : index, tensor<?x4096xf32>, tensor<?x4096xf32> -> tensor<?x4096xf32>
// CHECK:     return %[[V0]] : tensor<?x4096xf32>
// CHECK:   }
// CHECK: }
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @forward(%arg0: tensor<?x4096xf32>, %arg1: tensor<?x4096xf32>, %arg2: tensor<?x4096xf32>) -> tensor<?x4096xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf32>
  %0 = tcp.group attributes {group_type = "codegen_group"} {
    %1 = tensor.empty(%dim) : tensor<?x4096xf32>
    %2 = scf.forall (%arg3, %arg4) in (%dim, 4096) shared_outs(%arg5 = %1) -> (tensor<?x4096xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg4] [1, 1] [1, 1] : tensor<?x4096xf32> to tensor<1x1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, %arg4] [1, 1] [1, 1] : tensor<?x4096xf32> to tensor<1x1xf32>
      %3 = tensor.empty() : tensor<1x1xf32>
      %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3 : tensor<1x1xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %8 = arith.mulf %in, %in_4 : f32
        linalg.yield %8 : f32
      } -> tensor<1x1xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg5[%arg3, %arg4] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<?x4096xf32>
      }
    }
    tcp.yield %2 : tensor<?x4096xf32>
  } : tensor<?x4096xf32>
  return %0 : tensor<?x4096xf32>
}

// -----

// Ensure that we correctly drop `tcp.bind_symbolic_shape` ops within the
// newly created tcp.isolated_group region.

// CHECK:   func.func @test_symbolic_shape_ops(%[[ARG0:.+]]: tensor<?x3xf32>) -> tensor<?x3xf32> {
// CHECK:     %[[V0:.+]] = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
// CHECK:     tcp.bind_symbolic_shape %[[ARG0]], [%[[V0]]], affine_map<()[s0] -> (s0, 3)> : tensor<?x3xf32>
// CHECK:     %[[V1:.+]] = tcp.isolated_group %[[ARG0]] {
// CHECK:     ^bb0(%[[ARG1:.+]]: tensor<?x3xf32>):
// CHECK:       %[[V2:.+]] = tcp.add %[[ARG1]], %[[ARG1]] : tensor<?x3xf32>, tensor<?x3xf32> -> tensor<?x3xf32>
// CHECK-NOT: tcp.bind_symbolic_shape
// CHECK:       %[[V3:.+]] = tcp.mul %[[V2]], %[[V2]] : tensor<?x3xf32>, tensor<?x3xf32> -> tensor<?x3xf32>
// CHECK:       tcp.yield %[[V3]] : tensor<?x3xf32>
// CHECK:     } : tensor<?x3xf32> -> tensor<?x3xf32>
// CHECK:     tcp.bind_symbolic_shape %[[V1]], [%[[V0]]], affine_map<()[s0] -> (s0, 3)> : tensor<?x3xf32>
// CHECK:     return %[[V1]] : tensor<?x3xf32>
// CHECK:   }
func.func @test_symbolic_shape_ops(%arg0 : tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = tcp.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775806} : i64
  tcp.bind_symbolic_shape %arg0, [%0], affine_map<()[s0] -> (s0, 3)> : tensor<?x3xf32>
  %10 = "tcp.group" () ({
    ^bb0() :
      %2 = tcp.add %arg0, %arg0 : tensor<?x3xf32>, tensor<?x3xf32> -> tensor<?x3xf32>
      tcp.bind_symbolic_shape %2, [%0], affine_map<()[s0] -> (s0, 3)> : tensor<?x3xf32>
      %3 = tcp.mul %2, %2 : tensor<?x3xf32>, tensor<?x3xf32> -> tensor<?x3xf32>
      tcp.yield %3 : tensor<?x3xf32>
  }) : () -> tensor<?x3xf32>
  tcp.bind_symbolic_shape %10, [%0], affine_map<()[s0] -> (s0, 3)> : tensor<?x3xf32>
  return %10 : tensor<?x3xf32>
}
