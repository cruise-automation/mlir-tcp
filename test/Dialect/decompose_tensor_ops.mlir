// RUN: tcp-opt %s -split-input-file -decompose-tensor-ops | FileCheck %s

// CHECK-LABEL: func.func @tensor_concat_float_tensors(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x3xf32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x3xf32>) -> tensor<?x3xf32> {
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x3xf32>
// CHECK:        %[[DIM_0:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x3xf32>
// CHECK:        %[[T0:.*]] = affine.apply #map()[%[[DIM]], %[[DIM_0]]]
// CHECK:        %[[T1:.*]] = tensor.empty(%[[T0]]) : tensor<?x3xf32>
// CHECK:        %[[SLICE1:.*]] = tensor.insert_slice %[[ARG0]] into %[[T1]][0, 0] [%[[DIM]], 3] [1, 1] : tensor<?x3xf32> into tensor<?x3xf32>
// CHECK:        %[[SLICE_2:.*]] = tensor.insert_slice %arg1 into %[[SLICE1]][%[[DIM]], 0] [%[[DIM_0]], 3] [1, 1] : tensor<?x3xf32> into tensor<?x3xf32>
// CHECK:        return %[[SLICE_2]] : tensor<?x3xf32>
func.func @tensor_concat_float_tensors(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {                                                                                                                                                                             
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>                                                                                                                                                                 
  return %concat : tensor<?x3xf32>                                                                                                                                                                                                                                    
}                                                                                                                                                                                                                                                                     

// -----

// CHECK-LABEL: func.func @tensor_concat_int_tensors(
// CHECK-SAME:          %[[ARG0:.*]]: tensor<?x3xi32>,
// CHECK-SAME:          %[[ARG1:.*]]: tensor<?x3xi32>) -> tensor<?x3xi32> {
// CHECK:         tensor.empty
// CHECK:         tensor.insert_slice
// CHECK:         tensor.insert_slice
func.func @tensor_concat_int_tensors(%arg0: tensor<?x3xi32>, %arg1: tensor<?x3xi32>) -> tensor<?x3xi32> {                                                                                                                                                                             
  %concat = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x3xi32>, tensor<?x3xi32>) -> tensor<?x3xi32>                                                                                                                                                                 
  return %concat : tensor<?x3xi32>                                                                                                                                                                                                                                    
}       
