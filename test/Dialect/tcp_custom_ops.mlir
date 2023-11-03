// RUN: tcp-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @tcp_custom_op(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = tcp.custom_op("torch.aten.my_custom_op") %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[T0]] : tensor<?x?xf32>
func.func @tcp_custom_op(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.custom_op("torch.aten.my_custom_op") %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @tcp_custom_op_with_named_attrs(
// CHECK-SAME:               %[[ARG0:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[T0:.*]] = tcp.custom_op("torch.aten.my_custom_op") %[[ARG0]] {axis = 0 : i32} : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK:         return %[[T0]] : tensor<?x?xf32>
func.func @tcp_custom_op_with_named_attrs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.custom_op("torch.aten.my_custom_op") %arg0 {axis = 0 : i32} : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @tcp_custom_op_without_op_name(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1{{expected attribute value}}
  %0 = tcp.custom_op() %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}