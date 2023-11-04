// RUN: tcp-opt <%s -convert-torch-to-tcp-custom-op -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  func.func @torch.aten.gather_op(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[2,2],si64>
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[2,2],f32> -> tensor<2x2xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[2,2],si64> -> tensor<2x2xi64>
// CHECK:         %[[T2:.*]] = tcp.custom_op("torch.aten.gather") %[[T0]], %[[T1]] {axis = 1 : i64} : tensor<2x2xf32>, tensor<2x2xi64> -> tensor<2x2xf32>
// CHECK:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<2x2xf32> -> !torch.vtensor<[2,2],f32>
// CHECK:         return %[[T3]] : !torch.vtensor<[2,2],f32>
func.func @torch.aten.gather_op(%arg0: !torch.vtensor<[2,2],si64>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %false = torch.constant.bool false
  %int1 = torch.constant.int 1
  %0 = torch.aten.gather %arg1, %int1, %arg0, %false : !torch.vtensor<[2,2],f32>, !torch.int, !torch.vtensor<[2,2],si64>, !torch.bool -> !torch.vtensor<[2,2],f32>
  return %0 : !torch.vtensor<[2,2],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.aten.index_hacked_twin_op(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,30,19,41],f32>
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[1,1,1,1],si64>
// CHECK-SAME:         %[[ARG2:.*]]: !torch.vtensor<[30,1,1],si64>
// CHECK-SAME:         %[[ARG3:.*]]: !torch.vtensor<[19,1],si64>
// CHECK-SAME:         %[[ARG4:.*]]: !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,30,19,3],f32> {
// CHECK:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,30,19,41],f32> -> tensor<1x30x19x41xf32>
// CHECK:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[1,1,1,1],si64> -> tensor<1x1x1x1xi64>
// CHECK:         %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[30,1,1],si64> -> tensor<30x1x1xi64>
// CHECK:         %[[T3:.*]] = torch_c.to_builtin_tensor %[[ARG3]] : !torch.vtensor<[19,1],si64> -> tensor<19x1xi64>
// CHECK:         %[[T4:.*]] = torch_c.to_builtin_tensor %[[ARG4]] : !torch.vtensor<[3],si64> -> tensor<3xi64>
// CHECK:         %[[T5:.*]] = tcp.custom_op("torch.aten.index.Tensor_hacked_twin") %[[T0]], %[[T1]], %[[T2]], %[[T3]], %[[T4]] : tensor<1x30x19x41xf32>, tensor<1x1x1x1xi64>, tensor<30x1x1xi64>, tensor<19x1xi64>, tensor<3xi64> -> tensor<1x30x19x3xf32>
// CHECK:         %[[T6:.*]] = torch_c.from_builtin_tensor %[[T5]] : tensor<1x30x19x3xf32> -> !torch.vtensor<[1,30,19,3],f32>
// CHECK:         return %[[T6]] : !torch.vtensor<[1,30,19,3],f32>
func.func @torch.aten.index_hacked_twin_op(%arg0: !torch.vtensor<[1,30,19,41],f32>, %arg1: !torch.vtensor<[1,1,1,1],si64>, %arg2: !torch.vtensor<[30,1,1],si64>, %arg3: !torch.vtensor<[19,1],si64>, %arg4: !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,30,19,3],f32> {
  %0 = torch.prim.ListConstruct %arg1, %arg2, %arg3, %arg4 : (!torch.vtensor<[1,1,1,1],si64>, !torch.vtensor<[30,1,1],si64>, !torch.vtensor<[19,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<vtensor>
  %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[1,30,19,41],f32>, !torch.list<vtensor> -> !torch.vtensor<[1,30,19,3],f32>
  return %1 : !torch.vtensor<[1,30,19,3],f32>
}
