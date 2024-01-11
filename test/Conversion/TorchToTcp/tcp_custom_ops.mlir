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

// -----

// CHECK-LABEL: func.func @torch.aten.index_put_impl_op(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[25],f32>
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[10],si32>
// CHECK-SAME:         %[[ARG2:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[25],f32>
// CHECK:          %[[TO:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[25],f32> -> tensor<25xf32>
// CHECK:          %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:          %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[10],si32> -> tensor<10xi32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten._index_put_impl") %[[T0]], %[[T2]], %[[T1]]
// CHECK-SAME:                          {accumulate = false, unsafe = false}
// CHECK-SAME:                          tensor<25xf32>, tensor<10xi32>, tensor<f32> -> tensor<25xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<25xf32> -> !torch.vtensor<[25],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[25],f32>
func.func @torch.aten.index_put_impl_op(%arg0: !torch.vtensor<[25],f32>, %arg1: !torch.vtensor<[10],si32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[25],f32> {
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[10],si32>) -> !torch.list<optional<vtensor>>
  %1 = torch.aten._index_put_impl %arg0, %0, %arg2, %false, %false : !torch.vtensor<[25],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[25],f32>
  return %1 : !torch.vtensor<[25],f32>
}

// ---

// CHECK: tcp.custom_op("torch.aten.convolution") %{{.*}}
// CHECK: torch.aten.convolution %{{.*}}
func.func @forward(%input: !torch.vtensor<[1,64,1,100],f32>) -> (!torch.vtensor<[1,64,2,200],f32>, !torch.vtensor<[1,32,16,1600],f32>) {
  %true = torch.constant.bool true
  %false = torch.constant.bool false
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<0.0> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1x1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %output = torch.aten.convolution %input, %weight, %bias, %stride, %int1x1, %int1x1, %true, %int1x1, %int1 : !torch.vtensor<[1,64,1,100],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,2,200],f32>

  %input2 = torch.vtensor.literal(dense<0.0> : tensor<1x9x16x1600xf32>) : !torch.vtensor<[1,9,16,1600],f32>
  %weights2 = torch.vtensor.literal(dense<0.0> : tensor<32x9x3x3xf32>) : !torch.vtensor<[32,9,3,3],f32>
  %int0x0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %output2 = torch.aten.convolution %input2, %weights2, %none, %int1x1, %int1x1, %int1x1, %false, %int0x0, %int1 : !torch.vtensor<[1,9,16,1600],f32>, !torch.vtensor<[32,9,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,16,1600],f32>

  return %output, %output2 : !torch.vtensor<[1,64,2,200],f32>, !torch.vtensor<[1,32,16,1600],f32>
}
