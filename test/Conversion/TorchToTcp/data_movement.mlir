// RUN: tcp-opt %s -convert-torch-to-tcp -split-input-file | FileCheck %s

// CHECK-LABEL: @torch.aten.cat
//   CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[?,?],f32>, %[[ARG1:.+]]: !torch.vtensor<[?,?],f32>
//        CHECK:   %[[V1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
//        CHECK:   %[[V2:.+]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
//        CHECK:   %[[V3:.+]] = tensor.concat dim(0) %[[V1]], %[[V2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//        CHECK:   %[[V4:.+]] = torch_c.from_builtin_tensor %[[V3]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
func.func @torch.aten.cat(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>) -> !torch.list<vtensor>
  %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL: @torch.aten.slice.Tensor
//   CHECK-SAME:   %[[ARG0:.+]]: !torch.vtensor<[1,56,?,?],f32>) -> !torch.vtensor<[1,28,?,?],f32>
//        CHECK:   %[[V1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,56,?,?],f32> -> tensor<1x56x?x?xf32>
//        CHECK:   %[[V2:.+]] = tcp.slice %[[V1]] starts(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) sizes(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) strides(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : tensor<1x56x?x?xf32> -> tensor<1x28x?x?xf32>
//        CHECK:   %[[V3:.+]] = torch_c.from_builtin_tensor %[[V2]] : tensor<1x28x?x?xf32> -> !torch.vtensor<[1,28,?,?],f32>
func.func @torch.aten.slice.Tensor(%arg0: !torch.vtensor<[1,56,?,?],f32>) -> !torch.vtensor<[1,28,?,?],f32> {
  %int100 = torch.constant.int 100
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %int2 = torch.constant.int 2
  %0 = torch.aten.slice.Tensor %arg0, %int1, %int0, %int100, %int2 : !torch.vtensor<[1,56,?,?],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,28,?,?],f32>
  return %0 : !torch.vtensor<[1,28,?,?],f32>
}

// -----

// CHECK-LABEL: @torch.aten.gather
// CHECK-SAME:       %[[ARG0:.+]]: !torch.vtensor<[1,4,3],f32>,
// CHECK-SAME:       %[[ARG1:.+]]: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32>
// CHECK-DAG:      %[[V1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,4,3],f32> -> tensor<1x4x3xf32>
// CHECK-DAG:      %[[V2:.+]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[1,4,2],si64> -> tensor<1x4x2xi64>
// CHECK:          %[[GATHER:.+]] = tcp.gather %[[V1]], %[[V2]] {dim = 2 : index} :
// CHECK-SAME:                          tensor<1x4x3xf32>, tensor<1x4x2xi64> -> tensor<1x4x2xf32>
// CHECK:          %[[V3:.+]] = torch_c.from_builtin_tensor %[[GATHER]] : tensor<1x4x2xf32> -> !torch.vtensor<[1,4,2],f32>
// CHECK:          return %[[V3]] : !torch.vtensor<[1,4,2],f32>
func.func @torch.aten.gather(%arg0: !torch.vtensor<[1,4,3],f32>, %arg1: !torch.vtensor<[1,4,2],si64>) -> !torch.vtensor<[1,4,2],f32> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten.gather %arg0, %int-1, %arg1, %false : !torch.vtensor<[1,4,3],f32>, !torch.int, !torch.vtensor<[1,4,2],si64>, !torch.bool -> !torch.vtensor<[1,4,2],f32>
  return %0 : !torch.vtensor<[1,4,2],f32>
}

// -----

// CHECK-LABEL: @torch.aten.index_select
// CHECK-SAME:       %[[ARG0:.+]]: !torch.vtensor<[4,3],f32>,
// CHECK-SAME:       %[[ARG1:.+]]: !torch.vtensor<[2],si64>) -> !torch.vtensor<[4,2],f32>
// CHECK:          %[[EXPAND_SHAPE:.+]] = tensor.expand_shape
// CHECK-SAME:                                        tensor<2xi64> into tensor<1x2xi64>
// CHECK:          %[[BROADCAST:.+]] = tcp.broadcast %[[EXPAND_SHAPE]], %{{.*}} {axes = [0]} : tensor<1x2xi64>, index -> tensor<4x2xi64>
// CHECK:          %[[GATHER:.+]] = tcp.gather %{{.*}}, %[[BROADCAST]] {dim = 1 : index} :
// CHECK-SAME:                          tensor<4x3xf32>, tensor<4x2xi64> -> tensor<4x2xf32>
// CHECK:          %[[V3:.+]] = torch_c.from_builtin_tensor %[[GATHER]] : tensor<4x2xf32> -> !torch.vtensor<[4,2],f32>
// CHECK:          return %[[V3]] : !torch.vtensor<[4,2],f32>
func.func @torch.aten.index_select(%arg0: !torch.vtensor<[4,3],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[4,2],f32> {
  %int-1 = torch.constant.int -1
  %0 = torch.aten.index_select %arg0, %int-1, %arg1: !torch.vtensor<[4,3],f32>, !torch.int, !torch.vtensor<[2],si64> -> !torch.vtensor<[4,2],f32>
  return %0 : !torch.vtensor<[4,2],f32>
}

// -----

// CHECK-label: @torch.aten.index.tensor_hacked_twin
// CHECK-DAG: %[[CAST0:.+]] = torch_c.to_builtin_tensor %arg0
// CHECK-DAG: %[[GATHER0:.+]] = tcp.gather %[[CAST0]], %[[SELECT0:.+]] {dim = 0 : index} : tensor<1x20x30xf32>, tensor<1x20x30xi64> -> tensor<1x20x30xf32>
// CHECK-DAG: %[[GATHER1:.+]] = tcp.gather %[[GATHER0]], %[[SELECT1:.+]] {dim = 1 : index} : tensor<1x20x30xf32>, tensor<1x5x30xi64> -> tensor<1x5x30xf32>
// CHECK-DAG: %[[GATHER2:.+]] = tcp.gather %[[GATHER1]], %[[SELECT2:.+]] {dim = 2 : index} : tensor<1x5x30xf32>, tensor<1x5x20xi64> -> tensor<1x5x20xf32>
// CHECK-DAG: %[[RET:.+]] = torch_c.from_builtin_tensor %[[GATHER2]]
// CHECK: return %[[RET]]
func.func @torch.aten.index.tensor_hacked_twin(%arg0: !torch.vtensor<[1,20,30],f32>, %select1: !torch.vtensor<[5,1],si64>, %select2: !torch.vtensor<[20],si64>) -> !torch.vtensor<[1,5,20],f32> {
  // there is a strange pattern that is being generated when selecting one axis.  It seems that it uses the Tensor_hacked_twin to select along all axis, but uses
  // arange to select all of the
  %none = torch.constant.none
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int4 = torch.constant.int 4 // this is a dtype on arange....
  %int-1 = torch.constant.int -1
  %arange = torch.aten.arange.start_step %int0, %int1, %int1, %int4, %none, %none, %none : !torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[1],si64>
  %arange1 = torch.aten.unsqueeze %arange, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
  %arange2 = torch.aten.unsqueeze %arange1, %int-1 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1,1],si64>

  %l = torch.prim.ListConstruct %arange2, %select1, %select2 : (!torch.vtensor<[1,1,1],si64>, !torch.vtensor<[5,1],si64>, !torch.vtensor<[20],si64>) -> !torch.list<vtensor>
  %ret = torch.aten.index.Tensor_hacked_twin %arg0, %l : !torch.vtensor<[1,20,30],f32>, !torch.list<vtensor> -> !torch.vtensor<[1,5,20],f32>
  return %ret : !torch.vtensor<[1,5,20],f32>
}
