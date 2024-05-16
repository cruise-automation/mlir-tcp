// RUN: tcp-opt <%s -convert-torch-to-tcp-cruise-internal -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: execute_engine_variadic
//       CHECK:    tcp.custom_op("tensorrt.execute_engine")
//  CHECK-SAME:       engine = dense<[238, 254]> : tensor<2xui8>
//  CHECK-SAME:       precision = "fp32"
//  CHECK-SAME:       shape_info = "inputs=(input_0:f32[8,8];input_1:f32[8]),outputs=(output_0:f32[8,8];output_1:f32[8])"

func.func @execute_engine_variadic(%arg0: !torch.vtensor<[8,8],f32>, %arg1: !torch.vtensor<[8],f32>) -> (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>) {
    %0 = torch.tensorrt.create_engine() {serialized_engine = dense<[238, 254]> : tensor<2xui8>, shape_info = "inputs=(input_0:f32[8,8];input_1:f32[8]),outputs=(output_0:f32[8,8];output_1:f32[8])", precision = "fp32"} : !torch.TRTEngine
    %1:2 = "torch.tensorrt.execute_engine_variadic"(%arg0, %arg1, %0) : (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>, !torch.TRTEngine) -> (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>)
    %2 = torch.aten.tanh %1#0 : !torch.vtensor<[8,8],f32> -> !torch.vtensor<[8,8],f32>
    %3:2 = "torch.tensorrt.execute_engine_variadic"(%2, %1#1, %0) : (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>, !torch.TRTEngine) -> (!torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>)
    return %3#0, %3#1 : !torch.vtensor<[8,8],f32>, !torch.vtensor<[8],f32>
}

// -----

// CHECK-LABEL: fused_gather_submanifold_conv3d
//       CHECK:   tcp.custom_op("torch.cuda_gems_sparse_conv.fused_gather_submanifold_conv3d_op")
//  CHECK-SAME:      input_feature_size = 16 : i32
//  CHECK-SAME:      kernel_x_dim = 3 : i32
//  CHECK-SAME:      kernel_y_dim = 3 : i32
//  CHECK-SAME:      kernel_z_dim = 1 : i32
//  CHECK-SAME:      num_neighbors = 27 : i32
//  CHECK-SAME:      output_feature_size = 16 : i32
//  CHECK-SAME:      pointwise_op = 1 : i32

func.func @fused_gather_submanifold_conv3d(%arg0: !torch.vtensor<[1024,27],si32>, %arg1: !torch.vtensor<[1024,16],f32>, %arg2: !torch.vtensor<[3,3,1,16,16],f32>) -> !torch.vtensor<[1024,16],f32> {
  %0 = "torch.cuda_gems_sparse_conv.fused_gather_config"() {input_feature_size = 16 : i32, kernel_x_dim = 3 : i32, kernel_y_dim = 3 : i32, kernel_z_dim = 1 : i32, num_neighbors = 27 : i32, output_feature_size = 16 : i32, pointwise_op = 1 : i32} : () -> !torch.FusedGatherSubmanifoldConv3DConfig
  %1 = torch.cuda_gems_sparse_conv.fused_gather_submanifold_conv3d_op(%0, %arg0, %arg1, %arg2) : (!torch.FusedGatherSubmanifoldConv3DConfig, !torch.vtensor<[1024,27],si32>, !torch.vtensor<[1024,16],f32>, !torch.vtensor<[3,3,1,16,16],f32>) -> !torch.vtensor<[1024,16],f32>
  return %1 : !torch.vtensor<[1024,16],f32>
}

// -----

// CHECK-LABEL: sparse_array_to_dense_map
//       CHECK:   tcp.custom_op("torch.cuda_gems_sparse_conv.sparse_array_to_dense_map_op")
//       CHECK:   angle = 360 : i32
//       CHECK:   height = 32 : i32
//       CHECK:   radius = 480 : i32
func.func @sparse_array_to_dense_map(%arg0: !torch.vtensor<[?],si32>) -> !torch.vtensor<[32,360,480],si32> {
  %0 = "torch.cuda_gems_sparse_conv.sparse_array_to_dense_map_config"() {angle = 360 : i32, height = 32 : i32, radius = 480 : i32} : () -> !torch.SparseArrayToDenseMapConfig
  %1 = torch.cuda_gems_sparse_conv.sparse_array_to_dense_map_op(%0, %arg0) : (!torch.SparseArrayToDenseMapConfig, !torch.vtensor<[?],si32>) -> !torch.vtensor<[32,360,480],si32>
  return %1 : !torch.vtensor<[32,360,480],si32>
}

// -----

// CHECK-LABEL: build_neighbor_map_op
//       CHECK: tcp.custom_op("torch.cuda_gems_sparse_conv.build_neighbor_map_op")
//  CHECK-SAME: circular_padding = false
//  CHECK-SAME: grid_angle = 360 : i32
//  CHECK-SAME: grid_height = 32 : i32
//  CHECK-SAME: grid_radius = 480 : i32
//  CHECK-SAME: input_dilation_angle = 1 : i32
//  CHECK-SAME: input_dilation_height = 1 : i32
//  CHECK-SAME: input_dilation_radius = 1 : i32
//  CHECK-SAME: output_grid_angle = 360 : i32
//  CHECK-SAME: output_grid_height = 32 : i32
//  CHECK-SAME: output_grid_radius = 480 : i32
//  CHECK-SAME: output_stride_angle = 1 : i32
//  CHECK-SAME: output_stride_height = 1 : i32
//  CHECK-SAME: output_stride_radius = 1 : i32

func.func @build_neighbor_map_op(%arg0: !torch.vtensor<[32,360,480],si32>, %arg1: !torch.vtensor<[1024],si32>) -> !torch.vtensor<[1024,27],si32> {
  %0 = "torch.cuda_gems_sparse_conv.build_neighbor_map_config"() {circular_padding = false, grid_angle = 360 : i32, grid_height = 32 : i32, grid_radius = 480 : i32, input_dilation_angle = 1 : i32, input_dilation_height = 1 : i32, input_dilation_radius = 1 : i32, output_grid_angle = 360 : i32, output_grid_height = 32 : i32, output_grid_radius = 480 : i32, output_stride_angle = 1 : i32, output_stride_height = 1 : i32, output_stride_radius = 1 : i32} : () -> !torch.BuildNeighborMapConfig
  %1 = torch.cuda_gems_sparse_conv.build_neighbor_map_op(%0, %arg1, %arg0) : (!torch.BuildNeighborMapConfig, !torch.vtensor<[1024],si32>, !torch.vtensor<[32,360,480],si32>) -> !torch.vtensor<[1024,27],si32>
  return %1 : !torch.vtensor<[1024,27],si32>
}

// -----

// Since function arguments are not yet converted, casts will be added which will be
// resolved in the next pass
// CHECK:  @index_from_tensor_tensor_from_index(
// CHECK:   %[[ARG0:.+]]: !torch.vtensor<[1],si64>
// CHECK:        %[[CAST1:.+]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1],si64> -> tensor<1xi64>
// CHECK:        %[[INDEX:.+]] = tcp.caspr_index_from_tensor(%[[CAST1]]) : (tensor<1xi64>) -> index
// CHECK:        %[[RES:.+]] = tcp.caspr_create_tensor_from_index(%[[INDEX]]) : (index) -> tensor<1xi64>
// CHECK:        %[[CAST2:.+]] = torch_c.from_builtin_tensor %[[RES]] : tensor<1xi64> ->  !torch.vtensor<[1],si64>
// CHECK:        return %[[CAST2]] : !torch.vtensor<[1],si64>
func.func @index_from_tensor_tensor_from_index(%arg0: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> {
  %0 = torch.caspr_index_from_tensor(%arg0) : (!torch.vtensor<[1],si64>) -> !torch.index
  %1 = torch.caspr_create_tensor_from_index(%0) : (!torch.index) -> !torch.vtensor<[1],si64>
  return %1 : !torch.vtensor<[1],si64>
}
