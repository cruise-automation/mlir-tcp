// RUN: tcp-opt <%s -convert-torch-to-tcp-cruise-internal -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: execute_engine_variadic
//       CHECK:    tcp.custom_op("torch.tensorrt.execute_engine_variadic")
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

// CHECK-LABEL: gather_op
//       CHECK: tcp.custom_op("torch.aten.gather")
//  CHECK-SAME: axis = 1 : i64
func.func @gather_op(%arg0: !torch.vtensor<[2,2],si64>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %false = torch.constant.bool false
    %int1 = torch.constant.int 1
    %0 = torch.aten.gather %arg1, %int1, %arg0, %false : !torch.vtensor<[2,2],f32>, !torch.int, !torch.vtensor<[2,2],si64>, !torch.bool -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }

// -----

// CHECK-LABEL: index_hacked_twin_op
//       CHECK: tcp.custom_op("torch.aten.index.Tensor_hacked_twin")
func.func @index_hacked_twin_op(%arg0: !torch.vtensor<[1,30,19,41],f32>, %arg1: !torch.vtensor<[1,1,1,1],si64>, %arg2: !torch.vtensor<[30,1,1],si64>, %arg3: !torch.vtensor<[19,1],si64>, %arg4: !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,30,19,3],f32> {
    %0 = torch.prim.ListConstruct %arg1, %arg2, %arg3, %arg4 : (!torch.vtensor<[1,1,1,1],si64>, !torch.vtensor<[30,1,1],si64>, !torch.vtensor<[19,1],si64>, !torch.vtensor<[3],si64>) -> !torch.list<vtensor>
    %1 = torch.aten.index.Tensor_hacked_twin %arg0, %0 : !torch.vtensor<[1,30,19,41],f32>, !torch.list<vtensor> -> !torch.vtensor<[1,30,19,3],f32>
    return %1 : !torch.vtensor<[1,30,19,3],f32>
  }
