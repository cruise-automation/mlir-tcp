// RUN: tcp-opt <%s -convert-torch-to-tcp-custom-op -canonicalize -split-input-file | FileCheck %s


// CHECK-LABEL: func.func @torch.aten.index_put_impl_op(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[25],f32>
// CHECK-SAME:         %[[ARG1:.*]]: !torch.vtensor<[10],si32>
// CHECK-SAME:         %[[ARG2:.*]]: !torch.vtensor<[],f32>) -> !torch.vtensor<[25],f32>
// CHECK-DAG:      %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[25],f32> -> tensor<25xf32>
// CHECK-DAG:      %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[10],si32> -> tensor<10xi32>
// CHECK-DAG:      %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG2]] : !torch.vtensor<[],f32> -> tensor<f32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten._index_put_impl") %[[T1]], %[[T2]], %[[T0]]
// CHECK-SAME:                          {accumulate = false, torch_operand_names = ["self", "index_0", "values"], unsafe = false}
// CHECK-SAME:                          tensor<25xf32>, tensor<10xi32>, tensor<f32> -> tensor<25xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<25xf32> -> !torch.vtensor<[25],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[25],f32>
func.func @torch.aten.index_put_impl_op(%arg0: !torch.vtensor<[25],f32>, %arg1: !torch.vtensor<[10],si32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[25],f32> {
  %false = torch.constant.bool false
  %0 = torch.prim.ListConstruct %arg1 : (!torch.vtensor<[10],si32>) -> !torch.list<optional<vtensor>>
  %1 = torch.aten._index_put_impl %arg0, %0, %arg2, %false, %false : !torch.vtensor<[25],f32>, !torch.list<optional<vtensor>>, !torch.vtensor<[],f32>, !torch.bool, !torch.bool -> !torch.vtensor<[25],f32>
  return %1 : !torch.vtensor<[25],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.transposed_convolution(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,64,1,100],f32>) -> !torch.vtensor<[1,64,2,200],f32>
// CHECK:          %[[T0:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<64xf32>) : !torch.vtensor<[64],f32>
// CHECK:          %[[T1:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
// CHECK:          %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,64,1,100],f32> -> tensor<1x64x1x100xf32>
// CHECK:          %[[T3:.*]] = torch_c.to_builtin_tensor %[[T1]] : !torch.vtensor<[64,64,3,3],f32> -> tensor<64x64x3x3xf32>
// CHECK:          %[[T4:.*]] = torch_c.to_builtin_tensor %[[T0]] : !torch.vtensor<[64],f32> -> tensor<64xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.convolution") %[[T2]], %[[T3]], %[[T4]] {
// CHECK-SAME:                          dilation = [1 : index, 1 : index],
// CHECK-SAME:                          groups = 1 : i64,
// CHECK-SAME:                          output_padding = [1 : index, 1 : index],
// CHECK-SAME:                          padding = [1 : index, 1 : index],
// CHECK-SAME:                          stride = [2 : index, 2 : index],
// CHECK-SAME:                          torch_operand_names = ["input", "weight", "bias"],
// CHECK-SAME:                          transposed = true}
// CHECK-SAME:      tensor<1x64x1x100xf32>, tensor<64x64x3x3xf32>, tensor<64xf32> -> tensor<1x64x2x200xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x64x2x200xf32> -> !torch.vtensor<[1,64,2,200],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,64,2,200],f32>
func.func @torch.aten.transposed_convolution(%input: !torch.vtensor<[1,64,1,100],f32>) -> !torch.vtensor<[1,64,2,200],f32> {
  %true = torch.constant.bool true
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %weight = torch.vtensor.literal(dense<0.0> : tensor<64x64x3x3xf32>) : !torch.vtensor<[64,64,3,3],f32>
  %bias = torch.vtensor.literal(dense<0.0> : tensor<64xf32>) : !torch.vtensor<[64],f32>
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1x1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %output = torch.aten.convolution %input, %weight, %bias, %stride, %int1x1, %int1x1, %true, %int1x1, %int1 : !torch.vtensor<[1,64,1,100],f32>, !torch.vtensor<[64,64,3,3],f32>, !torch.vtensor<[64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,64,2,200],f32>
  return %output : !torch.vtensor<[1,64,2,200],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.regular_convolution_1d(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,256,500],f32>) -> !torch.vtensor<[?,256,500],f32>
// CHECK:          %[[T1:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<256x256x1xf32>) : !torch.vtensor<[256,256,1],f32>
// CHECK:          %[[T2:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,256,500],f32> -> tensor<?x256x500xf32>
// CHECK:          %[[T3:.*]] = torch_c.to_builtin_tensor %[[T1]] : !torch.vtensor<[256,256,1],f32> -> tensor<256x256x1xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.convolution") %[[T2]], %[[T3]] {
// CHECK-SAME:                          dilation = [1 : index],
// CHECK-SAME:                          groups = 1 : i64,
// CHECK-SAME:                          output_padding = [0 : index],
// CHECK-SAME:                          padding = [0 : index],
// CHECK-SAME:                          stride = [1 : index],
// CHECK-SAME:                          torch_operand_names = ["input", "weight"],
// CHECK-SAME:                          transposed = false}
// CHECK-SAME:      tensor<?x256x500xf32>, tensor<256x256x1xf32> -> tensor<?x256x500xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<?x256x500xf32> -> !torch.vtensor<[?,256,500],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[?,256,500],f32>
func.func @torch.aten.regular_convolution_1d(%input: !torch.vtensor<[?,256,500],f32>) -> !torch.vtensor<[?,256,500],f32> {
  %false = torch.constant.bool false
  %weights = torch.vtensor.literal(dense<0.0> : tensor<256x256x1xf32>) : !torch.vtensor<[256,256,1],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %listint0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %listint1 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %output = torch.aten.convolution %input, %weights, %none, %listint1, %listint0, %listint1, %false, %listint0, %int1 : !torch.vtensor<[?,256,500],f32>, !torch.vtensor<[256,256,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[?,256,500],f32>
  return %output : !torch.vtensor<[?,256,500],f32>
}

// -----

// CHECK: torch.aten.convolution %{{.*}}
func.func @torch.aten.regular_convolution_2d(%input: !torch.vtensor<[1,9,16,1600],f32>) -> !torch.vtensor<[1,32,16,1600],f32> {
  %false = torch.constant.bool false
  %weights = torch.vtensor.literal(dense<0.0> : tensor<32x9x3x3xf32>) : !torch.vtensor<[32,9,3,3],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int0x0 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %int1x1 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %output = torch.aten.convolution %input, %weights, %none, %int1x1, %int1x1, %int1x1, %false, %int0x0, %int1 : !torch.vtensor<[1,9,16,1600],f32>, !torch.vtensor<[32,9,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,32,16,1600],f32>
  return %output : !torch.vtensor<[1,32,16,1600],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.fake_quantize_per_tensor_affine(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,64,32,32],f32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.fake_quantize_per_tensor_affine") %[[T0]] {
// CHECK-SAME:                          quant_max = 255 : i64,
// CHECK-SAME:                          quant_min = 0 : i64,
// CHECK-SAME:                          scale = 1.000000e-05 : f64,
// CHECK-SAME:                          torch_operand_names = ["self"],
// CHECK-SAME:                          zero_point = 0 : i64}
// CHECK-SAME:      tensor<1x64x32x32xf32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x64x32x32xf32> -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,64,32,32],f32>
func.func @torch.aten.fake_quantize_per_tensor_affine(%input: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32> {
  %float1.000000e-05 = torch.constant.float 1.000000e-05
  %int0 = torch.constant.int 0
  %int255 = torch.constant.int 255
  %output = torch.aten.fake_quantize_per_tensor_affine %input, %float1.000000e-05, %int0, %int0, %int255 : !torch.vtensor<[1,64,32,32],f32>, !torch.float, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,64,32,32],f32>
  return %output : !torch.vtensor<[1,64,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.fake_quantize_per_tensor_affine.tensor_qparams(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,64,32,32],f32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.fake_quantize_per_tensor_affine.tensor_qparams") %[[T0]], %{{.*}}, %{{.*}} {
// CHECK-SAME:                          quant_max = 255 : i64,
// CHECK-SAME:                          quant_min = 0 : i64,
// CHECK-SAME:                          torch_operand_names = ["self", "scale", "zero_point"]} :
// CHECK-SAME:      tensor<1x64x32x32xf32>, tensor<1xf32>, tensor<1xi32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x64x32x32xf32> -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,64,32,32],f32>
func.func @torch.aten.fake_quantize_per_tensor_affine.tensor_qparams(%input: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32> {
  %scale = torch.vtensor.literal(dense<0.0393700786> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  %zero_point = torch.vtensor.literal(dense<2> : tensor<1xsi32>) : !torch.vtensor<[1],si32>
  %int0 = torch.constant.int 0
  %int255 = torch.constant.int 255
  %output = torch.aten.fake_quantize_per_tensor_affine.tensor_qparams %input, %scale, %zero_point, %int0, %int255 : !torch.vtensor<[1,64,32,32],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[1,64,32,32],f32>
  return %output : !torch.vtensor<[1,64,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.fake_quantize_per_tensor_affine.tensor_qparams_zero(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,64,32,32],f32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.fake_quantize_per_tensor_affine.tensor_qparams") %[[T0]], %{{.*}}, %{{.*}} {
// CHECK-SAME:                          quant_max = 255 : i64,
// CHECK-SAME:                          quant_min = 0 : i64,
// CHECK-SAME:                          torch_operand_names = ["self", "scale", "zero_point"]} :
// CHECK-SAME:      tensor<1x64x32x32xf32>, tensor<1xf32>, tensor<1xi32> -> tensor<1x64x32x32xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x64x32x32xf32> -> !torch.vtensor<[1,64,32,32],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,64,32,32],f32>
func.func @torch.aten.fake_quantize_per_tensor_affine.tensor_qparams_zero(%input: !torch.vtensor<[1,64,32,32],f32>) -> !torch.vtensor<[1,64,32,32],f32> {
  %scale = torch.vtensor.literal(dense<0.0393700786> : tensor<1xf32>) : !torch.vtensor<[1],f32>
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %none = torch.constant.none
  %5 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %cuda3A0 = torch.constant.device "cuda:0"
  %false = torch.constant.bool false
  %zero_point = torch.aten.zeros %5, %int3, %none, %cuda3A0, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1],si32>
  %int0 = torch.constant.int 0
  %int255 = torch.constant.int 255
  %output = torch.aten.fake_quantize_per_tensor_affine.tensor_qparams %input, %scale, %zero_point, %int0, %int255 : !torch.vtensor<[1,64,32,32],f32>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],si32>, !torch.int, !torch.int -> !torch.vtensor<[1,64,32,32],f32>
  return %output : !torch.vtensor<[1,64,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.fake_quantize_per_channel_affine(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,3,32,32],f32> -> tensor<1x3x32x32xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.fake_quantize_per_channel_affine") %[[T0]], %{{.*}}, %{{.*}} {
// CHECK-SAME:                          axis = 1 : i64,
// CHECK-SAME:                          quant_max = 255 : i64,
// CHECK-SAME:                          quant_min = 0 : i64,
// CHECK-SAME:                          torch_operand_names = ["self", "scale", "zero_point"]} :
// CHECK-SAME:      tensor<1x3x32x32xf32>, tensor<3xf32>, tensor<3xi32> -> tensor<1x3x32x32xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x3x32x32xf32> -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,3,32,32],f32>
func.func @torch.aten.fake_quantize_per_channel_affine(%input: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
  %scale = torch.vtensor.literal(dense<0.0393700786> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %zero_point = torch.vtensor.literal(dense<2> : tensor<3xsi32>) : !torch.vtensor<[3],si32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int255 = torch.constant.int 255
  %output = torch.aten.fake_quantize_per_channel_affine %input, %scale, %zero_point, %int1, %int0, %int255 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,3,32,32],f32>
  return %output : !torch.vtensor<[1,3,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.fake_quantize_per_channel_affine_zero_like(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,3,32,32],f32> -> tensor<1x3x32x32xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.fake_quantize_per_channel_affine") %[[T0]], %{{.*}}, %{{.*}} {
// CHECK-SAME:                          axis = 1 : i64,
// CHECK-SAME:                          quant_max = 255 : i64,
// CHECK-SAME:                          quant_min = 0 : i64,
// CHECK-SAME:                          torch_operand_names = ["self", "scale", "zero_point"]} :
// CHECK-SAME:      tensor<1x3x32x32xf32>, tensor<3xf32>, tensor<3xi32> -> tensor<1x3x32x32xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<1x3x32x32xf32> -> !torch.vtensor<[1,3,32,32],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[1,3,32,32],f32>
func.func @torch.aten.fake_quantize_per_channel_affine_zero_like(%input: !torch.vtensor<[1,3,32,32],f32>) -> !torch.vtensor<[1,3,32,32],f32> {
  %scale = torch.vtensor.literal(dense<0.0393700786> : tensor<3xf32>) : !torch.vtensor<[3],f32>
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int255 = torch.constant.int 255
  %int3 = torch.constant.int 3
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %false = torch.constant.bool false
  %zero_point = torch.aten.zeros_like %scale, %int3, %none, %cuda3A0, %false, %none : !torch.vtensor<[3],f32>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[3],si32>
  %output = torch.aten.fake_quantize_per_channel_affine %input, %scale, %zero_point, %int1, %int0, %int255 : !torch.vtensor<[1,3,32,32],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],si32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,3,32,32],f32>
  return %output : !torch.vtensor<[1,3,32,32],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.topk(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,2304],f32>) -> !torch.vtensor<[?,80],f32> {
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,2304],f32> -> tensor<?x2304xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.topk") %[[T0]] {dim = -1 : i64, k = 80 : i64, largest = true, sorted = true, torch_operand_names = ["self"]} :
// CHECK-SAME:      tensor<?x2304xf32> -> tensor<?x80xf32>, tensor<?x80xi64>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM:.*]] : tensor<?x80xf32> -> !torch.vtensor<[?,80],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[?,80],f32>
func.func @torch.aten.topk(%input: !torch.vtensor<[?,2304],f32>) -> !torch.vtensor<[?,80],f32> {
  %int-1 = torch.constant.int -1
  %int80 = torch.constant.int 80
  %true = torch.constant.bool true
  %output0, %output1 = torch.aten.topk %input, %int80, %int-1, %true, %true : !torch.vtensor<[?,2304],f32>, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[?,80],f32>, !torch.vtensor<[?,80],si64>
  return %output0 : !torch.vtensor<[?,80],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.sort(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,2304],f32>) -> !torch.vtensor<[?,2304],f32> {
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,2304],f32> -> tensor<?x2304xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.sort") %[[T0]] {descending = true, dim = -1 : i64, torch_operand_names = ["self"]} :
// CHECK-SAME:      tensor<?x2304xf32> -> tensor<?x2304xf32>, tensor<?x2304xi64>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM:.*]] : tensor<?x2304xf32> -> !torch.vtensor<[?,2304],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[?,2304],f32>
func.func @torch.aten.sort(%input: !torch.vtensor<[?,2304],f32>) -> !torch.vtensor<[?,2304],f32> {
  %int-1 = torch.constant.int -1
  %true = torch.constant.bool true
  %output0, %output1 = torch.aten.sort %input, %int-1, %true : !torch.vtensor<[?,2304],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?,2304],f32>, !torch.vtensor<[?,2304],si64>
  return %output0 : !torch.vtensor<[?,2304],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.cumsum(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?],si64> {
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?],si32> -> tensor<?xi32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.cumsum") %[[T0]] {dim = 0 : i64, torch_operand_names = ["self"]} : tensor<?xi32> -> tensor<?xi64>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM]] : tensor<?xi64> -> !torch.vtensor<[?],si64>
// CHECK:          return %[[RES]] : !torch.vtensor<[?],si64>
func.func @torch.aten.cumsum(%input: !torch.vtensor<[?],si32>) -> !torch.vtensor<[?],si64> {
  %int0 = torch.constant.int 0
  %none = torch.constant.none
  %1 = torch.aten.cumsum %input, %int0, %none : !torch.vtensor<[?],si32>, !torch.int, !torch.none -> !torch.vtensor<[?],si64>
  return %1 : !torch.vtensor<[?],si64>
}

// -----

// CHECK-LABEL: func.func @torch.aten.min.dim(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,80],f32>) -> !torch.vtensor<[?],f32> {
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,80],f32> -> tensor<?x80xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.min.dim") %[[T0]] {dim = 1 : i64, keepdim = false, torch_operand_names = ["self"]} :
// CHECK-SAME:      tensor<?x80xf32> -> tensor<?xf32>, tensor<?xi64>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM:.*]] : tensor<?xf32> -> !torch.vtensor<[?],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[?],f32>
func.func @torch.aten.min.dim(%input: !torch.vtensor<[?,80],f32>) -> !torch.vtensor<[?],f32> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %output0, %output1 = torch.aten.min.dim %input, %int1, %false : !torch.vtensor<[?,80],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],f32>, !torch.vtensor<[?],si64>
  return %output0 : !torch.vtensor<[?],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.view_dynamic_shape(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,384,16],f32>, %[[ARG1:.*]]: tensor<?x2736x16xf32>) -> !torch.vtensor<[?,24,16,16],f32> {
// CHECK:          %[[C0:.*]] = arith.constant 0 : index
// CHECK:          %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,384,16],f32> -> tensor<?x384x16xf32>
// CHECK:          %[[DIM:.*]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x2736x16xf32>
// CHECK:          %[[CUSTOM:.*]] = tcp.custom_op("torch.aten.view") %[[T0]], %[[DIM]] {size = array<i64: -9223372036854775808, 24, 16, 16>, torch_operand_names = ["self", "idx_0"]} :
// CHECK-SAME:      tensor<?x384x16xf32>, index -> tensor<?x24x16x16xf32>
// CHECK:          %[[RES:.*]] = torch_c.from_builtin_tensor %[[CUSTOM:.*]] : tensor<?x24x16x16xf32> -> !torch.vtensor<[?,24,16,16],f32>
// CHECK:          return %[[RES]] : !torch.vtensor<[?,24,16,16],f32>
func.func @torch.aten.view_dynamic_shape(%arg0: !torch.vtensor<[?,384,16],f32>, %arg1: tensor<?x2736x16xf32>) -> !torch.vtensor<[?,24,16,16],f32> {
  %c0 = arith.constant 0 : index
  %int24 = torch.constant.int 24
  %int16 = torch.constant.int 16
  %dim_32 = tensor.dim %arg1, %c0 : tensor<?x2736x16xf32>
  %1 = arith.index_cast %dim_32 : index to i64
  %2 = torch_c.from_i64 %1
  %3 = torch.prim.ListConstruct %2, %int24, %int16, %int16 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %4 = torch.aten.view %arg0, %3 : !torch.vtensor<[?,384,16],f32>, !torch.list<int> -> !torch.vtensor<[?,24,16,16],f32>
  return %4 : !torch.vtensor<[?,24,16,16],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.slice_scatter(
// CHECK-DAG: %[[ARG0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,3],f32> -> tensor<1x3xf32>
// CHECK-DAG: %[[ARG1:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
// CHECK: %[[OUT:.*]] = tcp.custom_op("torch.aten.slice_scatter") %[[ARG0]], %[[ARG1]] {dim = 1 : i64, end = 3 : i64, start = 2 : i64, step = 4 : i64, torch_operand_names = ["self", "src"]} : tensor<1x3xf32>, tensor<1x2xf32> -> tensor<1x3xf32>
// CHECK: %[[RET:.*]] = torch_c.from_builtin_tensor %[[OUT]] : tensor<1x3xf32> -> !torch.vtensor<[1,3],f32>
// CHECK: return %[[RET]]
func.func @torch.aten.slice_scatter(%arg0: !torch.vtensor<[1,3],f32>, %arg1: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,3],f32> {
  %dim = torch.constant.int 1
  %start = torch.constant.int 2
  %end = torch.constant.int 3
  %step = torch.constant.int 4
  %0 = torch.aten.slice_scatter %arg0, %arg1, %dim, %start, %end, %step : !torch.vtensor<[1,3],f32>, !torch.vtensor<[1,2],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,3],f32>
  return %0 : !torch.vtensor<[1,3],f32>
}

// -----

// CHECK-LABEL: func.func @torch.aten.arange.start_step(
// CHECK-SAME:         %[[ARG0:.*]]: !torch.int) -> !torch.vtensor<[?],si32> {
// CHECK: %[[IN:.*]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[OUT:.*]] = tcp.custom_op("torch.aten.arange.start_step") %[[IN]] {start = 0.000000e+00 : f64, step = 1.000000e+00 : f64, torch_operand_names = ["end"]} : i64 -> tensor<?xi32>
// CHECK: %[[RET:.*]] = torch_c.from_builtin_tensor %[[OUT]] : tensor<?xi32> -> !torch.vtensor<[?],si32>
// CHECK: return %[[RET]] : !torch.vtensor<[?],si32>
func.func @torch.aten.arange.start_step(%arg0: !torch.int) -> !torch.vtensor<[?],si32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cpu = torch.constant.device "cpu"
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %int3 = torch.constant.int 3
  %1 = torch.aten.arange.start_step %int0, %arg0, %int1, %int3, %none, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si32>
  return %1 : !torch.vtensor<[?],si32>
}
