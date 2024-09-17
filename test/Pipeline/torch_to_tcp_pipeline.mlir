// RUN: tcp-opt %s -torch-backend-to-tcp-backend-pipeline -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: torch.aten.mul.Scalar$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5xbf16>
// CHECK: %[[VAL_1:.*]] = tcp.const {value = dense<2.000000e+00> : tensor<f64>} : tensor<f64>
// CHECK: %[[C5:.*]] = arith.constant 5 : index
// CHECK: %[[VAL_2:.*]] = tcp.cast %[[VAL_1]] : tensor<f64> -> tensor<bf16>
// CHECK: %[[VAL_3:.*]] = tensor.expand_shape %[[VAL_2]] [] output_shape [1] : tensor<bf16> into tensor<1xbf16>
// CHECK: %[[VAL_4:.*]] = tcp.broadcast %[[VAL_3]], %[[C5]] {axes = [0]} : tensor<1xbf16>, index -> tensor<5xbf16>
// CHECK: %[[VAL_5:.*]] = tcp.mul %[[VAL_0]], %[[VAL_4]] : tensor<5xbf16>, tensor<5xbf16> -> tensor<5xbf16>
// CHECK: return %[[VAL_5]] : tensor<5xbf16>
func.func @torch.aten.mul.Scalar$mixed_type(%arg0: !torch.vtensor<[5],bf16>) -> !torch.vtensor<[5],bf16> {
  %float2.000000e00 = torch.constant.float 2.000000e+00
  %0 = torch.aten.mul.Scalar %arg0, %float2.000000e00 : !torch.vtensor<[5],bf16>, !torch.float -> !torch.vtensor<[5],bf16>
  return %0 : !torch.vtensor<[5],bf16>
}

// -----

// CHECK-LABEL: torch.aten.add.Tensor$mixed_type_fp
// CHECK-SAME: %[[VAL_0:.*]]: tensor<6xbf16>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<6xf32>
// CHECK: %[[VAL_3:.*]] = tcp.cast %[[VAL_1]] : tensor<6xf32> -> tensor<6xbf16>
// CHECK: %[[VAL_4:.*]] = tcp.add %[[VAL_0]], %[[VAL_3]] : tensor<6xbf16>, tensor<6xbf16> -> tensor<6xbf16>
// CHECK: return %[[VAL_4]] : tensor<6xbf16>
func.func @torch.aten.add.Tensor$mixed_type_fp(%arg0: !torch.vtensor<[6],bf16>, %arg1: !torch.vtensor<[6],f32>, %arg2: !torch.float) -> !torch.vtensor<[6],bf16> {
  %float1 = torch.constant.float 1.000000e+00
  %0 = torch.aten.add.Tensor %arg0, %arg1, %float1 : !torch.vtensor<[6],bf16>, !torch.vtensor<[6],f32>, !torch.float -> !torch.vtensor<[6],bf16>
  return %0 : !torch.vtensor<[6],bf16>
}

// -----

// CHECK-LABEL: torch.aten.add.Tensor$mixed_type_int
// CHECK-SAME: %[[VAL_0:.*]]: tensor<5xf32>
// CHECK-SAME: %[[VAL_1:.*]]: tensor<5xbf16>
// CHECK: %[[VAL_2:.*]] = tcp.cast %[[VAL_1]] : tensor<5xbf16> -> tensor<5xf32>
// CHECK: %[[VAL_3:.*]] = tcp.add %[[VAL_0]], %[[VAL_2]] : tensor<5xf32>, tensor<5xf32> -> tensor<5xf32>
// CHECK: return %[[VAL_3]] : tensor<5xf32>
func.func @torch.aten.add.Tensor$mixed_type_int(%arg0: !torch.vtensor<[5],f32>, %arg1: !torch.vtensor<[5],bf16>) -> !torch.vtensor<[5],f32> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],bf16>, !torch.int -> !torch.vtensor<[5],f32>
  return %0 : !torch.vtensor<[5],f32>
}

// -----

// CHECK-LABEL: torch.aten.Scalar$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x1x32x64xi16>
// CHECK: %[[VAL_1:.*]] = tcp.const {value = dense<256> : tensor<i64>} : tensor<i64>
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[VAL_2:.*]] = tcp.cast %[[VAL_1]] {in_int_signedness = #tcp<signedness Signed>, out_int_signedness = #tcp<signedness Signed>} : tensor<i64> -> tensor<i32>
// CHECK: %[[VAL_3:.*]] = tcp.cast %[[VAL_0]] {in_int_signedness = #tcp<signedness Signed>, out_int_signedness = #tcp<signedness Signed>} : tensor<1x1x32x64xi16> -> tensor<1x1x32x64xi32>
// CHECK: %[[VAL_4:.*]] = tensor.expand_shape %[[VAL_2]] [] output_shape [1, 1, 1, 1] : tensor<i32> into tensor<1x1x1x1xi32>
// CHECK: %[[VAL_5:.*]] = tcp.broadcast %[[VAL_4]], %[[C32]], %[[C64]] {axes = [2, 3]} : tensor<1x1x1x1xi32>, index, index -> tensor<1x1x32x64xi32>                                                                    
// CHECK: %[[VAL_6:.*]] = tcp.add %[[VAL_3]], %[[VAL_5]] : tensor<1x1x32x64xi32>, tensor<1x1x32x64xi32> -> tensor<1x1x32x64xi32>                                                                                              
// CHECK: return %[[VAL_6]] : tensor<1x1x32x64xi32>    
func.func @torch.aten.Scalar$mixed_type(%arg0: !torch.vtensor<[1,1,32,64],si16>) -> !torch.vtensor<[1,1,32,64],si32> {
  %int1 = torch.constant.int 1
  %int256 = torch.constant.int 256
  %0 = torch.aten.add.Scalar %arg0, %int256, %int1 : !torch.vtensor<[1,1,32,64],si16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,32,64],si32>
  return %0 : !torch.vtensor<[1,1,32,64],si32>
}

// -----

// CHECK-LABEL: torch.aten.sub.Scalar$mixed_type
// CHECK-SAME: %[[VAL_0:.*]]: tensor<bf16>,
// CHECK: %[[VAL_2:.*]] = tcp.const {value = dense<1> : tensor<i64>} : tensor<i64>
// CHECK: %[[VAL_3:.*]] = tcp.cast %[[VAL_2]] {in_int_signedness = #tcp<signedness Signed>} : tensor<i64> -> tensor<bf16>
// CHECK: %[[VAL_4:.*]] = tcp.sub %[[VAL_0]], %[[VAL_3]] : tensor<bf16>, tensor<bf16> -> tensor<bf16>
// CHECK: return %[[VAL_4]] : tensor<bf16>
func.func @torch.aten.sub.Scalar$mixed_type(%arg0: !torch.vtensor<[],bf16>, %arg1: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Scalar %arg0, %int1, %int1 : !torch.vtensor<[],bf16>, !torch.int,  !torch.int -> !torch.vtensor<[],bf16>
  return %0 : !torch.vtensor<[],bf16>
}

// -----

func.func @torch.aten.maximum$mixed_type(%arg0: !torch.vtensor<[1,3,1],si32>, %arg1: !torch.vtensor<[1,3,1],f32>) -> !torch.vtensor<[1,3,1],f32> {
  // expected-error @below {{failed to legalize operation 'torch.aten.maximum'}}
  %0 = torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[1,3,1],si32>, !torch.vtensor<[1,3,1],f32> -> !torch.vtensor<[1,3,1],f32>
  return %0 : !torch.vtensor<[1,3,1],f32>
}

// -----

func.func @torch.aten.bitwise_and.Tensor$mixed_type(%arg0: !torch.vtensor<[?,?],si16>, %arg1: !torch.vtensor<[?,?],si32>) -> !torch.vtensor<[?,?],si32> {
  // expected-error @below {{failed to legalize operation 'torch.aten.bitwise_and.Tensor'}}
  %0 = torch.aten.bitwise_and.Tensor %arg0, %arg1 : !torch.vtensor<[?,?],si16>, !torch.vtensor<[?,?],si32> -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL: torch.aten.div.Tensor$mixed_type_fp
// CHECK-SAME: %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME: %[[VAL_1:.*]]: tensor<?x?xi32>
// CHECK: %[[VAL_2:.*]] = tcp.cast %[[VAL_1]] {in_int_signedness = #tcp<signedness Signed>} : tensor<?x?xi32> -> tensor<?x?xf32>
// CHECK: %[[VAL_3:.*]] = tcp.divf %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK: return %[[VAL_3]] : tensor<?x?xf32>
func.func @torch.aten.div.Tensor$mixed_type_fp(%arg0: !torch.vtensor<[?, ?],f32>, %arg1: !torch.vtensor<[?, ?],si32>) -> !torch.vtensor<[?, ?],f32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],f32>, !torch.vtensor<[?, ?],si32> -> !torch.vtensor<[?, ?],f32>
  return %0 : !torch.vtensor<[?, ?],f32>
}

// -----

// CHECK:   func.func @torch.aten.div.Tensor$mixed_type_int(%[[ARG0:.+]]: tensor<?x?xi16>, %[[ARG1:.+]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK:     %[[V0:.+]] = tcp.cast %[[ARG0]] {in_int_signedness = #tcp<signedness Signed>, out_int_signedness = #tcp<signedness Signed>} : tensor<?x?xi16> -> tensor<?x?xi32>
// CHECK:     %[[V1:.+]] = tcp.divsi %[[V0]], %[[ARG1]] {rounding_mode = #tcp<roundingMode Trunc>} : tensor<?x?xi32>, tensor<?x?xi32> -> tensor<?x?xi32>
// CHECK:     return %[[V1]] : tensor<?x?xi32>
func.func @torch.aten.div.Tensor$mixed_type_int(%arg0: !torch.vtensor<[?, ?],si16>, %arg1: !torch.vtensor<[?, ?],si32>) -> !torch.vtensor<[?, ?],si32> {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[?, ?],si16>, !torch.vtensor<[?, ?],si32> -> !torch.vtensor<[?, ?],si32>
  return %0 : !torch.vtensor<[?, ?],si32>
}
