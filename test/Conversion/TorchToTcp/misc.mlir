// RUN: tcp-opt %s -convert-torch-to-tcp -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>} : tensor<4xf32>
// CHECK:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xf32> -> !torch.vtensor<[4],f32>
// CHECK:         return %[[T2]] : !torch.vtensor<[4],f32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],f32> {
  %0 = torch.vtensor.literal(dense<[5.000000e-01, 4.000000e-01, 3.000000e-01, 6.000000e-01]> : tensor<4xf32>) : !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : tensor<4xi32>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi32> -> !torch.vtensor<[4],si32>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],si32>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],si32> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xsi32>) : !torch.vtensor<[4],si32>
  return %0 : !torch.vtensor<[4],si32>
}

// -----

// CHECK-LABEL:  func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
//       CHECK:  %[[T1:.+]] = tcp.const {value = dense<[1, 2, 3, 4]> : tensor<4xi8>} : tensor<4xi8>
//       CHECK:  %[[T2:.+]] = torch_c.from_builtin_tensor %[[T1]] : tensor<4xi8> -> !torch.vtensor<[4],ui8>
//       CHECK:  return %[[T2]] : !torch.vtensor<[4],ui8>
func.func @torch.vtensor.literal() -> !torch.vtensor<[4],ui8> {
  %0 = torch.vtensor.literal(dense<[1, 2, 3, 4]> : tensor<4xui8>) : !torch.vtensor<[4],ui8>
  return %0 : !torch.vtensor<[4],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_f32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zeros_f32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_si32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],si32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.zeros_si32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],si32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_ui8(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],ui8> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i8>} : tensor<i8>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i8> into tensor<1x1xi8>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.zeros_ui8(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],ui8> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.zeros %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],ui8>
  return %1 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_f32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],f32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.ones_f32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_si32(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],si32> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.ones_si32(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],si32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],si32>
  return %1 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_ui8(%arg0: !torch.int, %arg1: !torch.int)
// CHECK-SAME:    -> !torch.vtensor<[?,?],ui8> {
// CHECK:         %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i8>} : tensor<i8>
// CHECK:         %[[T2:.*]] = torch_c.to_i64 %arg0
// CHECK:         %[[T3:.*]] = arith.index_cast %[[T2]] : i64 to index
// CHECK:         %[[T4:.*]] = torch_c.to_i64 %arg1
// CHECK:         %[[T5:.*]] = arith.index_cast %[[T4]] : i64 to index
// CHECK:         %[[T6:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i8> into tensor<1x1xi8>
// CHECK:         %[[T7:.*]] = tcp.broadcast  %[[T6]], %[[T3]], %[[T5]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:         %[[T8:.*]] = torch_c.from_builtin_tensor %[[T7]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:         return %[[T8]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.ones_ui8(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],ui8> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %cpu = torch.constant.device "cpu"
  %1 = torch.aten.ones %0, %none, %none, %cpu, %false : !torch.list<int>, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?,?],ui8>
  return %1 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_f32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.zeros_like_f32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_si32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i32>} : tensor<i32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.zeros_like_si32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.zeros_like_ui8(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<0> : tensor<i8>} : tensor<i8>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i8> into tensor<1x1xi8>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.zeros_like_ui8(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.zeros_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],ui8>
  return %0 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_f32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1.000000e+00> : tensor<f32>} : tensor<f32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<f32> into tensor<1x1xf32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xf32>, index, index -> tensor<?x?xf32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],f32>
func.func @torch.aten.ones_like_f32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f32>
  return %0 : !torch.vtensor<[?,?],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_si32(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i32>} : tensor<i32>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i32> into tensor<1x1xi32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi32>, index, index -> tensor<?x?xi32>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi32> -> !torch.vtensor<[?,?],si32>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],si32>
func.func @torch.aten.ones_like_si32(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],si32> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],si32>
  return %0 : !torch.vtensor<[?,?],si32>
}

// -----

// CHECK-LABEL:  @torch.aten.ones_like_ui8(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = tcp.const {value = dense<1> : tensor<i8>} : tensor<i8>
// CHECK:        %[[T2:.*]] = tensor.expand_shape %[[T1]] [] output_shape [1, 1] : tensor<i8> into tensor<1x1xi8>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[DIM1:.*]] = tensor.dim %[[T0]], %[[C1]] : tensor<?x?xf32>
// CHECK:        %[[BC:.*]] = tcp.broadcast %[[T2]], %[[DIM0]], %[[DIM1]] {axes = [0, 1]} : tensor<1x1xi8>, index, index -> tensor<?x?xi8>
// CHECK:        %[[T3:.*]] = torch_c.from_builtin_tensor %[[BC]] : tensor<?x?xi8> -> !torch.vtensor<[?,?],ui8>
// CHECK:        return %[[T3]] : !torch.vtensor<[?,?],ui8>
func.func @torch.aten.ones_like_ui8(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],ui8> {
  %int0 = torch.constant.int 0
  %int3 = torch.constant.int 3
  %false = torch.constant.bool false
  %none = torch.constant.none
  %cuda3A0 = torch.constant.device "cuda:0"
  %0 = torch.aten.ones_like %arg0, %int3, %int0, %cuda3A0, %false, %none : !torch.vtensor<[?,?],f32>, !torch.int, !torch.int, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],ui8>
  return %0 : !torch.vtensor<[?,?],ui8>
}

// -----

// CHECK-LABEL:  @torch.aten.size.int(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,?],f32>) {
// CHECK:        %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK:        %[[T1:.*]] = torch.constant.int 0
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM0:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x?xf32>
// CHECK:        %[[C1:.*]] = arith.index_cast %[[DIM0]] : index to i64
// CHECK:        return
func.func @torch.aten.size.int(%arg0: !torch.vtensor<[?,?],f32>) -> () {
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
  return
}

// -----

// CHECK-LABEL:  @torch.aten.expand(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[3,2],f32> {
// CHECK:        %[[TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
// CHECK:        %[[CONSTANT0:.*]] = torch.constant.int 3
// CHECK:        %[[CONSTANT1:.*]] = torch.constant.int -1
// CHECK:        %[[CAST0:.*]] = torch_c.to_i64 %[[CONSTANT0]]
// CHECK:        %[[BROADCAST_DIM0:.*]] = arith.index_cast %[[CAST0]] : i64 to index
// CHECK:        %{{.*}} = tcp.broadcast %[[TENSOR]], %[[BROADCAST_DIM0]] {axes = [0]} : tensor<1x2xf32>, index -> tensor<3x2xf32>
func.func @torch.aten.expand(%arg0: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[3,2],f32> {
  %int3 = torch.constant.int 3
  %int-1 = torch.constant.int -1
  %0 = torch.prim.ListConstruct %int3, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %1 = torch.aten.expand %arg0, %0, %false : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2],f32>
  return %1 : !torch.vtensor<[3,2],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.expand$rank_increase(
// CHECK-SAME:   %[[ARG0:.*]]: !torch.vtensor<[1,2],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,3,2],f32>) -> !torch.vtensor<[?,3,2],f32> {
// CHECK-DAG:    %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?,3,2],f32> -> tensor<?x3x2xf32>
// CHECK-DAG:    %[[T1:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
// CHECK:        %[[INT0:.*]] = torch.constant.int 0
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM:.*]] = tensor.dim %[[T0]], %[[C0]] : tensor<?x3x2xf32>
// CHECK:        %[[T2:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:        %[[T3:.*]] = torch_c.from_i64 %[[T2]]
// CHECK:        %[[INT3:.*]] = torch.constant.int 3
// CHECK:        %[[INT2:.*]] = torch.constant.int 2
// CHECK:        %[[EXPANDED:.*]] = tensor.expand_shape %[[T1]] {{\[}}[0, 1], [2]{{\]}} output_shape [1, 1, 2] : tensor<1x2xf32> into tensor<1x1x2xf32>
// CHECK:        %[[T5:.*]] = torch_c.to_i64 %[[T3]]
// CHECK:        %[[T6:.*]] = arith.index_cast %[[T5]] : i64 to index
// CHECK:        %[[T7:.*]] = torch_c.to_i64 %[[INT3]]
// CHECK:        %[[T8:.*]] = arith.index_cast %[[T7]] : i64 to index
// CHECK:        %[[T9:.*]] = tcp.broadcast %[[EXPANDED]], %[[T6]], %[[T8]] {axes = [0, 1]} : tensor<1x1x2xf32>, index, index -> tensor<?x3x2xf32>
// CHECK:        %[[T10:.*]] = torch_c.from_builtin_tensor %[[T9]] : tensor<?x3x2xf32> -> !torch.vtensor<[?,3,2],f32>
// CHECK:        return %[[T10]] : !torch.vtensor<[?,3,2],f32>
func.func @torch.aten.expand$rank_increase(%arg0: !torch.vtensor<[1,2],f32>, %arg1: !torch.vtensor<[?,3,2],f32>) -> !torch.vtensor<[?,3,2],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg1, %int0 : !torch.vtensor<[?,3,2],f32>, !torch.int -> !torch.int
  %int3 = torch.constant.int 3
  %int2 = torch.constant.int 2
  %1 = torch.prim.ListConstruct %0, %int3, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %2 = torch.aten.expand %arg0, %1, %false : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,3,2],f32>
  return %2 : !torch.vtensor<[?,3,2],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.broadcast_to(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[1,2,1,2],f32>) -> !torch.vtensor<[4,2,4,2],f32> {
// CHECK:        %[[TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[1,2,1,2],f32> -> tensor<1x2x1x2xf32>
// CHECK:        %[[CONSTANT:.*]] = torch.constant.int 4
// CHECK:        %[[CAST0:.*]] = torch_c.to_i64 %[[CONSTANT]]
// CHECK:        %[[BROADCAST_DIM0:.*]] = arith.index_cast %[[CAST0]] : i64 to index
// CHECK:        %[[CAST1:.*]] = torch_c.to_i64 %[[CONSTANT]]
// CHECK:        %[[BROADCAST_DIM1:.*]] = arith.index_cast %[[CAST1]] : i64 to index
// CHECK:        %[[AFTER_BROADCAST:.*]] = tcp.broadcast %[[TENSOR]], %[[BROADCAST_DIM0]], %[[BROADCAST_DIM1]] {axes = [0, 2]} : tensor<1x2x1x2xf32>, index, index -> tensor<4x2x4x2xf32>
// CHECK:        %[[OUT:.*]] = torch_c.from_builtin_tensor %[[AFTER_BROADCAST]] : tensor<4x2x4x2xf32> -> !torch.vtensor<[4,2,4,2],f32>
// CHECK:        return %[[OUT]] : !torch.vtensor<[4,2,4,2],f32>
func.func @torch.aten.broadcast_to(%arg0: !torch.vtensor<[1,2,1,2],f32>) -> !torch.vtensor<[4,2,4,2],f32> {
  %int2 = torch.constant.int 2
  %int4 = torch.constant.int 4
  %1 = torch.prim.ListConstruct %int4, %int2, %int4, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.broadcast_to %arg0, %1 : !torch.vtensor<[1,2,1,2],f32>, !torch.list<int> -> !torch.vtensor<[4,2,4,2],f32>
  return %2 : !torch.vtensor<[4,2,4,2],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.broadcast_to_with_dynamic_dim_input(
// CHECK-SAME:   %[[ARG:.*]]: !torch.vtensor<[?,2736,1],f32>) -> !torch.vtensor<[?,2736,16],f32> {
// CHECK:        %[[TENSOR:.*]] = torch_c.to_builtin_tensor %[[ARG]] : !torch.vtensor<[?,2736,1],f32> -> tensor<?x2736x1xf32>
// CHECK:        %[[CONSTANT:.*]] = torch.constant.int 16
// CHECK:        %[[CAST0:.*]] = torch_c.to_i64 %[[CONSTANT]]
// CHECK:        %[[BROADCAST_DIM:.*]] = arith.index_cast %[[CAST0]] : i64 to index
// CHECK:        %[[AFTER_BROADCAST:.*]] = tcp.broadcast %[[TENSOR]], %[[BROADCAST_DIM]] {axes = [2]} : tensor<?x2736x1xf32>, index -> tensor<?x2736x16xf32>
// CHECK:        %[[OUT:.*]] = torch_c.from_builtin_tensor %[[AFTER_BROADCAST]] : tensor<?x2736x16xf32> -> !torch.vtensor<[?,2736,16],f32>
// CHECK:        return %[[OUT]] : !torch.vtensor<[?,2736,16],f32>
func.func @torch.aten.broadcast_to_with_dynamic_dim_input(%arg0: !torch.vtensor<[?,2736,1],f32>) -> !torch.vtensor<[?,2736,16],f32> {
  %int0 = torch.constant.int 0
  %int2736 = torch.constant.int 2736
  %int16 = torch.constant.int 16
  %0 = torch.aten.size.int %arg0, %int0 : !torch.vtensor<[?,2736,1],f32>, !torch.int -> !torch.int
  %1 = torch.prim.ListConstruct %0, %int2736, %int16 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.broadcast_to %arg0, %1 : !torch.vtensor<[?,2736,1],f32>, !torch.list<int> -> !torch.vtensor<[?,2736,16],f32>
  return %2 : !torch.vtensor<[?,2736,16],f32>
}

// -----

// CHECK-LABEL:  @torch.aten.broadcast_to_dynamic_dim(
// CHECK-SAME:   %[[ARG0:.*]]: !torch.vtensor<[1,2],f32>, %[[ARG1:.*]]: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,2],f32> {
// CHECK-DAG:    %[[ARG1_T:.*]] = torch_c.to_builtin_tensor %[[ARG1]] : !torch.vtensor<[?],f32> -> tensor<?xf32>
// CHECK-DAG:    %[[ARG0_T:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[1,2],f32> -> tensor<1x2xf32>
// CHECK:        %[[C0:.*]] = arith.constant 0 : index
// CHECK:        %[[DIM:.*]] = tensor.dim %[[ARG1_T]], %[[C0]] : tensor<?xf32>
// CHECK:        %[[DIM_CAST:.*]] = arith.index_cast %[[DIM]] : index to i64
// CHECK:        %[[FROM:.*]] = torch_c.from_i64 %[[DIM_CAST]]
// CHECK:        %[[TO:.*]] = torch_c.to_i64 %[[FROM]]
// CHECK:        %[[CAST:.*]] = arith.index_cast %[[TO]] : i64 to index
// CHECK:        %[[B_RESULT:.*]] = tcp.broadcast %[[ARG0_T]], %[[CAST]] {axes = [0]} : tensor<1x2xf32>, index -> tensor<?x2xf32>
// CHECK:        %[[OUT:.*]] = torch_c.from_builtin_tensor %[[B_RESULT]] : tensor<?x2xf32> -> !torch.vtensor<[?,2],f32>
// CHECK:        return %[[OUT]] : !torch.vtensor<[?,2],f32>
func.func @torch.aten.broadcast_to_dynamic_dim(%arg0: !torch.vtensor<[1,2],f32>, %arg1: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,2],f32> {
  %int0 = torch.constant.int 0
  %0 = torch.aten.size.int %arg1, %int0 : !torch.vtensor<[?],f32>, !torch.int -> !torch.int
  %int-1 = torch.constant.int -1
  %1 = torch.prim.ListConstruct %0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
  %2 = torch.aten.broadcast_to %arg0, %1  : !torch.vtensor<[1,2],f32>, !torch.list<int> -> !torch.vtensor<[?,2],f32>
  return %2 : !torch.vtensor<[?,2],f32>
}

// -----

// CHECK-LABEL:  @symbolic_shape_ops(
// CHECK-SAME:       %[[ARG0:.*]]: !torch.vtensor<[?,?,3],f32>, %[[ARG1:.*]]: !torch.vtensor<[?,?,3],f32>, %[[ARG2:.*]]: !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,3],f32> {
// CHECK:        %[[S0:.*]] = tcp.symbolic_int "s0" {min_val = 5, max_val = 10} : i64
// CHECK:        %[[S1:.*]] = tcp.symbolic_int "s1" {min_val = 0, max_val = 100} : i64
// CHECK:        %[[S3:.*]] = tcp.symbolic_int "s3" {min_val = 0, max_val = 50} : i64
// CHECK:        %[[S5:.*]] = tcp.symbolic_int "s5" {min_val = 0, max_val = {{[0-9]+}}} : i64
// CHECK:        tcp.bind_symbolic_shape %{{.*}}, [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
// CHECK:        tcp.bind_symbolic_shape %{{.*}}, [%[[S0]], %[[S3]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
// CHECK:        tcp.bind_symbolic_shape %{{.*}}, [%[[S0]], %[[S5]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
// CHECK:        %[[TANH:.*]] = tcp.tanh %{{.*}} : tensor<?x?x3xf32> -> tensor<?x?x3xf32>
// CHECK:        tcp.bind_symbolic_shape %[[TANH]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
// CHECK:        %[[SIGM:.*]] = tcp.sigmoid %{{.*}} : tensor<?x?x3xf32> -> tensor<?x?x3xf32>
// CHECK:        tcp.bind_symbolic_shape %[[SIGM]], [%[[S0]], %[[S3]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : tensor<?x?x3xf32>
// CHECK:        %[[CAT:.*]] = tensor.concat dim(1) %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<?x?x3xf32>, tensor<?x?x3xf32>, tensor<?x?x3xf32>, tensor<?x?x3xf32>) -> tensor<?x?x3xf32>
// CHECK:        tcp.bind_symbolic_shape %[[CAT]], [%[[S0]], %[[S1]], %[[S3]], %[[S5]]], affine_map<()[s0, s1, s2, s3] -> (s0, s2 + s3 + s1 * 2, 3)> : tensor<?x?x3xf32>
// CHECK:        return %{{.*}} : !torch.vtensor<[?,?,3],f32>
func.func @symbolic_shape_ops(%arg0: !torch.vtensor<[?,?,3],f32>, %arg1: !torch.vtensor<[?,?,3],f32>, %arg2: !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,3],f32> {
  %0 = torch.symbolic_int "s0" {min_val = 5, max_val = 10} : !torch.int
  %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 100} : !torch.int
  %2 = torch.symbolic_int "s3" {min_val = 0, max_val = 50} : !torch.int
  %3 = torch.symbolic_int "s5" {min_val = 0, max_val = 9223372036854775806} : !torch.int
  torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
  torch.bind_symbolic_shape %arg1, [%0, %2], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
  torch.bind_symbolic_shape %arg2, [%0, %3], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
  %4 = torch.aten.tanh %arg0 : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
  torch.bind_symbolic_shape %4, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
  %5 = torch.aten.sigmoid %arg1 : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
  torch.bind_symbolic_shape %5, [%0, %2], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
  %6 = torch.prim.ListConstruct %4, %4, %5, %arg2 : (!torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>) -> !torch.list<vtensor>
  %int1 = torch.constant.int 1
  %7 = torch.aten.cat %6, %int1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,3],f32>
  torch.bind_symbolic_shape %7, [%0, %1, %2, %3], affine_map<()[s0, s1, s2, s3] -> (s0, s2 + s3 + s1 * 2, 3)> : !torch.vtensor<[?,?,3],f32>
  return %7 : !torch.vtensor<[?,?,3],f32>
}
