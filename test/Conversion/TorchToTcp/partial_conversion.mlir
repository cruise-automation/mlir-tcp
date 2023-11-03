// RUN: tcp-opt %s -convert-torch-to-tcp="convert-torch-ops=aten.atan"    -verify-diagnostics | FileCheck %s -check-prefix=CHECK1
// RUN: tcp-opt %s -convert-torch-to-tcp="convert-torch-ops=aten.log"     -verify-diagnostics | FileCheck %s -check-prefix=CHECK2
// RUN: tcp-opt %s -convert-torch-to-tcp="convert-torch-ops=aten.sigmoid" -verify-diagnostics | FileCheck %s -check-prefix=CHECK3
// RUN: tcp-opt %s -convert-torch-to-tcp                                  -verify-diagnostics | FileCheck %s -check-prefix=CHECK4

// CHECK1-LABEL:  func.func @torch.aten.atan.log(
// CHECK1-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK1:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK1:         %[[T1:.*]] = tcp.atan %[[T0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK1:         %[[T2:.*]] = torch_c.from_builtin_tensor %[[T1]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK1:         %[[T3:.*]] = torch.aten.log %[[T2]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK1:         return %[[T3]] : !torch.vtensor<[?,?],f32>

// CHECK2-LABEL:  func.func @torch.aten.atan.log(
// CHECK2-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK2:         %[[T0:.*]] = torch.aten.atan %[[ARG0]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK2:         %[[T1:.*]] = torch_c.to_builtin_tensor %[[T0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK2:         %[[T2:.*]] = tcp.log %[[T1]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK2:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK2:         return %[[T3]] : !torch.vtensor<[?,?],f32>

// CHECK3-LABEL:  func.func @torch.aten.atan.log(
// CHECK3-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK3:         %[[T0:.*]] = torch.aten.atan %[[ARG0]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK3:         %[[T1:.*]] = torch.aten.log %[[T0]] : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
// CHECK3:         return %[[T1]] : !torch.vtensor<[?,?],f32>

// CHECK4-LABEL:  func.func @torch.aten.atan.log(
// CHECK4-SAME:         %[[ARG0:.*]]: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
// CHECK4:         %[[T0:.*]] = torch_c.to_builtin_tensor %[[ARG0]] : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
// CHECK4:         %[[T1:.*]] = tcp.atan %[[T0]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK4:         %[[T2:.*]] = tcp.log %[[T1]] : tensor<?x?xf32> -> tensor<?x?xf32>
// CHECK4:         %[[T3:.*]] = torch_c.from_builtin_tensor %[[T2]] : tensor<?x?xf32> -> !torch.vtensor<[?,?],f32>
// CHECK4:         return %[[T3]] : !torch.vtensor<[?,?],f32>

func.func @torch.aten.atan.log(%arg0: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
  %0 = torch.aten.atan %arg0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  %1 = torch.aten.log %0 : !torch.vtensor<[?,?],f32> -> !torch.vtensor<[?,?],f32>
  return %1 : !torch.vtensor<[?,?],f32>
}
