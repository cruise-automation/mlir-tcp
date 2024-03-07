// RUN: tcp-opt %s -convert-torch-to-tcp | FileCheck %s

// CHECK-LABEL:  func.func @main(
// CHECK:         tcp.tanh
// CHECK:         tcp.const {value = dense_resource<torch_tensor_1_4_torch.float32>
// CHECK:         arith.const
// CHECK:         arith.const
// CHECK:         tcp.broadcast
// CHECK:         tcp.mul
// CHECK:         tcp.const {value = dense_resource<torch_tensor_3_1_torch.float32>
// CHECK:         arith.const
// CHECK:         arith.const
// CHECK:         tcp.broadcast
// CHECK:         tcp.mul
// CHECK:         tcp.const {value = dense_resource<torch_tensor_1_1_torch.float32>
// CHECK:         arith.const
// CHECK:         arith.const
// CHECK:         arith.const
// CHECK:         arith.const
// CHECK:         tcp.broadcast
// CHECK:         tcp.mul
module {
  func.func @main(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f32> {
    %0 = torch.aten.tanh %arg0 : !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_1_4_torch.float32> : tensor<1x4xf32>) : !torch.vtensor<[1,4],f32>
    %2 = torch.aten.mul.Tensor %0, %1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32> -> !torch.vtensor<[3,4],f32>
    %3 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_torch.float32> : tensor<3x1xf32>) : !torch.vtensor<[3,1],f32>
    %4 = torch.aten.mul.Tensor %2, %3 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,1],f32> -> !torch.vtensor<[3,4],f32>
    %5 = torch.vtensor.literal(dense_resource<torch_tensor_1_1_torch.float32> : tensor<1x1xf32>) : !torch.vtensor<[1,1],f32>
    %6 = torch.aten.mul.Tensor %4, %5 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32> -> !torch.vtensor<[3,4],f32>
    return %6 : !torch.vtensor<[3,4],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_1_4_torch.float32: "0x04000000E76675BFEB9AFFBF91519D3F7571B2BE",
      torch_tensor_3_1_torch.float32: "0x040000001CE828BF28207ABF6BEE4C3F",
      torch_tensor_1_1_torch.float32: "0x040000000D78D73D"
    }
  }
#-}
