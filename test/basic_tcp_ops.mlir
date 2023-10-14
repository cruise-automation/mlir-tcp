func.func @func_1(%arg0: tensor<?x?xf32>,
                  %arg1: tensor<?x?xf32>,
                  %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.mul %0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

func.func @func_2(%arg0: tensor<?x?xf32>,
                  %arg1: tensor<?x?xf32>,
                  %arg2: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %0 = tcp.add %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  %1 = tcp.mul %0, %arg2 : tensor<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
  return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
}

func.func @func_3(%arg0: tensor<f32>,
                  %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
  %arg0_ex = tensor.expand_shape %arg0 [] : tensor<f32> into tensor<1xf32>
  %arg0_bcast = tcp.broadcast %arg0_ex, %dim {axes = [0]} : tensor<1xf32>, index -> tensor<?xf32>
  %0 = tcp.add %arg0_bcast, %arg1 : tensor<?xf32>, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
