func.func @fuse_tcp_slice(%arg0: tensor<40x40xf32>) -> tensor<32x32xf32> {
    %shape40 = tensor.empty() : tensor<40x40xf32>

    %0 = linalg.elemwise_binary ins(%arg0, %arg0 : tensor<40x40xf32>, tensor<40x40xf32>)
                             outs(%shape40: tensor<40x40xf32>) -> tensor<40x40xf32>

    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %slice = tcp.slice %0 starts ( %c3, %c5 ) sizes ( %c32, %c32 ) strides ( %c1, %c1 ) : tensor<40x40xf32> -> tensor<32x32xf32>

    %shape = tensor.empty() : tensor<32x32xf32>
    %ret = linalg.elemwise_unary ins(%slice: tensor<32x32xf32>) outs(%shape: tensor<32x32xf32>) -> tensor<32x32xf32>

    return %ret : tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %unary = transform.structured.match ops{["linalg.elemwise_unary"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    %1, %loops:2 = transform.structured.fuse %unary {tile_sizes = [1, 1], tile_interchange = [0, 1]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.tensor.fold_tensor_subset_ops
    } : !transform.op<"func.func">

    transform.yield
  }
}