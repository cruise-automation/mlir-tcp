// RUN: tcp-opt --split-input-file -transform-interpreter -canonicalize -cse %s | FileCheck %s

// CHECK: func.func @fuse_tcp_slice(%[[ARG0:.+]]: tensor<40x40xf32>) -> tensor<32x32xf32> {
// CHECK:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:   %[[C5:.+]] = arith.constant 5 : index
// CHECK:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:   %[[VAL0:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:   %[[VAL1:.+]] = scf.for %arg1 = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%arg2 = %[[VAL0]]) -> (tensor<32x32xf32>) {
// CHECK:     %[[VAL2:.+]] = scf.for %arg3 = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%arg4 = %arg2) -> (tensor<32x32xf32>) {
// CHECK:       %[[VAL3:.+]] = arith.addi %arg1, %[[C3]] : index
// CHECK:       %[[VAL4:.+]] = arith.addi %arg3, %[[C5]] : index
// CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[VAL3]], %[[VAL4]]] [1, 1] [1, 1] : tensor<40x40xf32> to tensor<1x1xf32>
// CHECK:       %[[VAL5:.+]] = tensor.empty() : tensor<1x1xf32>
// CHECK:       %[[VAL6:.+]] = linalg.elemwise_binary ins(%[[SLICE0]], %[[SLICE0]] : tensor<1x1xf32>, tensor<1x1xf32>) outs(%[[VAL5]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:       %[[SLICE1:.+]] = tcp.slice %[[VAL6]] starts(%[[C0]], %[[C0]]) sizes(%[[C1]], %[[C1]]) strides(%[[C1]], %[[C1]]) : tensor<1x1xf32> -> tensor<1x1xf32>
// CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<32x32xf32> to tensor<1x1xf32>
// CHECK:       %[[UNARY:.+]] = linalg.elemwise_unary ins(%[[SLICE1]] : tensor<1x1xf32>) outs(%[[SLICE2]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:       %[[ISLICE:.+]] = tensor.insert_slice %[[UNARY]] into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<32x32xf32>
// CHECK:       scf.yield %[[ISLICE]] : tensor<32x32xf32>
// CHECK:     }
// CHECK:     scf.yield %[[VAL2]] : tensor<32x32xf32>
// CHECK:   }
// CHECK:   return %[[VAL1]] : tensor<32x32xf32>
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

// -----

// CHECK: func.func @fuse_tcp_slice_stride2(%[[ARG0:.+]]: tensor<40x40xf32>) -> tensor<32x16xf32> {
// CHECK:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:   %[[C5:.+]] = arith.constant 5 : index
// CHECK:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:   %[[VAL0:.+]] = tensor.empty() : tensor<32x16xf32>
// CHECK:   %[[VAL1:.+]] = scf.for %arg1 = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%arg2 = %[[VAL0]]) -> (tensor<32x16xf32>) {
// CHECK:     %[[VAL2:.+]] = scf.for %arg3 = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%arg4 = %arg2) -> (tensor<32x16xf32>) {
// CHECK:       %[[VAL3:.+]] = arith.addi %arg1, %[[C3]] : index
// CHECK:       %[[MUL1:.+]] = arith.muli %arg3, %[[C2]] : index
// CHECK:       %[[VAL4:.+]] = arith.addi %[[MUL1]], %[[C5]] : index
// CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[VAL3]], %[[VAL4]]] [1, 2] [1, 1] : tensor<40x40xf32> to tensor<1x2xf32>
// CHECK:       %[[VAL5:.+]] = tensor.empty() : tensor<1x2xf32>
// CHECK:       %[[VAL6:.+]] = linalg.elemwise_binary ins(%[[SLICE0]], %[[SLICE0]] : tensor<1x2xf32>, tensor<1x2xf32>) outs(%[[VAL5]] : tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK:       %[[SLICE1:.+]] = tcp.slice %[[VAL6]] starts(%[[C0]], %[[C0]]) sizes(%[[C1]], %[[C1]]) strides(%[[C1]], %[[C2]]) : tensor<1x2xf32> -> tensor<1x1xf32>
// CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<32x16xf32> to tensor<1x1xf32>
// CHECK:       %[[UNARY:.+]] = linalg.elemwise_unary ins(%[[SLICE1]] : tensor<1x1xf32>) outs(%[[SLICE2]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:       %[[ISLICE:.+]] = tensor.insert_slice %[[UNARY]] into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<32x16xf32>
// CHECK:       scf.yield %[[ISLICE]] : tensor<32x16xf32>
// CHECK:     }
// CHECK:     scf.yield %[[VAL2]] : tensor<32x16xf32>
// CHECK:   }
// CHECK:   return %[[VAL1]] : tensor<32x16xf32>
func.func @fuse_tcp_slice_stride2(%arg0: tensor<40x40xf32>) -> tensor<32x16xf32> {
    %shape40 = tensor.empty() : tensor<40x40xf32>

    %0 = linalg.elemwise_binary ins(%arg0, %arg0 : tensor<40x40xf32>, tensor<40x40xf32>)
                             outs(%shape40: tensor<40x40xf32>) -> tensor<40x40xf32>

    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %slice = tcp.slice %0 starts ( %c3, %c5 ) sizes ( %c32, %c16 ) strides ( %c1, %c2 ) : tensor<40x40xf32> -> tensor<32x16xf32>

    %shape = tensor.empty() : tensor<32x16xf32>
    %ret = linalg.elemwise_unary ins(%slice: tensor<32x16xf32>) outs(%shape: tensor<32x16xf32>) -> tensor<32x16xf32>

    return %ret : tensor<32x16xf32>
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

// -----

// CHECK: func.func @fuse_tcp_slice_multiple(%[[ARG0:.+]]: tensor<40x40xf32>) -> tensor<16x16xf32> {
// CHECK:   %[[C8:.+]] = arith.constant 8 : index
// CHECK:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:   %[[VAL0:.+]] = tensor.empty() : tensor<16x16xf32>
// CHECK:   %[[VAL1:.+]] = scf.for %arg1 = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%arg2 = %[[VAL0]]) -> (tensor<16x16xf32>) {
// CHECK:     %[[VAL2:.+]] = scf.for %arg3 = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%arg4 = %arg2) -> (tensor<16x16xf32>) {
// CHECK:       %[[VAL3:.+]] = arith.addi %arg1, %[[C8]] : index
// CHECK:       %[[VAL4:.+]] = arith.addi %arg3, %[[C8]] : index
// CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[VAL3]], %[[VAL4]]] [1, 1] [1, 1] : tensor<40x40xf32> to tensor<1x1xf32>
// CHECK:       %[[VAL5:.+]] = tensor.empty() : tensor<1x1xf32>
// CHECK:       %[[VAL6:.+]] = linalg.elemwise_binary ins(%[[SLICE0]], %[[SLICE0]] : tensor<1x1xf32>, tensor<1x1xf32>) outs(%[[VAL5]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:       %[[SLICE1:.+]] = tcp.slice %[[VAL6]] starts(%[[C0]], %[[C0]]) sizes(%[[C1]], %[[C1]]) strides(%[[C1]], %[[C1]]) : tensor<1x1xf32> -> tensor<1x1xf32>
// CHECK:       %[[SLICE2:.+]] = tcp.slice %[[SLICE1]] starts(%[[C0]], %[[C0]]) sizes(%[[C1]], %[[C1]]) strides(%[[C1]], %[[C1]]) : tensor<1x1xf32> -> tensor<1x1xf32>
// CHECK:       %[[SLICE3:.+]] = tensor.extract_slice %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<16x16xf32> to tensor<1x1xf32>
// CHECK:       %[[UNARY:.+]] = linalg.elemwise_unary ins(%[[SLICE2]] : tensor<1x1xf32>) outs(%[[SLICE3]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:       %[[ISLICE:.+]] = tensor.insert_slice %[[UNARY]] into %arg4[%arg1, %arg3] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<16x16xf32>
// CHECK:       scf.yield %[[ISLICE]] : tensor<16x16xf32>
// CHECK:     }
// CHECK:     scf.yield %[[VAL2]] : tensor<16x16xf32>
// CHECK:   }
// CHECK:   return %[[VAL1]] : tensor<16x16xf32>
func.func @fuse_tcp_slice_multiple(%arg0: tensor<40x40xf32>) -> tensor<16x16xf32> {
    %shape40 = tensor.empty() : tensor<40x40xf32>

    %0 = linalg.elemwise_binary ins(%arg0, %arg0 : tensor<40x40xf32>, tensor<40x40xf32>)
                             outs(%shape40: tensor<40x40xf32>) -> tensor<40x40xf32>

    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %slice = tcp.slice %0 starts ( %c3, %c5 ) sizes ( %c32, %c32 ) strides ( %c1, %c1 ) : tensor<40x40xf32> -> tensor<32x32xf32>
    %slice2 = tcp.slice %slice starts ( %c5, %c3 ) sizes ( %c16, %c16 ) strides ( %c1, %c1 ) : tensor<32x32xf32> -> tensor<16x16xf32>

    %shape = tensor.empty() : tensor<16x16xf32>
    %ret = linalg.elemwise_unary ins(%slice2: tensor<16x16xf32>) outs(%shape: tensor<16x16xf32>) -> tensor<16x16xf32>

    return %ret : tensor<16x16xf32>
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

// -----

// CHECK: func.func @fuse_tcp_slice_non_unit_tile(%[[ARG0:.+]]: tensor<40x40xf32>) -> tensor<32x32xf32> {
// CHECK:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:   %[[C5:.+]] = arith.constant 5 : index
// CHECK:   %[[C3:.+]] = arith.constant 3 : index
// CHECK:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:   %[[VAL0:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:   %[[VAL1:.+]] = scf.for %arg1 = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%arg2 = %[[VAL0]]) -> (tensor<32x32xf32>) {
// CHECK:     %[[VAL2:.+]] = scf.for %arg3 = %[[C0]] to %[[C32]] step %[[C4]] iter_args(%arg4 = %arg2) -> (tensor<32x32xf32>) {
// CHECK:       %[[VAL3:.+]] = arith.addi %arg1, %[[C3]] : index
// CHECK:       %[[VAL4:.+]] = arith.addi %arg3, %[[C5]] : index
// CHECK:       %[[SLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[VAL3]], %[[VAL4]]] [4, 4] [1, 1] : tensor<40x40xf32> to tensor<4x4xf32>
// CHECK:       %[[VAL5:.+]] = tensor.empty() : tensor<4x4xf32>
// CHECK:       %[[VAL6:.+]] = linalg.elemwise_binary ins(%[[SLICE0]], %[[SLICE0]] : tensor<4x4xf32>, tensor<4x4xf32>) outs(%[[VAL5]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:       %[[SLICE1:.+]] = tcp.slice %[[VAL6]] starts(%[[C0]], %[[C0]]) sizes(%[[C4]], %[[C4]]) strides(%[[C1]], %[[C1]]) : tensor<4x4xf32> -> tensor<4x4xf32>
// CHECK:       %[[SLICE2:.+]] = tensor.extract_slice %arg4[%arg1, %arg3] [4, 4] [1, 1] : tensor<32x32xf32> to tensor<4x4xf32>
// CHECK:       %[[UNARY:.+]] = linalg.elemwise_unary ins(%[[SLICE1]] : tensor<4x4xf32>) outs(%[[SLICE2]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK:       %[[ISLICE:.+]] = tensor.insert_slice %[[UNARY]] into %arg4[%arg1, %arg3] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<32x32xf32>
// CHECK:       scf.yield %[[ISLICE]] : tensor<32x32xf32>
// CHECK:     }
// CHECK:     scf.yield %[[VAL2]] : tensor<32x32xf32>
// CHECK:   }
// CHECK:   return %[[VAL1]] : tensor<32x32xf32>
func.func @fuse_tcp_slice_non_unit_tile(%arg0: tensor<40x40xf32>) -> tensor<32x32xf32> {
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

    %1, %loops:2 = transform.structured.fuse %unary {tile_sizes = [4, 4], tile_interchange = [0, 1]}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.tensor.fold_tensor_subset_ops
    } : !transform.op<"func.func">

    transform.yield
  }
}
