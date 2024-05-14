# Gather Ops in Tcp

## Gather elements along a given dim

`tcp.gather` op gathers elements from a given tensor based on indices that index along a given dim.

Syntax:

    operation ::= `tcp.gather` $input `,` $indices attr-dict `:`
                          type($input) `,` type($indices) `->` type($out)

Attributes:

    dim : index

Inputs:

    input : tensor of any supported type, rank r
    indices : tensor of int64, rank r

Output:

    out : tensor of any supported type, rank r, same shape as indices

Semantics:

    For rank 2 input and indices:
        out[i][j] = input[index[i][j]][j]  # if dim == 0
        out[i][j] = input[i][index[i][j]]  # if dim == 1

    For rank 3 input and indices:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

This op is similar to `torch.gather` [[1]](https://pytorch.org/docs/stable/generated/torch.gather.html) and `onnx.GatherElements` [[2]](https://onnx.ai/onnx/operators/onnx__GatherElements.html#l-onnx-doc-gatherelements).

### Examples

1. Modeling `torch.gather`

        input = torch.randn(3, 4)
        indices = torch.tensor([[0, 1], [2, 0], [2, 3]]) # Shape is [3, 2]
        x = torch.gather(input, 1, indices) # Result shape is [3, 2]

    This will get mapped to `tcp.gather` as follows:

        %input = ...
        %indices = ...
        %x = tcp.gather %input, %indices { dim = 1 } :
                (tensor<3x4xf32>, tensor<3x2xi64>) -> tensor<3x2xf32>

2. Modeling `onnx.GatherElements`

        input = ... # Shape is [3, 3]
        indices = ... # Shape is [2, 3]
        x = onnx.GatherElements(input, indices, 0) # Result shape is [2, 3]

    This will get mapped to `tcp.gather` as follows:

        %input = ...
        %indices = ...
        %x = tcp.gather %input, %indices { dim = 0 } :
                (tensor<3x3xf32>, tensor<2x3xi64>) -> tensor<2x3xf32>


## Gather slices along a given dim

This requires gathering slices from a given tensor based on indices that index along a given dim.

Our design is to use `tcp.gather` op for these cases as follows. Suppose that the `input` has shape `[a, b, c]`, `indices` has shape `[x, y]` and `dim = 0`. Shape of `output` in this case will be `[x, y, b, c]`.
* Broadcast `input` from `[a, b, c]` to `[a, y, b, c]` by introducing `y` dim.
* Broadcast `indices` from `[x, y]` to `[x, y, b, c]` by introducing `b` and `c` dims.
* Perform `tcp.gather` on these broadcasted `input` and `indices`, whose `output` will now have the shape `[x, y, b, c]`.


This approach can be used to represent ops like `torch.index_select` [[3]](https://pytorch.org/docs/stable/generated/torch.index_select.html), `tf.gather` [[4]](https://www.tensorflow.org/api_docs/python/tf/gather), and `onnx.Gather` [[5]](https://onnx.ai/onnx/operators/onnx__Gather.html#l-onnx-doc-gather).

### Examples

1. Modeling `torch.index_select`

        input = torch.randn(3, 4)
        indices = torch.tensor([0, 2]) # Shape is [2]
        x = torch.index_select(input, 0, indices) # Result shape is [2, 4]

    This will get mapped to `tcp.gather` as follows:

        %input = ... # Shape is [3, 4]
        %indices = ... # Shape is [2]
        %indices_2d = tensor.expand_shape %indices [[0, 1]] :
                (tensor<2xi64>) -> tensor<2x1xi64>
        %cst4 = arith.constant 4 : index
        %indices_bcast = tcp.broadcast_to %indices_2d, %cst4 { axes = [1] } :
                (tensor<2x1xi64>, index) -> tensor<2x4xi64>
        %x = tcp.gather %input, %indices_bcast { dim = 0 } :
                (tensor<3x4xf32>, tensor<2x4xi64>) -> tensor<2x4xf32>

2. Modeling `tf.gather`

        input = ... # Shape is [3, 4, 5]
        indices = ... # Shape is [3]
        x = tf.gather(input, indices, axis=1) # Result shape is [3, 3, 5]

    This will get mapped to `tcp.gather` as follows:

        %input = ... # Shape is [3, 4, 5]
        %indices = ... # Shape is [3]
        %indices_3d = tensor.expand_shape %indices [[0, 1, 2]] :
                (tensor<3xi64>) -> tensor<1x3x1xi64>
        %indices_bcast = tcp.broadcast_to %indices_3d, %cst3, %cst5 { axes = [0, 2] } :
                (tensor<1x3x1xi64>, index, index) -> tensor<3x3x5xi64>
        %x = tcp.gather %input, %indices_bcast { dim = 1 } :
                (tensor<3x4x5xf32>, tensor<3x3x5xi64>) -> tensor<3x3x5xf32>

3. Modeling `onnx.Gather`

    This case is exactly similar to `tf.gather`.

### Alternative considered

We considered a separate `tcp.gather_slice` op for this particular case with the following design.

Syntax:

    operation ::= `tcp.gather_slice` $input `,` $indices attr-dict `:`
                          type($input) `,` type($indices) `->` type($out)

Attributes:

    dim : index

Inputs:

    input : tensor of any supported type, rank r
    indices : tensor of int64, rank q

Output:

    out : tensor of any supported type, rank r + q - 1

Semantics:

    For input of rank 2 and indices of rank 2:
        out[i][j][k] = input[indices[i][j]][k]   # if dim == 0
        out[i][j][k] = input[i][indices[j][k]]   # if dim == 1

    For input of rank 3 and indices of rank 2:
        out[i][j][k][m] = input[indices[i][j]][k][m]  # if dim == 0
        out[i][j][k][m] = input[i][indices[j][k]][m]  # if dim == 1
        out[i][j][k][m] = input[i][j][indices[k][m]]  # if dim == 2

The above approach of reusing `tcp.gather` is preferred to avoid adding a new op here.

## Gather slices along N dims

`tcp.gather_nd` op gathers slices from a given tensor based on indices that index along the first `n` dims.

Syntax:

    operation ::= `tcp.gather_nd` $input `,` $indices attr-dict `:`
                          type($input) `,` type($indices) `->` type($out)

Inputs:

    input : tensor of any supported type, rank r
    indices : tensor of int64, rank q

Output:

    out : tensor of any supported type, rank r + q - indices_shape[-1] - 1

Semantics:

    For input of rank 2 and indices of shape (N, 2):
        a, b = indices[i]
        out[i] = input[a][b]

    For input of rank 3 and indices of shape (N, 2):
        a, b = indices[i]
        out[i][j] = input[a][b][j]

    For input of rank 4 and indices of shape (N, 2):
        a, b = indices[i]
        out[i][j][k] = input[a][b][j][k]

    For input of rank 4 and indices of shape (N, 3):
        a, b, c = indices[i]
        out[i][j] = input[a][b][c][j]

This op can be used to represent ops like `tf.gather_nd` [[6]](https://www.tensorflow.org/api_docs/python/tf/gather_nd) and `onnx.GatherND` [[7]](https://onnx.ai/onnx/operators/onnx__GatherND.html#l-onnx-doc-gathernd), except for cases when they support batching.

### Examples

1. Modeling `tf.gather_nd` without batching

        input = ... # Shape is [5, 7, 3]
        indices = ... # Shape is [4, 2]
        x = tf.gather_nd(input, indices) # Result shape is [4, 3]

    This will get mapped to `tcp.gather_nd` as follows:

        %input = ... # Shape is [5, 7, 3]
        %indices = ... # Shape is [4, 2]
        %x = tcp.gather_nd %input, %indices :
                (tensor<5x7x3xf32>, tensor<4x2xi64>) -> tensor<4x3xf32>

2. Modeling `onnx.GatherND` without batching

    This case is exactly similar to `tf.gather_nd`.
