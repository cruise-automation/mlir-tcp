# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch.export import Dim

from tools.aot.torch_loader_utils import TorchLoaderOutput


def add_mul_single_output_loader() -> TorchLoaderOutput:
    class AddMulSingleOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            add = torch.add(x, y)
            mul = torch.mul(add, z)
            return mul

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = torch.randn(2, 3)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {
        "x": {0: dim_0, 1: dim_1},
        "y": {0: dim_0, 1: dim_1},
        "z": {0: dim_0, 1: dim_1},
    }

    return TorchLoaderOutput(
        model=AddMulSingleOutput(),
        inputs=(x, y, z),
        dynamic_shapes=dynamic_shapes,
    )


def add_mul_multi_output_loader() -> TorchLoaderOutput:
    class AddMulMultiOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor]:
            add = torch.add(x, y)
            mul = torch.mul(add, z)
            return add, mul

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = torch.randn(2, 3)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {
        "x": {0: dim_0, 1: dim_1},
        "y": {0: dim_0, 1: dim_1},
        "z": {0: dim_0, 1: dim_1},
    }

    return TorchLoaderOutput(
        model=AddMulMultiOutput(),
        inputs=(x, y, z),
        dynamic_shapes=dynamic_shapes,
    )


def add_tensor_mixed_ranks_loader() -> TorchLoaderOutput:
    class AddTensorMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.add(x, y)
            return add

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {
        "x": None,
        "y": {0: batch},
    }

    return TorchLoaderOutput(
        model=AddTensorMixedRanks(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def add_tensor_with_alpha_loader() -> TorchLoaderOutput:
    class AddTensorWithAlpha(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.add(x, y, alpha=2)
            return add

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {
        "x": {0: batch},
        "y": {0: batch},
    }

    return TorchLoaderOutput(
        model=AddTensorWithAlpha(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def sub_tensor_with_alpha_loader() -> TorchLoaderOutput:
    class SubTensorWithAlpha(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.sub(x, y, alpha=2)
            return add

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {
        "x": {0: batch},
        "y": {0: batch},
    }

    return TorchLoaderOutput(
        model=SubTensorWithAlpha(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def div_tensor_mixed_ranks_loader() -> TorchLoaderOutput:
    class DivTensorMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.div(x, y)
            return add

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {
        "x": None,
        "y": {0: batch},
    }

    return TorchLoaderOutput(
        model=DivTensorMixedRanks(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def add_sub_mul_div_scalar_loader() -> TorchLoaderOutput:
    class AddSubMulDivScalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.int1 = 1
            self.int2 = 2
            self.int3 = 3
            self.int4 = 4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            add = torch.add(x, self.int1)
            sub = torch.sub(add, self.int2)
            mul = torch.mul(sub, self.int3)
            div = torch.div(mul, self.int4)
            return div

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {
        "x": {0: batch},
    }

    return TorchLoaderOutput(
        model=AddSubMulDivScalar(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )


def sigmoid_loader() -> TorchLoaderOutput:
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {
        "x": {0: dim_0, 1: dim_1},
    }

    return TorchLoaderOutput(
        model=Sigmoid(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )


def tanh_loader() -> TorchLoaderOutput:
    class Tanh(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {
        "x": {0: dim_0, 1: dim_1},
    }

    return TorchLoaderOutput(
        model=Tanh(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )


def clamp_loader() -> TorchLoaderOutput:
    class Clamp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.float1 = 1e-01
            self.float2 = 1.024e01

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x, self.float1, self.float2)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {
        "x": {0: dim_0, 1: dim_1},
    }

    return TorchLoaderOutput(
        model=Clamp(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )


def concat_float_tensors_loader() -> TorchLoaderOutput:
    class ConcatFloatTensors(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.cat((x, y), dim=0)

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(3, 3)

    # Dynamic dim constraints
    batch_x = Dim("batch_x")
    batch_y = Dim("batch_y")
    dynamic_shapes = {
        "x": {0: batch_x},
        "y": {0: batch_y},
    }

    return TorchLoaderOutput(
        model=ConcatFloatTensors(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def concat_int_tensors_loader() -> TorchLoaderOutput:
    class ConcatIntTensors(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.cat((x, y), dim=0)

    # Sample inputs
    x = torch.randint(0, 5, (2, 3), dtype=torch.int32)
    y = torch.randint(2, 8, (3, 3), dtype=torch.int32)

    # Dynamic dim constraints
    batch_x = Dim("batch_x")
    batch_y = Dim("batch_y")
    dynamic_shapes = {
        "x": {0: batch_x},
        "y": {0: batch_y},
    }

    return TorchLoaderOutput(
        model=ConcatIntTensors(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def slice_tensor_loader() -> TorchLoaderOutput:
    class SliceTensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[0:2, :1]

    # Sample inputs
    x = torch.randn(4, 3)

    # Dynamic dim constraints
    batch = Dim("batch", min=3)
    dynamic_shapes = {
        "x": {0: batch},
    }

    return TorchLoaderOutput(
        model=SliceTensor(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )
