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
        model=AddMulSingleOutput(), inputs=(x, y, z), dynamic_shapes=dynamic_shapes
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
        model=AddMulMultiOutput(), inputs=(x, y, z), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": None, "y": {0: batch}}

    return TorchLoaderOutput(
        model=AddTensorMixedRanks(), inputs=(x, y), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch}}

    return TorchLoaderOutput(
        model=AddTensorWithAlpha(), inputs=(x, y), dynamic_shapes=dynamic_shapes
    )


def sub_tensor_with_alpha_loader() -> TorchLoaderOutput:
    class SubTensorWithAlpha(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            sub = torch.sub(x, y, alpha=2)
            return sub

    # Sample inputs
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch}}

    return TorchLoaderOutput(
        model=SubTensorWithAlpha(), inputs=(x, y), dynamic_shapes=dynamic_shapes
    )


def div_tensor_mixed_ranks_loader() -> TorchLoaderOutput:
    class DivTensorMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            div = torch.div(x, y)
            return div

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": None, "y": {0: batch}}

    return TorchLoaderOutput(
        model=DivTensorMixedRanks(), inputs=(x, y), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=AddSubMulDivScalar(), inputs=(x,), dynamic_shapes=dynamic_shapes
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
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=Sigmoid(), inputs=(x,), dynamic_shapes=dynamic_shapes
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
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(model=Tanh(), inputs=(x,), dynamic_shapes=dynamic_shapes)


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
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(model=Clamp(), inputs=(x,), dynamic_shapes=dynamic_shapes)


def relu_loader() -> TorchLoaderOutput:
    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(model=Relu(), inputs=(x,), dynamic_shapes=dynamic_shapes)


def log1p_loader() -> TorchLoaderOutput:
    class Log1p(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.log1p(x)

    # Sample inputs
    x = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(model=Log1p(), inputs=(x,), dynamic_shapes=dynamic_shapes)


def round_even_loader() -> TorchLoaderOutput:
    class RoundEven(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.round(x)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=RoundEven(), inputs=(x,), dynamic_shapes=dynamic_shapes
    )


def sqrt_float_loader() -> TorchLoaderOutput:
    class SqrtFloat(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x)

    # Sample inputs
    x = torch.abs(torch.randn(2, 3))

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=SqrtFloat(), inputs=(x,), dynamic_shapes=dynamic_shapes
    )


def sqrt_int_loader() -> TorchLoaderOutput:
    class SqrtInt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(x)

    # Sample inputs
    x = torch.randint(0, 5, (2, 3), dtype=torch.int32)

    # Dynamic dim constraints
    batch = Dim("batch")
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=SqrtInt(), inputs=(x,), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": {0: batch_x}, "y": {0: batch_y}}

    return TorchLoaderOutput(
        model=ConcatFloatTensors(), inputs=(x, y), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": {0: batch_x}, "y": {0: batch_y}}

    return TorchLoaderOutput(
        model=ConcatIntTensors(), inputs=(x, y), dynamic_shapes=dynamic_shapes
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
    dynamic_shapes = {"x": {0: batch}}

    return TorchLoaderOutput(
        model=SliceTensor(), inputs=(x,), dynamic_shapes=dynamic_shapes
    )


def broadcast_unit_dim_to_static_with_explicit_dim_static_loader() -> TorchLoaderOutput:
    class BroadcastUnitDimToStaticWithExplicitDimStatic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (3, 2))

    # Sample inputs
    x = torch.randn(1, 2)

    return TorchLoaderOutput(
        model=BroadcastUnitDimToStaticWithExplicitDimStatic(), inputs=(x,)
    )


def broadcast_unit_dim_to_static_with_unchanged_dim_static_loader() -> (
    TorchLoaderOutput
):
    class BroadcastUnitDimToStaticWithUnchangedDimStatic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (3, -1))

    # Sample inputs
    x = torch.randn(1, 2)

    return TorchLoaderOutput(
        model=BroadcastUnitDimToStaticWithUnchangedDimStatic(), inputs=(x,)
    )


def broadcast_unit_dim_to_static_with_unchanged_dim_dynamic_loader() -> (
    TorchLoaderOutput
):
    class BroadcastUnitDimToStaticWithUnchangedDimDynamic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (3, -1))

    # Sample inputs
    x = torch.randn(1, 2)

    dim_1 = Dim("dim_1")
    dynamic_shapes = {"x": {1: dim_1}}

    return TorchLoaderOutput(
        model=BroadcastUnitDimToStaticWithUnchangedDimDynamic(),
        inputs=(x,),
        dynamic_shapes=dynamic_shapes,
    )


def broadcast_unit_dim_to_dynamic_with_unchanged_dim_static_loader() -> (
    TorchLoaderOutput
):
    class BroadcastUnitDimToDynamicWithUnchangedDimStatic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (y.shape[0], -1))

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(10)

    dim_0 = Dim("dim_0")
    dynamic_shapes = {"x": {}, "y": {0: dim_0}}

    return TorchLoaderOutput(
        model=BroadcastUnitDimToDynamicWithUnchangedDimStatic(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def broadcast_unit_dim_to_dynamic_with_unchanged_dim_dynamic_loader() -> (
    TorchLoaderOutput
):
    class BroadcastUnitDimToDynamicWithUnchangedDimDynamic(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, (y.shape[0], -1))

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(10)

    dim_0 = Dim("dim_0")
    dim_1 = Dim("dim_1")
    dynamic_shapes = {"x": {1: dim_1}, "y": {0: dim_0}}

    return TorchLoaderOutput(
        model=BroadcastUnitDimToDynamicWithUnchangedDimDynamic(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def broadcast_unit_dim_to_static_with_rank_increase_loader() -> TorchLoaderOutput:
    class BroadcastUnitDimToStaticWithRankIncrease(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, y.size())

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(4, 3, 2)

    return TorchLoaderOutput(
        model=BroadcastUnitDimToStaticWithRankIncrease(), inputs=(x, y)
    )


def broadcast_unit_dim_to_dynamic_with_rank_increase_loader() -> TorchLoaderOutput:
    class BroadcastUnitDimToDynamicWithRankIncrease(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.broadcast_to(x, y.size())

    # Sample inputs
    x = torch.randn(1, 2)
    y = torch.randn(4, 3, 2)

    dim_0 = Dim("dim_0")
    dynamic_shapes = {"x": {}, "y": {0: dim_0}}

    return TorchLoaderOutput(
        model=BroadcastUnitDimToDynamicWithRankIncrease(),
        inputs=(x, y),
        dynamic_shapes=dynamic_shapes,
    )


def gather_elements_loader() -> TorchLoaderOutput:
    class GatherElements(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.gather(x, 0, y)

    # Sample inputs
    x = torch.randn(4, 3)
    y = torch.tensor([[0, 0, 0], [1, 1, 1]])

    # Dynamic dim constraints
    batch = Dim("batch", min=3)
    dynamic_shapes = {"x": {0: batch}, "y": {}}

    return TorchLoaderOutput(
        model=GatherElements(), inputs=(x, y), dynamic_shapes=dynamic_shapes
    )


def gather_slices_loader() -> TorchLoaderOutput:
    class GatherSlices(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.index_select(x, 1, y)

    # Sample inputs
    x = torch.randn(4, 3)
    y = torch.tensor([2, 0])

    # Dynamic dim constraints
    batch = Dim("batch", min=3)
    dynamic_shapes = {"x": {0: batch}, "y": {}}

    return TorchLoaderOutput(
        model=GatherSlices(), inputs=(x, y), dynamic_shapes=dynamic_shapes
    )


def index_hacked_twin_loader() -> TorchLoaderOutput:
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # not using dynamic dim currently as the i1 tensor would ideally
            # be generated conditioned on the shape
            i1 = torch.tensor([[0], [1], [2], [3]])
            return x[i1, [2, 5, 7]]

    x = torch.rand(4, 10)

    return TorchLoaderOutput(
        model=Model(),
        inputs=(x,),
    )

def slice_write_back_loader() -> TorchLoaderOutput:
    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor, i3: torch.Tensor) -> torch.Tensor:
            x[i1, i2, i3] = y
            return x

    model = Model()
    x = torch.rand(10,10,10)
    y = torch.rand(7)
    i1 = torch.arange(7)
    i2 = torch.arange(7) + 1
    i3 = torch.tensor([0])

    return TorchLoaderOutput(
        model=model,
        inputs=(x, y, i1, i2, i3),
    )
