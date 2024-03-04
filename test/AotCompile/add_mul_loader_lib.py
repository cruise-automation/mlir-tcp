# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch.export import dynamic_dim

from tools.aot.torch_loader_utils import TorchLoaderOutput


def add_mul_single_output_loader() -> TorchLoaderOutput:
    class AddMulNetSingleOutput(torch.nn.Module):
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
    constraints = [
        # Dim 1
        dynamic_dim(x, 0) == dynamic_dim(y, 0),
        dynamic_dim(y, 0) == dynamic_dim(z, 0),
        # Dim 2
        dynamic_dim(x, 1) == dynamic_dim(y, 1),
        dynamic_dim(y, 1) == dynamic_dim(z, 1),
    ]

    return TorchLoaderOutput(
        model=AddMulNetSingleOutput(),
        inputs=[x, y, z],
        constraints=constraints,
    )


def add_mul_multi_output_loader() -> TorchLoaderOutput:
    class AddMulNetMultiOutput(torch.nn.Module):
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
    constraints = [
        # Dim 1
        dynamic_dim(x, 0) == dynamic_dim(y, 0),
        dynamic_dim(y, 0) == dynamic_dim(z, 0),
        # Dim 2
        dynamic_dim(x, 1) == dynamic_dim(y, 1),
        dynamic_dim(y, 1) == dynamic_dim(z, 1),
    ]

    return TorchLoaderOutput(
        model=AddMulNetMultiOutput(),
        inputs=[x, y, z],
        constraints=constraints,
    )


def broadcast_add_mixed_ranks_loader() -> TorchLoaderOutput:
    class BroadcastAddMixedRanks(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            add = torch.add(x, y)
            return add

    # Sample inputs
    x = torch.tensor(10.0)
    y = torch.randn(2)

    # Dynamic dim constraints
    constraints = [dynamic_dim(y, 0)]

    return TorchLoaderOutput(
        model=BroadcastAddMixedRanks(),
        inputs=[x, y],
        constraints=constraints,
    )
