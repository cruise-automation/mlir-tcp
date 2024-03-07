# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch.export import dynamic_dim

from tools.aot.torch_loader_utils import TorchLoaderOutput


def sigmoid_loader() -> TorchLoaderOutput:
    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x)

    # Sample inputs
    x = torch.randn(2, 3)

    # Dynamic dim constraints
    constraints = [
        dynamic_dim(x, 0),
        dynamic_dim(x, 1),
    ]

    return TorchLoaderOutput(
        model=Sigmoid(),
        inputs=[x],
        constraints=constraints,
    )
