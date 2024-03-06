# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, NamedTuple, Optional

import torch


class TorchLoaderOutput(NamedTuple):
    model: torch.nn.Module
    inputs: List[torch.Tensor]
    constraints: Optional[List[torch.export.dynamic_dim]] = None
    func_name: Optional[str] = "func_main"
