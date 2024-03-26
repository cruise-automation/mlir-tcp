# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import NamedTuple, Optional, Tuple, Union, Dict, Any

import torch


class TorchLoaderOutput(NamedTuple):
    model: torch.nn.Module
    inputs: Tuple[torch.Tensor]
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None
    func_name: Optional[str] = "func_main"
