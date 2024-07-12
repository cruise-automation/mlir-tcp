# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

import torch
import torch.export
import torch.nn as nn
from torch.export import Dim

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


@run
# CHECK-LABEL: test_tanh_sigmoid_cat
# CHECK:      func.func @main(
# CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>,
# CHECK-SAME:       %[[ARG1:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>,
# CHECK-SAME:       %[[ARG2:[a-zA-Z0-9]+]]: !torch.vtensor<[?,?,3],f32>) -> !torch.vtensor<[?,?,3],f32> {
# CHECK:        %[[S0:.+]] = torch.symbolic_int "s0" {min_val = 5, max_val = 10} : !torch.int
# CHECK:        %[[S1:.+]] = torch.symbolic_int "s1" {min_val = {{[0-9]+}}, max_val = 100} : !torch.int
# CHECK:        %[[S2:.+]] = torch.symbolic_int "s3" {min_val = {{[0-9]+}}, max_val = 50} : !torch.int
# CHECK:        %[[S3:.+]] = torch.symbolic_int "s5" {min_val = {{[0-9]+}}, max_val = {{[0-9]+}}} : !torch.int
# CHECK:        torch.bind_symbolic_shape %[[ARG0]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG1]], [%[[S0]], %[[S2]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[ARG2]], [%[[S0]], %[[S3]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[TANH:.+]] = torch.aten.tanh %[[ARG0]] : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[TANH]], [%[[S0]], %[[S1]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[SIG:.+]] = torch.aten.sigmoid %[[ARG1]] : !torch.vtensor<[?,?,3],f32> -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[SIG]], [%[[S0]], %[[S2]]], affine_map<()[s0, s1] -> (s0, s1, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        %[[LIST:.+]] = torch.prim.ListConstruct %[[TANH]], %[[TANH]], %[[SIG]], %[[ARG2]] : (!torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>, !torch.vtensor<[?,?,3],f32>) -> !torch.list<vtensor>
# CHECK:        %[[CAT:.+]] = torch.aten.cat %[[LIST]], {{.*}} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,?,3],f32>
# CHECK:        torch.bind_symbolic_shape %[[CAT]], [%[[S0]], %[[S1]], %[[S2]], %[[S3]]], affine_map<()[s0, s1, s2, s3] -> (s0, s2 + s3 + s1 * 2, 3)> : !torch.vtensor<[?,?,3],f32>
# CHECK:        return %[[CAT]] : !torch.vtensor<[?,?,3],f32>
def test_tanh_sigmoid_cat():
    class TanhSigmoidCat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            a = torch.tanh(x)
            b = torch.sigmoid(y)
            return torch.cat((a, a, b, z), dim=1)

    # Sample inputs
    x = torch.randn(5, 2, 3)
    y = torch.randn(5, 6, 3)
    z = torch.randn(5, 4, 3)

    # Dynamic dim constraints
    dim_n = Dim("n", min=5, max=10)
    dim_x1 = Dim("x1", max=100)
    dim_y1 = Dim("y1", max=50)
    dim_z1 = Dim("z1")
    dynamic_shapes = {
        "x": {0: dim_n, 1: dim_x1},
        "y": {0: dim_n, 1: dim_y1},
        "z": {0: dim_n, 1: dim_z1},
    }

    m = fx.export_and_import(
        TanhSigmoidCat(),
        x,
        y,
        z,
        dynamic_shapes=dynamic_shapes,
        import_symbolic_shape_expressions=True,
    )
    print(m)
