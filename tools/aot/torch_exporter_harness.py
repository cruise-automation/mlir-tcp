# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import argparse
import importlib
import numpy as np

import torch
from torch_mlir import fx

from tools.aot.torch_loader_utils import TorchLoaderOutput

parser = argparse.ArgumentParser(
    description="Harness for running user provided torch_loader_lib to export Torch dialect programs"
)
parser.add_argument(
    "--torch_loader_path",
    required=True,
    help="Full python import path (dot separated) to the torch loader function.",
)
parser.add_argument(
    "--reference_tensors_path",
    required=False,
    help="Path to the file to save the reference inputs and outputs to (as .npz)",
)


def main():
    args = parser.parse_args()
    loader_module, loader_function = args.torch_loader_path.rsplit(".", 1)
    m = importlib.import_module(loader_module)
    loader_result = getattr(m, loader_function)()

    assert isinstance(
        loader_result, TorchLoaderOutput
    ), "Please use tools.aot.torch_loader_utils.TorchLoaderOutput as your torch_loader function's return type"
    assert isinstance(
        loader_result.inputs, tuple
    ), "Please provide Tuple[torch.Tensor] as TorchLoaderOutput.inputs in your torch_loader function"

    # Used by gen_{name}_mlir_torch genrule
    if not args.reference_tensors_path:
        # torch.export + fx_importer
        torch_program = fx.export_and_import(
            loader_result.model,
            *loader_result.inputs,  # unpack list of input tensors
            dynamic_shapes=loader_result.dynamic_shapes,
            import_symbolic_shape_expressions=True,
            # This is the Torch dialect imported from Dynamo/FX export and run
            # through `torchdynamo-export-to-torch-backend-pipeline` (which
            # runs `ReduceOpVariantsPass` and `DecomposeComplexOpsPass`) to
            # get it in a backend compliant form (aka torch backend contract).
            output_type="torch",
            func_name=loader_result.func_name,
        )

        # Important: This print is needed to pipe outputs in aot_compile's genrule
        print(torch_program)

    # Used by gen_{name}_reference_tensors genrule
    else:
        # Feed sample inputs to the model to get reference outputs
        reference_outputs = loader_result.model(*loader_result.inputs)

        reference_tensors = {}

        # Collect reference inputs
        for i, reference_input in enumerate(loader_result.inputs):
            reference_tensors[f"Input{i}"] = reference_input.numpy()

        # Collect reference outputs
        if isinstance(reference_outputs, tuple):
            # output is a tuple of torch.Tensor's
            for i, reference_output in enumerate(reference_outputs):
                reference_tensors[f"Output{i}"] = reference_output.numpy()
        elif isinstance(reference_outputs, torch.Tensor):
            # output is a single torch.Tensor
            reference_tensors["Output0"] = reference_outputs.numpy()

        # Save reference tensors as numpy archive (.npz)
        np.savez(args.reference_tensors_path, **reference_tensors)


if __name__ == "__main__":
    main()
