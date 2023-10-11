# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os

import lit.formats
from lit.llvm import llvm_config

# Populate Lit configuration with the minimal required metadata.
# Some metadata is populated in lit.site.cfg.py.in.
config.name = 'MLIR_TCP_TESTS_SUITE'
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

# Make LLVM and TCP tools available in RUN directives
tools = [
  'tcp-opt',
  'FileCheck',
  'count',
  'not',
]

tool_dirs = [
  config.llvm_tools_dir,
  config.tcp_tools_dir,
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
