# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG} as dev-base

# Install basic packages
RUN apt-get update && \
    apt-get install -y \
    wget \
    lld \
    clang \
    clang-format \
    gdb

# Install bazel
ARG ARCH="x86_64"
ARG BAZEL_VERSION=5.4.0
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

# Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src/mlir-tcp
