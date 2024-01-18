# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

ARG BASE_IMG=ubuntu:22.04
FROM ${BASE_IMG} as dev-base

# Specify user IDs
ARG GROUP
ARG GID
ARG USER
ARG UID

# Run below commands as root
USER root

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
ARG BAZEL_VERSION=7.0.1
RUN wget -q https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-${ARCH} -O /usr/bin/bazel \
    && chmod a+x /usr/bin/bazel

# Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set workdir before launching container
WORKDIR /opt/src/mlir-tcp

# Add user permissions
RUN groupadd -o -g ${GID} ${GROUP} && \
    useradd -u ${UID} -g ${GROUP} -ms /bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    chown -R ${USER}:${GROUP} /opt/src/mlir-tcp

# Switch to user
USER ${USER}
