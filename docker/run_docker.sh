#!/usr/bin/env bash

docker build -f docker/Dockerfile \
             -t mlir-tcp:dev \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/mlir-tcp" \
           -v "${HOME}/.cache/bazel":"/root/.cache/bazel" \
           mlir-tcp:dev
