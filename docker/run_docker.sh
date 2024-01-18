#!/usr/bin/env bash

docker build -f docker/Dockerfile \
             -t mlir-tcp:dev \
             --build-arg GROUP=$(id -gn) \
             --build-arg GID=$(id -g) \
             --build-arg USER=$(id -un) \
             --build-arg UID=$(id -u) \
             .

docker run -it \
           -v "$(pwd)":"/opt/src/mlir-tcp" \
           -v "${HOME}/.cache/bazel":"${HOME}/.cache/bazel" \
           mlir-tcp:dev
