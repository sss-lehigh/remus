#!/bin/env bash
# This script builds the project in debug mode using clang-18.

# Go into root dir
cd $(git rev-parse --show-toplevel)
# Refresh build
rm -rf build/
mkdir build
cd build
# Pass flags to cmake
CC=clang-18 CXX=clang++-18 cmake -DLOG_LEVEL=DEBUG ..
# Compile
make -j$(nproc)