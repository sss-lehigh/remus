#!/usr/bin/env bash

cd /root
rm -rf build
mkdir build
cd build
cmake ..
make