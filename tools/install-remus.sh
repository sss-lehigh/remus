#!/bin/env bash

BUILD_TYPE=Release
INSTALL_PREFIX=/opt/remus

# Terminate on error
set -e
# Grab root directory
cd $(git rev-parse --show-toplevel)
# If build dir DNE, we dont error out
rm -rf build || true
mkdir build
cd build
# Delete old prefix if it exists
if [ -d ${INSTALL_PREFIX} ]; then
    echo "Deleting old install prefix"
    sudo rm -rf ${INSTALL_PREFIX}
fi
# Create new directory and give it permissions
sudo mkdir ${INSTALL_PREFIX}
sudo chmod 777 ${INSTALL_PREFIX}

# Pass the flags to cmake and compile to the install prefix
CC=clang-18 CXX=clang++-18 cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX ..
make -j $(nproc)
sudo make install

