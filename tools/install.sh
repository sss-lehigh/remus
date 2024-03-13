#!/bin/env bash

BUILD_TYPE=Release
INSTALL_PREFIX=/opt/remus

cd $(git rev-parse --show-toplevel)
mkdir build
cd build

sudo mkdir ${INSTALL_PREFIX}
sudo chmod 777 ${INSTALL_PREFIX}

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX ..
make -j install

