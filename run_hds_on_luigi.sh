#!/bin/env bash

set -e

USER=depaul

cd build

make -j

scp ./hds/examples/2_rdma_linked_list $USER@luigi.cse.lehigh.edu:~

ssh $USER@luigi.cse.lehigh.edu ./2_rdma_linked_list 10.0.0.1

