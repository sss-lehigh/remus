#!/bin/bash
workspace=/home/manager/Research/RDMA/rome
nodefile=/home/manager/Research/RDMA/rome/scripts/nodefiles/r320.csv
#** FUNCTION DEFINITIONS **#
sync_nodes() {
  tmp=$(pwd)
  cd ../rome/scripts
  python rexec.py -n ${nodefile} --remote_user=esl225 --remote_root=/users/esl225/RDMA --local_root=/home/manager/Research/RDMA --sync
  cd ${tmp}
  echo "Sync Complete\n"
}

test_loopback() {
  tmp=$(pwd)
  cd ../rome/scripts
  python rexec.py -n ${nodefile} --remote_user=esl225 --remote_root=/users/esl225 --local_root=/home/manager/Research/RDMA --cmd="cd RDMA/rome/rome/rdma/memory_pool && ~/go/bin/bazelisk test --log_level=trace memory_pool_test"
  cd ${tmp}
  echo "Build Complete\n"
}

#** START OF SCRIPT **#
echo "Testing..."
sync_nodes
test_loopback
echo "Done!"