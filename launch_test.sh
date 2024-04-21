#!/bin/env bash

DEBUG=0

cd build
make -j multinode_test
cd ..

user="adb321"
machines=("apt183" "apt173" "apt176")
domain="apt.emulab.net"
node_list="node0,node1,node2"

exe_dir="./build/rdma"
exe="multinode_test"

for m in ${machines[@]}; do
  ssh "${user}@$m.$domain" hostname
done

for m in ${machines[@]}; do
  ssh "${user}@$m.$domain" pkill -9 "${exe}"
  scp "${exe_dir}/${exe}" "${user}@$m.$domain":~
done

echo "#!/bin/env bash" > run_experiment.sh

for m in ${machines[@]}; do
  echo "ssh ${user}@$m.$domain pkill -9 ${exe}" >> run_experiment.sh
done

echo -n "tmux new-session \\; " >> run_experiment.sh
idx=0
for m in ${machines[@]}; do
  echo " \\" >> run_experiment.sh
  if [[ $idx -ne 0 ]]; then
    echo " new-window \; \\" >> run_experiment.sh
  fi
  echo -n " send-keys 'ssh ${user}@$m.$domain ./${exe} -p 8080 -t 2 -n $node_list -i $idx' C-m \\; " >> run_experiment.sh
  idx=$((idx + 1))
done

chmod +x run_experiment.sh

./run_experiment.sh
