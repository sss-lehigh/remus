#!/bin/env bash

# Modify these variables
DEBUG=0 # 1 for debug with gdb
SET_BASHRC=1 # set to 0 to not change bashrc remember .bashrc.old still exists

user="person"
machines=("amd157" "amd204" "amd146" "amd145" "amd166" "amd156")
domain="utah.cloudlab.us"
node_list="node0,node1,node2,node3,node4,node5"

exe_dir="./build/rdma"
#Modify this variable if changing the executable and make sure to modify cmd at line 59
exe="multinode_test"

#######################################################################################################################

cd build
make -j "$exe"
cd ..

# Loops through all machines and connects to setup
for m in "${machines[@]}"; do
  ssh "${user}@$m.$domain" hostname 
done

if [[ $SET_BASHRC -eq 1 ]]; then
  # sets ulimit for open files to hard limit
  # must be set in bashrc to always reset on ssh
  for m in "${machines[@]}"; do
    hardlimit=$(ssh "${user}@$m.$domain" ulimit -Hn)
    ssh "${user}@$m.$domain" mv .bashrc .bashrc.old
    echo "Setting to $hardlimit"
    echo "ulimit -n $hardlimit" > temp
    scp temp "${user}@$m.$domain":~/.bashrc
    ssh "${user}@$m.$domain" ulimit -n
  done
  rm temp
else
  echo "Make sure ulimits are high enough!!!!" 
fi

for m in "${machines[@]}"; do
  ssh "${user}@$m.$domain" sudo pkill -SIGTERM "${exe}"
  scp "${exe_dir}/${exe}" "${user}@$m.$domain":~
done

echo "#!/bin/env bash" > run_experiment.sh

for m in "${machines[@]}"; do
  echo "ssh ${user}@$m.$domain pkill -9 ${exe}" >> run_experiment.sh
done

echo -n "tmux new-session \\; " >> run_experiment.sh
idx=0
for m in "${machines[@]}"; do
  
  # Modify this if changing code
  cmd="./${exe} -p 10000 -t 32 -n $node_list -i $idx"

  echo " \\" >> run_experiment.sh
  if [[ $idx -ne 0 ]]; then
    echo " new-window \; \\" >> run_experiment.sh
  fi
  if [[ "$DEBUG" -eq 0 ]]; then
    echo -n " send-keys 'ssh ${user}@$m.$domain $cmd' C-m \\; " >> run_experiment.sh
  else
    echo -n " send-keys 'ssh ${user}@$m.$domain' C-m \\; " >> run_experiment.sh
    echo -n " send-keys 'gdb --args $cmd' C-m \\; " >> run_experiment.sh
  fi
  idx=$((idx + 1))
done

chmod +x run_experiment.sh

./run_experiment.sh
