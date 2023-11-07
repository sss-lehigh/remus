# CMDs

Ethan's guide to some long commands for copying-pasting. TODO: Replace for an easier interface

Generate a Makefile, compile_commands.json (for clangd) and debug mode (can change to release if running benchmarks). And then compile!

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug ..
make

## Syncing files

Send folders recorded in scripts/include.txt (and not in scripts/exclude.txt) to cloudlab

python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225 --local_root=/Users/ethan/Research/librome_iht --sync

## Make clean

python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225 --local_root=/Users/ethan/Research/librome_iht --sync --cmd="cd iht_rdma_minimal && make clean"

## Installing dependencies

Send folders recorded in scripts/include.txt to cloudlab and install dependencies

python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225 --local_root=/Users/ethan/Research/librome_iht --sync --cmd="cd iht_rdma_minimal && sh cloudlab_depend.sh"

## Running

Start running the IHT

python launch.py --experiment_name=exp --nodry_run --from_param_config=exp_conf.json --send_exp --bin_dir=iht_rdma_minimal

## Manual Normal

LD_LIBRARY_PATH=./build:./build/protos ./iht_rome --send_exp --experiment_params "qps_sample_rate: 10 max_qps_second: -1 runtime: 10 unlimited_stream: true op_count: 1000 contains: 80 insert: 10 remove: 10 key_lb: 0 key_ub: 10000 region_size: 25 thread_count: 2 node_count: 1 qp_max: 1 node_id: 0 "

## Manual for GDB

LD_LIBRARY_PATH=./build:./build/protos gdb ./iht_rome

run --experiment_params "qps_sample_rate: 10 max_qps_second: -1 runtime: 10 unlimited_stream: true op_count: 1000 contains: 80 insert: 10 remove: 10 key_lb: 0 key_ub: 10000 region_size: 25 thread_count: 4 node_count: 1 qp_max: 1 node_id: 0 "

## Testing

LD_LIBRARY_PATH=.:./protos ./iht_rome_test --send_test

LD_LIBRARY_PATH=.:./protos gdb ./iht_rome_test
run --send_test

LD_LIBRARY_PATH=.:./protos ./iht_rome_test --send_bulk

LD_LIBRARY_PATH=.:./protos gdb ./iht_rome_test
run --send_bulk
