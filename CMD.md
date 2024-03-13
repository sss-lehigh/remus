# CMDs

Ethan's guide to some long commands for copying-pasting. TODO: Replace for an easier interface

Generate a Makefile, compile_commands.json (for clangd) and debug mode (can change to release if running benchmarks). And then compile!

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug ..
make

## Syncing files

Send everything in iht_rdma_minimal (but not in scripts/exclude.txt) to cloudlab

python sync.py -u esl225

OR

sh sync.sh -u esl225

(use -h to get usage for sync)

## Installing dependencies

Sync first and then install dependencies

python sync.py -u esl225 -i
sh sync.sh -u esl225 -i

## Running

### Start running the IHT (benchmark)

python launch.py -u esl225 -e exp --runtype bench --from_param_config exp_conf.json

If running correctness tests, use --runtype test or --runtype concurrent_test

### Stopping the IHT if it stalls/deadlocks

python shutdown.py -u esl225

### Manual Normal

LD_LIBRARY_PATH=./build:./build/protos ./iht_remus --node_id 0 --runtime 10 --op_count 0 --contains 80 --insert 10 --remove 10 --key_lb 0 --key_ub 10000 --region_size 25 --thread_count 1 --node_count 1 --qp_max 1 --unlimited_stream

### Manual for GDB

LD_LIBRARY_PATH=./build:./build/protos gdb ./iht_remus

run --node_id 0 --runtime 10 --op_count 0 --contains 80 --insert 10 --remove 10 --key_lb 0 --key_ub 10000 --region_size 25 --thread_count 1 --node_count 1 --qp_max 1 --unlimited_stream

### Manual Testing

LD_LIBRARY_PATH=.:./protos ./iht_remus_test --send_test

LD_LIBRARY_PATH=.:./protos gdb ./iht_remus_test
run --send_test

LD_LIBRARY_PATH=.:./protos ./iht_remus_test --send_bulk

LD_LIBRARY_PATH=.:./protos gdb ./iht_remus_test
run --send_bulk
