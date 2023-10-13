# Interlocked Hash Table in RDMA

A Rome based IHT implementation.

## Deploying

### Downloading dependencies

Rome is built using CMake on Ubuntu 22.04 machines. You can download all the required dependencies for running IHT using cloudlab_depend.sh

### Quick Cloudlab Deploy

1. Environment: Ubuntu 22.04 (OS), r320 (Node Type)
2. If on cloudlab, edit ./rome/scripts/nodefiles/r320.csv with data from listview
3. conda activate rdma
4. cd into rome/scripts
5. Sync and download dependencies

```{bash}
python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225/librome_iht --local_root=/Users/ethan/Research/librome_iht --sync --cmd="cd librome_iht/iht_rdma_minimal && sh cloudlab_depend.sh"
```

7. Wait while configuring. Can check /tmp/rome/logs/cmd for updates.
8. [ONCE FINISHED] Login to nodes or continue to run commands using launch.py

```{bash}
python launch.py --experiment_name=exp --nodry_run --from_param_config=exp_conf.json --send_exp
```

### Build on Dockerfile

```{bash}
docker build --tag sss-dev --file Dockerfile .
docker run --privileged --rm -v {$MOUNT_DIR}:/home --name sss -it sss-dev
```

### Running in GDB

```{bash}
cd RDMA/rdma_iht
```

```{bash}
bazel build main --compilation_mode=dbg --log_level=info && gdb bazel-bin/main
```

```{bash}
run --send_exp '--experiment_params=think_time: 100 qps_sample_rate: 10 max_qps_second: -1 runtime: 5 unlimited_stream: true op_count: 10000 contains: 80 insert: 10 remove: 10 key_lb: 0 key_ub: 100000 region_size: 24 thread_count: 10 node_count: 1 node_id: 0 max_qp: 30' 
```

## Common Errors

```{bash}
OOM!!
```

This signifies out of memory.

## Installing librome in Docker container

TODO!