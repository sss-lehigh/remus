# Interlocked Hash Table in RDMA

A Rome based IHT implementation.

## Necessary build files to write

- .bazelrc
- WORKSPACE
- BUILD
- spdlog.BUILD
- fmt.BUILD

## Deploying

### Normal Deploy

1. Check availability
2. Create experiment
3. Select Ubuntu 20.04 (OS), r320 (Node Type), name (Experiment Name), hours (Duration)
4. Edit ./rome/scripts/nodefiles/r320.csv with data from listview
5. conda activate rdma. And then wait while configuring.
6. cd into ./rome/scripts
7. [ONCE FINISHED] Run sync command to check availability (if alias available)
```{bash}
python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225/RDMA --local_root=/home/manager/Research/RDMA --sync
```
8. Check logs at /tmp/rome/logs for success
9. Run start up script
```{bash}
python rexec.py --nodefile=nodefiles/r320.csv --remote_user=esl225 --remote_root=/users/esl225/RDMA --local_root=/home/manager/Research/RDMA --sync --cmd="cd RDMA/rome/scripts/setup && python3 run.py --resources all"
```
10. Wait while configuring. Can check /tmp/rome/logs for updates.
11. [ONCE FINISHED] Login to nodes or continue to run commands using launch.py
```{bash}
python launch.py --experiment_name={exp} --nodry_run --from_param_config=exp_conf.json --send_{bulk|test|exp}
```

### Running in GDB

```
cd RDMA/rdma_iht
```

```
bazel build main --compilation_mode=dbg --log_level=info && gdb bazel-bin/main
```

```
run --send_exp '--experiment_params=think_time: 100 qps_sample_rate: 10 max_qps_second: -1 runtime: 5 unlimited_stream: true op_count: 10000 contains: 80 insert: 10 remove: 10 key_lb: 0 key_ub: 100000 region_size: 24 thread_count: 10 node_count: 1' 
```

## Errors

### Issue connecting to peers

```
[2023-02-12 21:1142 thread:54545] [error] [external/rome/rome/rdma/memory_pool/memory_pool_impl.h:148] INTERNAL: ibv_modify_qp(): Invalid argument
```

Solved by making sure peers didn't include self 
<br><br>

### Barrier Dependency Issue

```
fatal error: 'barrier' file not found
```
By changing tool_path for gcc on line 74 from /usr/bin/clang to /usr/bin/clang-12, I was able to compile correctly 
<br><br>

### Other issues

Implementation Issue:
```
libc++abi: terminating with uncaught exception of type std::out_of_range: unordered_map::at: key not found
```
This happens when we try to use the RDMA API to deal with remote pointers 'pointing' to ourselves...
If we look at ibstat. We see two port numbers. The issue was, for some unknown reason, the second port made loopback fail.

<br><br>

Implementation Issue:
```
OOM!
```
This happens when we leak memory. For example with in-correct alignment [64] or if we don't free the space we allocate from reading data.
<br><br>


Spontaneous Issue:
```
[2023-03-14 16:4508 thread:60254] [critical] [external/rome/rome/rdma/memory_pool/memory_pool_impl.h:256] ibv_poll_cq(): remote access error @ (id=0, address=0x7fbe86c10800)
```
This happens if the server shuts off before the client finishes!
<br><br>

## Configuring Your Enviornment For Development

- Installations
    - absl (/usr/local/include/absl) <i>[Abseil Source](https://github.com/abseil/abseil-cpp)</i>
    - rdma_core (?) <i>[RDMA Core Source](https://github.com/linux-rdma/rdma-core)</i>
    - Google Test Framework <i>[gmock Source](https://github.com/google/googletest)</i>
        - gmock (/usr/local/include/gmock) 
        - gtest (/usr/local/include/gtest)
    - protos () <i>[Protocol Buffer Source](https://github.com/protocolbuffers/protobuf)</i>
        - Compiling protoc -I=. --cpp_out=. --python_out=. file.proto
        - Experiment.proto --> protoc -I=../../rome --cpp_out=. --python_out=. --proto_path=. experiment.proto
        - Also adding path ```~/INSTALL/protobuf/src```
        - https://chromium.googlesource.com/external/github.com/protocolbuffers/protobuf/+/refs/tags/v3.7.1/src


> Note for VSCode. Edit the include path setting to allow for better Intellisense

