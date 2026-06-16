exefile="build/benchmark/perftestv2"
experiment_args="--seg-size 29 --segs-per-mn 1 --first-cn-id 0 --last-cn-id 1 --first-mn-id 0 --last-mn-id 1 --qp-lanes 1 --qp-sched-pol MOD --mn-port 33330 --cn-threads 1 --cn-thread-msgs 1 --cn-thread-bufsz 14 --alloc-pol GLOBAL-RR --exp-name perftest --ops 100000 --exp-op Read --elements 65536 --zero-copy 1 --overlap 0 --think-time 0 --read-p 100"
