
for t in 1 2 3 4 5
do
    for n in 3
    do 
        python launch.py --experiment_name="${t}t${n}n" --nodry_run --send_exp --region_size=24 --key_range=0,100000 --op_count=1 --op_distribution=80-10-10 --unlimited_stream=True --runtime=60 --think_time=100 --thread_count=$t --node_count=$n
    done
done
