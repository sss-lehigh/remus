from multiprocessing import Process
import subprocess
import os
from typing import List
import csv
import json
import argparse

def domain_name(nodetype):
    """Function to get domain name"""
    node_i = ['r320',           'luigi',          'r6525',               'xl170',            'c6525-100g',       'c6525-25g',        'd6515']
    node_h = ['apt.emulab.net', 'cse.lehigh.edu', 'clemson.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us']
    return node_h[node_i.index(nodetype)]

# set colors
red = "\033[31m"
black = "\033[0m"
green = "\033[32m"
blue = "\033[34m"

parser = argparse.ArgumentParser(description='Process the parameters for running an experiment or a test-suite')
# Experiment configuration
parser.add_argument('-u', '--ssh_user', type=str, required=True, help='Username for login)')
parser.add_argument('-e', '--experiment_name', type=str, required=True, help="Used as local save directory")
parser.add_argument('--nodefile', type=str, default="../../scripts/cloudlab.csv", help='Path to csv with the node names')
parser.add_argument('--dry_run', action='store_true', help='Print the commands instead of running them')
parser.add_argument('--exp_result', type=str, default='iht_result.csv', help='File to retrieve experiment result')

# Program run-types
parser.add_argument('--runtype', required=True, choices=['test', 'concurrent_test', 'bench'], help="Define the type of experiment to run. Test will run correctness tests single-threaded. Concurrent test will run a correctness test with multiple threads. And bench will run a benchmark")
parser.add_argument('--log_level', default='info', choices=['info', 'debug', 'trace'], help='The level of print-out in the program')

# Experiment parameters
parser.add_argument('--from_param_config', type=str, default=None, help="If to override the parameters with a config file.")
qps_sample_rate = 10 # not sure what these do, so leaving them out of the cmd-line options
max_qps_second = -1
parser.add_argument('--runtime', type=int, default=10, help="How long to run the experiment before cutting off")
parser.add_argument('--unlimited_stream', action='store_true', help="If to run the stream for an infinite amount or just until the operations run out")
parser.add_argument('--op_distribution', type=str, default="80-10-10", help="The distribution of operations as contains-insert-remove. Must add up to 100")
parser.add_argument('--op_count', type=int, default=10000, help="The number of operations to run if unlimited stream is passed as False.")
parser.add_argument('-lb', '--lb', type=str, default='0', help="Pass in the lower bound of the key range. Can use e-notation as well.")
parser.add_argument('-ub', '--ub', type=str, default='1e5', help="Pass in the ubber bound of the key range. Can use e-notation as well.")
parser.add_argument('--region_size', type=int, default=25, help="2 ^ x bytes to allocate on each node")
# Experiment resources
parser.add_argument('--thread_count', type=int, default=1, help="The number of threads to start per client. Only applicable in send_exp")
parser.add_argument('--node_count', type=int, default=1, help="The number of nodes to use in the experiment. Will use node0-nodeN")
parser.add_argument('--qp_max', type=int, default=30, help="The number of queue pairs to use in the experiment MAX")
ARGS = parser.parse_args()

# Get parent folder name
dir = os.path.abspath("../..")
bin_dir = dir.split("/").pop()

def quote(string):
    return f"'{string}'"

def is_valid(string):
    """Determines if a string is a valid experiment name"""
    for letter in string:
        if not letter.isalpha() and letter not in [str(i) for i in range(10)] and letter != "_":
            return False
    return True

def process_exp_flags(node_id):
    params = f" --node_id {node_id}"
    """Returns a string to append to the payload"""
    if ARGS.from_param_config is not None:
        with open(ARGS.from_param_config, "r") as f:
            # Load the json into the proto
            json_data = f.read()
            mapper = json.loads(json_data)
            one_to_ones = ["runtime", "op_count", "contains", "insert", "remove", "key_lb", "key_ub", "region_size", "thread_count", "node_count", "qp_max"]
            for param in one_to_ones:
                params += f" --{param} " + str(mapper[param]).lower()
            if mapper['unlimited_stream'] == True:
                params += f" --unlimited_stream "
    else:
        one_to_ones = ["runtime", "op_count", "region_size", "thread_count", "node_count", "qp_max"]
        for param in one_to_ones:
            params += f" --{param} " + str(eval(f"ARGS.{param}")).lower()
        if ARGS.unlimited_stream == True:
            params += f" --unlimited_stream "
        contains, insert, remove = ARGS.op_distribution.split("-")
        if int(contains) + int(insert) + int(remove) != 100:
            print("Must specify values that add to 100 in op_distribution")
            exit(1)
        params += " --contains " + str(contains)
        params += " --insert " + str(insert)
        params += " --remove " + str(remove)
        params += " --key_lb " + str(eval(ARGS.lb))
        params += " --key_ub " + str(eval(ARGS.ub))
    return params

# Create a function that will create a file and run the given command using that file as stout
def __run__(cmd, outfile, file_perm):
    with open(f"{outfile}.txt", file_perm) as f:
        if ARGS.dry_run:
            print(cmd)
        else:
            try:
                subprocess.run(cmd, shell=True, check=True, stderr=f, stdout=f)
                print(outfile, "Successful Startup")
                return
            except subprocess.CalledProcessError as e:
                print(outfile, "Invalid Startup because", e)

def execute(commands, file_perm):
    """For each command in commands, start a process"""
    processes: List[Process] = []
    for cmd, file in commands:
        # Start a thread
        file_out = os.path.join("results", ARGS.experiment_name, file)
        processes.append(Process(target=__run__, args=(cmd, file_out, file_perm)))
        processes[-1].start()

    # Wait for all threads to finish
    for process in processes:
        process.join()


def main():
    # Simple input validation
    if not is_valid(ARGS.experiment_name):
        print("Invalid Experiment Name")
        exit(1)
    print("Starting Experiment")
    # Create results directory
    os.makedirs(os.path.join("results", ARGS.experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", ARGS.experiment_name + "-stats"), exist_ok=True)
    
    commands = []
    commands_copy = []
    with open(ARGS.nodefile, "r") as f:
        for node in csv.reader(f):
            # For every node in nodefile, get the node info
            nodename, nodealias, nodetype = node
            node_id = int(nodename.replace("node", ""))
            # Construct ssh command and payload
            ssh_login = f"ssh {ARGS.ssh_user}@{nodealias}.{domain_name(nodetype)}"
            bazel_path = f"/users/{ARGS.ssh_user}/go/bin/bazelisk"
            payload = f"cd {bin_dir} && cmake . && make && LD_LIBRARY_PATH=.:./protos ./"
            if ARGS.runtype == "test":
                payload += "iht_rome_test --send_test"
            elif ARGS.runtype == "concurrent_test":
                payload += "iht_rome_test --send_bulk"
            elif ARGS.runtype == "bench":
                payload += "iht_rome"
                # Adding experiment flags
                payload += process_exp_flags(node_id)
            else:
                print("Found unknown runtype")
                exit(1)
            # Tuple: (Creating Command | Output File Name)
            commands.append((' '.join([ssh_login, quote(payload)]), nodename))
            if ARGS.runtype == "bench":
                filepath = os.path.join(f"/users/{ARGS.ssh_user}", bin_dir, ARGS.exp_result)
                local_dir = os.path.join("./results", ARGS.experiment_name + "-stats", nodename + "-" + ARGS.exp_result)
                copy = f"scp {ssh_login[4:]}:{filepath} {local_dir}"
                commands_copy.append((copy, nodename))
                continue # do for all nodes
            break # break -- just running the first node when testing
    # Execute the commands and let us know we've finished
    execute(commands, "w+")
    execute(commands_copy, "a")

    print("Finished Experiment")


if __name__ == "__main__":
    # Run abseil app
    main()
