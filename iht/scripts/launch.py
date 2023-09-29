from absl import app, flags
from multiprocessing import Process
import subprocess
import os
from typing import List
import csv
import json
import sys
sys.path.insert(1, '..')
import protos.experiment_pb2 as protos

def domain_name(nodetype):
    """Function to get domain name"""
    node_i = ['r320',           'luigi',          'r6525',               'xl170',            'c6525-100g',       'c6525-25g',        'd6515']
    node_h = ['apt.emulab.net', 'cse.lehigh.edu', 'clemson.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us']
    return node_h[node_i.index(nodetype)]



# Define FLAGS to represet the flags
FLAGS = flags.FLAGS

# Experiment configuration
flags.DEFINE_string('ssh_keyfile', '~/.ssh/id_rsa', 'Path to ssh file for authentication')
flags.DEFINE_string('ssh_user', 'esl225', 'Username for login')
flags.DEFINE_string('nodefile', '../../rome/scripts/nodefiles/r320.csv', 'Path to csv with the node names')
flags.DEFINE_string('experiment_name', None, 'Used as local save directory', required=True)
flags.DEFINE_string('bin_dir', 'RDMA/rdma_iht', 'Directory where we run bazel build from')
flags.DEFINE_bool('dry_run', required=True, default=None, help='Print the commands instead of running them')
flags.DEFINE_string('exp_result', 'iht_result.pbtxt', 'File to retrieve experiment result')

# Program run-types
flags.DEFINE_bool('send_bulk', required=False, default=False, help="Whether to run bulk operations (more for benchmarking)")
flags.DEFINE_bool('send_test', required=False, default=False, help="Whether to run basic method testing")
flags.DEFINE_bool('send_exp', required=False, default=False, help="Whether to run an experiment")
flags.DEFINE_enum('log_level', 'info', ['info', 'debug', 'trace'], 'The level of print-out in the program')

# Experiment parameters
flags.DEFINE_string('from_param_config', required=False, default=None, help="If to override the parameters with a config file.")
flags.DEFINE_integer('think_time', required=False, default=0, help="How long to wait in between operations (in nano seconds)")
flags.DEFINE_integer('qps_sample_rate', required=False, default=10, help="The QP sampling rate")
flags.DEFINE_integer('max_qps_second', required=False, default=-1, help="The max qps per second. Leaving -1 will be infinite")
flags.DEFINE_integer('runtime', required=False, default=10, help="How long to run the experiment before cutting off")
flags.DEFINE_bool('unlimited_stream', required=False, default=False, help="If to run the stream for an infinite amount or just until the operations run out")
flags.DEFINE_string('op_distribution', required=False, default="80-10-10", help="The distribution of operations as contains-insert-remove. Must add up to 100")
flags.DEFINE_integer('op_count', required=False, default=10000, help="The number of operations to run if unlimited stream is passed as False.")
flags.DEFINE_list('key_range', required=False, default=['0', '1e6'], help="Pass in two values to be the [lb,ub] of the key range. Can use e-notation as well.")
flags.DEFINE_integer('region_size', required=False, default=22, help="2 ^ x bytes to allocate on each node")
flags.DEFINE_bool('default', required=False, default=False, help="If to run the experiment with the default proto command")

# Cluster parameters
flags.DEFINE_integer('thread_count', required=False, default=1, help="The number of threads to start per client. Only applicable in send_exp")
flags.DEFINE_integer('node_count', required=False, default=1, help="The number of nodes to use in the experiment. Will use node0-nodeN")

SINGLE_QUOTE = "'\"'\"'"
def make_one_line(proto):
    return SINGLE_QUOTE + ' '.join(line for line in str(proto).split('\n')) + SINGLE_QUOTE 

def quote(string):
    return f"'{string}'"

def is_valid(string):
    """Determines if a string is a valid experiment name"""
    for letter in string:
        if not letter.isalpha() and letter not in [str(i) for i in range(10)] and letter != "_":
            return False
    return True

def execute(commands, file_perm):
    """For each command in commands, start a process"""
    # Create a function that will create a file and run the given command using that file as stout
    def __run__(cmd, outfile):
        with open(f"{outfile}.txt", file_perm) as f:
            if FLAGS.dry_run:
                print(cmd)
            else:
                try:
                    subprocess.run(cmd, shell=True, check=True, stderr=f, stdout=f)
                    print(outfile, "Successful Startup")
                    return
                except subprocess.CalledProcessError as e:
                    print(outfile, "Invalid Startup because", e)

    processes: List[Process] = []
    for cmd, file in commands:
        # Start a thread
        processes.append(Process(target=__run__, args=(cmd, os.path.join("results", FLAGS.experiment_name, file))))
        processes[-1].start()

    # Wait for all threads to finish
    for process in processes:
        process.join()

def process_exp_flags():
    params = protos.ExperimentParams()
    """Returns a string to append to the payload"""
    if FLAGS.from_param_config is not None:
        with open(FLAGS.from_param_config, "r") as f:
            # Load the json into the proto
            json_data = f.read()
            mapper = json.loads(json_data)
            one_to_ones = ["think_time", "qps_sample_rate", "max_qps_second", "runtime", "unlimited_stream", "op_count", "contains", "insert", "remove", "key_lb", "key_ub", "region_size", "thread_count", "node_count"]
            for param in one_to_ones:
                exec(f"params.{param} = mapper['{param}']")
    elif not FLAGS.default:
        one_to_ones = ["think_time", "qps_sample_rate", "max_qps_second", "runtime", "unlimited_stream", "op_count", "region_size", "thread_count", "node_count"]
        for param in one_to_ones:
            exec(f"params.{param} = FLAGS.{param}")
        contains, insert, remove = FLAGS.op_distribution.split("-")
        if int(contains) + int(insert) + int(remove) != 100:
            print("Must specify values that add to 100 in op_distribution")
            exit(1)
        params.contains = int(contains)
        params.insert = int(insert)
        params.remove = int(remove)
        params.key_lb = int(eval(FLAGS.key_range[0]))
        params.key_ub = int(eval(FLAGS.key_range[1]))
    return params


def main(args):
    # Simple input validation
    if not is_valid(FLAGS.experiment_name):
        print("Invalid Experiment Name")
        exit(1)
    print("Starting Experiment")
    # Create results directory
    os.makedirs(os.path.join("results", FLAGS.experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", FLAGS.experiment_name + "-stats"), exist_ok=True)
    
    commands = []
    commands_copy = []
    with open(FLAGS.nodefile, "r") as f:
        for node in csv.reader(f):
            # For every node in nodefile, get the node info
            nodename, nodealias, nodetype = node
            # Construct ssh command and payload
            ssh_login = f"ssh -i {FLAGS.ssh_keyfile} {FLAGS.ssh_user}@{nodealias}.{domain_name(nodetype)}"
            bazel_path = f"/users/{FLAGS.ssh_user}/go/bin/bazelisk"
            payload = f"cd {FLAGS.bin_dir}; {bazel_path} run main --log_level={FLAGS.log_level} --"
            # Adding run-type
            if FLAGS.send_test:
                payload += " --send_test"
            elif FLAGS.send_bulk:
                payload += " --send_bulk"
            elif FLAGS.send_exp:
                payload += " --send_exp"
            else:
                print("Must specify whether testing methods '--send_test', doing bulk operations '--send_bulk', or sending experiment '--send_exp'")
                exit(1)
            # Adding experiment flags
            payload += " --experiment_params=" + make_one_line(process_exp_flags())
            # Tuple: (Creating Command | Output File Name)
            commands.append((' '.join([ssh_login, quote(payload)]), nodename))
            if FLAGS.send_exp:
                bridge = 'bazel-out/k8-fastbuild/bin/main.runfiles/rdma_iht'
                filepath = os.path.join(f"/users/{FLAGS.ssh_user}", FLAGS.bin_dir, bridge, FLAGS.exp_result)
                local_dir = os.path.join("./results", FLAGS.experiment_name + "-stats", nodename + "-" + FLAGS.exp_result)
                copy = f"scp {ssh_login[4:]}:{filepath} {local_dir}"
                commands_copy.append((copy, nodename))
    # Execute the commands and let us know we've finished
    execute(commands, "w+")
    execute(commands_copy, "a")

    print("Finished Experiment")


if __name__ == "__main__":
    # Run abseil app
    app.run(main)
