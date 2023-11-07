from absl import app, flags
from multiprocessing import Process
import subprocess
import os
from typing import List
import csv
import sys

def domain_name(nodetype):
    """Function to get domain name"""
    node_i = ['r320',           'luigi',          'r6525',               'xl170',            'c6525-100g',       'c6525-25g',        'd6515']
    node_h = ['apt.emulab.net', 'cse.lehigh.edu', 'clemson.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us', 'utah.cloudlab.us']
    return node_h[node_i.index(nodetype)]

flags.DEFINE_string('ssh_keyfile', '~/.ssh/id_rsa', 'Path to ssh file for authentication')
flags.DEFINE_string('ssh_user', 'esl225', 'Username for login')
flags.DEFINE_string('nodefile', '../../scripts/nodefiles/r320.csv', 'Path to csv with the node names')
flags.DEFINE_bool('dry_run', required=False, default=False, help='Print the commands instead of running them')

# Define FLAGS to represet the flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)   # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx

def quote(string):
    return f"'{string}'"

# Create a function that will create a file and run the given command using that file as stout
def __run__(cmd):
    if FLAGS.dry_run:
        print(cmd)
    else:
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=None, stdout=None)
            print("Successful Execution")
            return
        except subprocess.CalledProcessError:
            print("Invalid Execution")

def execute(commands):
    """For each command in commands, start a process"""
    processes: List[Process] = []
    for cmd, file in commands:
        # Start a thread
        processes.append(Process(target=__run__, args=(cmd,)))
        processes[-1].start()

    # Wait for all threads to finish
    for process in processes:
        process.join()

def main(args):
    print("Shutting Down Servers...")
    commands = []
    with open(FLAGS.nodefile, "r") as f:
        for node in csv.reader(f):
            # For every node in nodefile, get the node info
            nodename, nodealias, nodetype = node
            # Construct ssh command and payload
            ssh_login = f"ssh -i {FLAGS.ssh_keyfile} {FLAGS.ssh_user}@{nodealias}.{domain_name(nodetype)}"
            # !!! The purpose here is to shutdown running instances of the program if its gets stuck somewhere !!!
            payload = f"/usr/bin/killall -15 iht_rome; /usr/bin/killall -15 iht_rome_test"
            # Tuple: (Creating Command | Output File Name)
            commands.append((' '.join([ssh_login, quote(payload)]), nodename))
    # Execute the commands and let us know we've finished
    execute(commands)
    print("Completed Task")


if __name__ == "__main__":
    # Run abseil app
    app.run(main)
