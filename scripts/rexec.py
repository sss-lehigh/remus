#!/usr/bin/env python

from absl import app
from absl import flags
import configparser
import subprocess
import os
import threading
import time
from datetime import datetime

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "configfile",
    "rexec.ini",
    "Configuration file to use in conjunction with the command line parameters.",
    short_name="c",
)

# Global variables for configuration and command line flags
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
common = {}
FLAGS = flags.FLAGS

# Configuration information.
flags.DEFINE_string(
    "nodefile",
    None,
    "File containing a comma separated list of nodes to be included in the remote execution.",
    required=True,
    short_name="n",
)
flags.DEFINE_integer(
    "num_nodes",
    None,
    "Number of nodes to run on. Takes the first N nodes in the nodefile. Ignored when syncing with remote nodes.",
    short_name="N",
)

# Directory location information
flags.DEFINE_string(
    "remote_user",
    None,
    "The username to use at the remote endpoint when executing commands",
    required=True,
)
flags.DEFINE_string(
    "remote_root",
    None,
    "The root directory of the project at the remote endpoint",
    required=True,
)
flags.DEFINE_string(
    "local_root",
    None,
    "The root directory of the project on the local machine",
    required=True,
)

flags.DEFINE_list(
    "skip",
    [],
    "Hosts to skip while running an experiment. Acts as if the command was executed on the host for debugging.",
)
flags.DEFINE_list(
    "offline",
    [],
    "Hosts that should be considered offline (i.e., are not included in the host set).",
)

flags.DEFINE_bool(
    "sync",
    False,
    "Sync the local copy of the repo with the remote nodes. If a command or experiment is also provided, the sync is performed first.",
    short_name="s",
)
flags.DEFINE_string("cmd", None, "Command to execute on the remote machines.")

flags.DEFINE_string(
    "outfile", None, "File to write output of experiments to on the remote machine"
)
flags.DEFINE_string(
    "get_data", None, "Get data from the provided remote directory.", short_name="D"
)
flags.DEFINE_bool("sudo", False, "Run experiment as sudo.", short_name="S")
flags.DEFINE_bool("debug", False, "Print out debug information.")

# Running experiments.
flags.DEFINE_string(
    "exp",
    None,
    "Name of experiment to run (must have a corresponding section in the configfile).",
)
flags.DEFINE_string(
    "exp_bin",
    None,
    "Binary file to run for a given experiment. Can be used to provide the same flow for multiple binary files to test different compile-time configurations.",
)
flags.DEFINE_list(
    "exp_args",
    [],
    "Arguments to be passed during execution of the experiment named in the --exp flag.",
)


def get_config_str():
    config_str = "Configuration:\n"
    for s in config.sections():
        config_str += "[" + s + "]\n"
        for k in config[s]:
            config_str += str(k) + "=" + config[s][k] + "\n"
        config_str += "\n"
    return config_str


def get_hosts(nodefile):
    nodes = get_nodes(nodefile)
    hosts = {}
    for n in nodes:
        if n[0] not in FLAGS.offline:
            hosts[n[0]] = build_hostname(n[2], n[1])
    return hosts


def get_nodes(nodefile):
    nodes = []
    with open(nodefile, "r") as in_file:
        line = in_file.readline()
        while line != "":
            nodes.append(tuple(line.strip().split(",")))
            line = in_file.readline()
    return nodes


__utah__ = [
    "xl170",
    "d6515",
    "c6525-100g",
    "c6525-25g",
    "m510",
]

__clem__ = [
    "r6525",
]

__emulab__ = [
    "r320",
]

__luigi__ = [
    "luigi",
]


def build_hostname(nodetype, id):
    if nodetype in __utah__:
        return id + ".utah.cloudlab.us"
    elif nodetype in __emulab__:
        return id + ".apt.emulab.net"
    elif nodetype in __clem__:
        return id + ".clemson.cloudlab.us"
    elif nodetype in __luigi__:
        return id + ".cse.lehigh.edu"
    else:
        print("Undefined host type in nodefile.")
        exit(1)


def DEBUG(msg):
    if FLAGS.debug:
        print(msg)


def prepare_log(name, host, testbed, cmd):
    log_dir = os.path.join(common["log_dir"], cmd, testbed)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, name + "." + host + ".log")


# Executes a command locally but logs it according to the host and provide logging string
def local_exec(cmd, log_path):
    with open(log_path, "w") as outfile:
        outfile.write(get_config_str())
        outfile.write(str(cmd) + "\n\n")
        outfile.flush()
        try:
            DEBUG(cmd)
            subprocess.run(cmd, shell=True, check=True, stdout=outfile, stderr=outfile)
        except subprocess.CalledProcessError:
            print("Check logs: {}".format(log_path))


def remote_exec(host, cmd, log_path):
    DEBUG(cmd)
    with open(log_path, "w") as outfile:
        outfile.write(get_config_str())
        outfile.write(str(cmd) + "\n\n")
        outfile.flush()
        cmd = (
            common["ssh"]
            + " "
            + FLAGS.remote_user
            + "@"
            + host
            + " 'bash -l -c \""
            + cmd
            + "\"'"
        )
        DEBUG(cmd)
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=outfile, stderr=outfile)
        except subprocess.CalledProcessError:
            print("Check logs: {}".format(log_path))


def get_data(args):
    # print_config("get_data")
    pass


def main(argv):
    # Read configuration file
    global common
    global config
    config.read(FLAGS.configfile)
    common = config["common"]
    testbed = os.path.split(FLAGS.nodefile)[-1].split(".")[0]
    hosts = get_hosts(FLAGS.nodefile)
    skip = FLAGS.skip
    if FLAGS.num_nodes is None:
        FLAGS.num_nodes = len(hosts)

    DEBUG("configfile={}".format(FLAGS.configfile))
    DEBUG("\n{}".format(get_config_str()))
    DEBUG("nodefile={}".format(FLAGS.nodefile))
    DEBUG("testbed={}".format(testbed))
    DEBUG("hosts={}".format(hosts))

    # Perform a sync before any other operations
    threads = {}
    if FLAGS.sync:
        DEBUG("Syncing nodes")
        # Prepare the sync command
        src = os.path.normpath(FLAGS.local_root) + "/"
        for h in hosts:
            dest = FLAGS.remote_user + "@" + hosts[h] + ":" + FLAGS.remote_root
            exclude = os.path.join(src, config["sync"]["exclude_file"])
            include = os.path.join(src, config["sync"]["include_file"])
            cmd = (
                "rsync -r -e '"
                + common["ssh"]
                + "' --exclude-from="
                + exclude
                + " --files-from="
                + include
                + " --progress -uva "
                + src
                + " "
                + dest
            )

            t = threading.Thread(
                target=local_exec, args=[cmd, prepare_log(h, hosts[h], testbed, "sync")]
            )
            threads[h] = t
            t.start()
        for h in threads.keys():
            DEBUG("Joining host: {}".format(h))
            threads[h].join()

    threads = {}
    active_hosts = [h for h in hosts.keys()]
    active_hosts = active_hosts[0 : FLAGS.num_nodes]
    if FLAGS.cmd is not None:
        DEBUG("Running command: {}".format(FLAGS.cmd))
        for h in active_hosts:
            dest = FLAGS.remote_user + "@" + hosts[h]
            t = threading.Thread(
                target=remote_exec,
                args=[hosts[h], FLAGS.cmd, prepare_log(h, hosts[h], testbed, "cmd")],
            )
            threads[h] = t
            t.start()
        for h in threads.keys():
            DEBUG("Joining host: {}".format(h))
            threads[h].join()


if __name__ == "__main__":
    app.run(main)
