from unittest import skip
import mlnx_ofed
import zookeeper
import apt
import golang
import bazel
import conda
import pprof
import argparse
from util import *

SUPPORTED = ['apt', 'golang', 'bazel', 'conda'] # 'zookeeper', 'pprof', 'mlnx_ofed']


def main(args):
    init(args.config)

    # Ask the user for the root directory of the project if the configured one does not exist.
    if not os.path.isdir(config["workspace"]["root"]):
        root = input("Enter the project root directory: ")
        config["workspace"]["root"] = root
        with open(args.config, 'w') as configfile:
            config.write(configfile)

    if not os.path.exists(config["workspace"]["env_file"]):
        with open(config["workspace"]["env_file"], 'w') as f:
            f.write("#! /bin/bash\n")

    resources = []
    if len(args.resources) == 1 and args.resources[0] == 'all':
        args.resources = SUPPORTED
    for r in [x for x in args.resources if x not in args.skip]:
        if r == 'apt':
            resources.append(apt.APTPackages(config))
        elif r == 'golang':
            resources.append(golang.Golang(config))
        elif r == 'bazel':
            resources.append(bazel.Bazel(config))
        elif r == 'conda':
            resources.append(conda.Conda(config))
        elif r == 'zookeeper':
            resources.append(zookeeper.Zookeeeper(config))
        elif r == 'pprof':
            resources.append(pprof.Pprof(config))
        elif r == 'mlnx_ofed':
            resources.append(mlnx_ofed.MlnxOfed(config))
        else:
            print(f"Unknown resource: {r}")

    for r in resources:
        try:
            r.setup()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform environment setup")
    parser.add_argument(
        '--config', metavar='C', type=str, nargs=1, default='config.ini',
        help="Path to configuration (.ini) file")
    parser.add_argument(
        '--resources', metavar="R", type=str, nargs='+', required=True,
        help="Names of resources corresponding to headers in `config.ini` to install")
    parser.add_argument(
        '--skip', metavar='S', type=str, nargs='+', default='', 
        help='Names of resources to skip during setup')
    main(parser.parse_args())
