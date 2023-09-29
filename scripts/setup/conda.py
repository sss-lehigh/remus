import subprocess
import sys
from resources import HostedResource
from util import *
import os


class Conda(HostedResource):
    conda = None
    env_file = None
    env_name = None
    install_path = os.path.join(os.environ['HOME'], 'miniconda3')

    def __init__(self, config):
        super().__init__(config, "conda")
        self.env_name = config["conda"]["env_name"]
        self.env_file = config["conda"]["env_file"]
        self.conda = os.path.join(self.install_path, "bin", "conda")

    def create_env(self):
        subprocess.run([self.conda, "env", "create", "-n", self.env_name, "-f",
                        self.env_file, "-q"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def init(self):
        subprocess.run(
            [self.conda, "init", "--all"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def setup(self):
        try:
            subprocess.run(['conda', '--version'], check=True)
            print("Conda already installed, skipping.")
        except FileNotFoundError:
            super().setup()
            # Install and intialize
            subprocess.run(['sudo', 'chmod', '+x', self.download_path])
            subprocess.run([self.download_path, "-b"])

            # Cleanup
            subprocess.run(["rm", self.download_path])
            subprocess.run(
                f"find {self.install_path} -follow -type f -name '*.a' -delete && find {self.install_path} -follow -type f -name '*.js.map' -delete && {self.install_path}/bin/conda clean -afy",
                shell=True)
        self.init()
        self.create_env()
        try_add_bashrc("conda activate " + self.env_name)
