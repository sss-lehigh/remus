import subprocess
from resources import HostedArchive, __Resource__
from util import *
import os
import json


class Golang(HostedArchive):

    def __init__(self, config):
        super().__init__(config, "golang")

    def setup(self):
        try:
            subprocess.run(["go", "version"], check=True)
        except FileNotFoundError:
            super().setup()
        except Exception as e:
            raise(e)
        try_add_path(os.path.join(self.dest_path, "go", "bin"))


class GoPackages(__Resource__):
    packages = None

    def __init__(self, config):
        self.packages = json.loads(config["go"]["packages"])

    def prepared(self):
        return False

    def setup(self):
        for p in self.packages:
            subprocess.run(["go", "install", p], check=True)
        try_add_path("\$(go env GOROOT)/bin")
