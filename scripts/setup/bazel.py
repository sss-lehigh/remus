from resources import __Resource__
import json
from util import *


class Bazel(__Resource__):
    packages = None

    def __init__(self, config):
        self.packages = json.loads(config["bazel"]["packages"])

    def prepared(self):
        return False

    def setup(self):
        for p in self.packages:
            subprocess.run(['go', 'install', p])
        try_add_path("$(go env GOPATH)/bin")
        try_add_bashrc("alias bazel='bazelisk'")
