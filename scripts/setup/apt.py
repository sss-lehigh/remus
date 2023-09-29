from resources import __Resource__
import subprocess
import json


class APTPackages(__Resource__):
    packages = None
    repos = None

    def __init__(self, config):
        assert(config is not None)
        section_config = config["apt"]
        self.packages = json.loads(section_config["packages"])
        self.repos = json.loads(section_config["repos"])

    def prepared(self):
        return True  # Never skip the setup

    def setup(self):
        repos_string = ' '.join(r for r in self.repos)
        packages_string = ' '.join(p for p in self.packages)

        apt_add_cmd = ['sudo', 'add-apt-repository', '-y']
        apt_add_cmd.extend(self.repos)
        result = subprocess.run(apt_add_cmd)
        if result.returncode != 0:
            raise RuntimeError(
                'Failed to add APT repositories: ' + repos_string)
        result = subprocess.run(['sudo', 'apt-get', 'update'])
        if result.returncode != 0:
            raise RuntimeError('Failed to update APT')
        result = subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'])
        if result.returncode != 0:
            raise RuntimeError('Failed to upgrade APT')
        install_cmd = ['sudo', 'apt-get', 'install', '-y']
        install_cmd.extend(self.packages)
        result = subprocess.run(install_cmd)
        if result.returncode != 0:
            raise RuntimeError('Failed to install packages: ' + packages_string)
