from abc import ABC, abstractclassmethod
import pathlib
import sys
import os

curr_path = pathlib.Path(__file__).absolute()
sys.path.append(str(curr_path.parent.parent))
from util import *  # nopep8


class __Resource__(ABC):
    """ Defines the interface for a resource, which can be a tool, a library or basically anything that needs to be set up before things get started.
    """
    @abstractclassmethod
    def prepared(self):
        pass

    @abstractclassmethod
    def setup(self):
        pass


class HostedResource(__Resource__):
    """ A hosted resource is one that is available at some remote location. This class takes care of downloading it and validating its checksum.
    """
    name = None
    hosted_name = None
    resource = None
    url = None
    dest_dir = None
    checksum = None
    mode = None
    download_path = None
    dest_path = None

    def __init__(self, config, section_name):
        assert(config is not None)
        section_config = config[section_name]
        self.name = section_config["name"]
        self.hosted_name = section_config["hosted_name"]
        self.resource = section_config["resource"]
        self.url = section_config["url"]
        self.dest_dir = section_config["dest_dir"]
        self.checksum = section_config["checksum"]
        self.mode = section_config["mode"]
        self.download_path = os.path.join(self.dest_dir, self.resource)
        self.dest_path = os.path.join(self.dest_dir, self.name)

    def prepared(self):
        return os.path.exists(self.download_path)

    def setup(self):
        if not self.prepared():
            download(self.url, self.dest_dir)
            validate_checksum(self.download_path, self.mode, self.checksum)


class HostedArchive(HostedResource):
    """A hosted archive is like a hosted resource except that it is a compressed archive and must be unpacked (e.g., zip). This class extends `HostedResource` and calls the base class `setup()` method to obtain the resource, then it unpacks the archive and removes the original compressed file.
    """
    def __init__(self, config, section_name):
        super().__init__(config, section_name)

    def cleanup(self):
        rm(self.download_path)

    def prepared(self):
        return os.path.exists(self.dest_path)

    def setup(self):
        if not self.prepared():
            super().setup()
            unpack(self.download_path, self.dest_path)
            self.cleanup()
