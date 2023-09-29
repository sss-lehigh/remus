from resources import HostedArchive


class Zookeeeper(HostedArchive):
    def __init__(self, config):
        super().__init__(config, "zookeeper")