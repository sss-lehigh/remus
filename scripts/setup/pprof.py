from resources import HostedArchive
import os


class Pprof(HostedArchive):

    def __init__(self, config):
        super().__init__(config, "pprof")

    def setup(self):
        super().setup()
        path = os.path.join(self.dest_dir, self.name,
                            self.name + "-" + self.hosted_name)
        if os.system(f"cd {path} && go install .") != 0:
            raise
