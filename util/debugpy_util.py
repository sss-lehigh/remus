import debugpy
from absl import flags

flags.DEFINE_integer("rome_debugpy_port", 59718,
                     "Port to launch debug server on")
flags.DEFINE_boolean("rome_debugpy", False,
                     "Launch debug server and wait for client")


def debugpy_hook():
    if (flags.FLAGS.rome_debugpy):
        debugpy.listen(flags.FLAGS.rome_debugpy_port)
        debugpy.wait_for_client()
