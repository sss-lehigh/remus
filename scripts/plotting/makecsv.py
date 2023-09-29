from absl import app
from absl import flags
import re
import os
import sys
FLAGS = flags.FLAGS
flags.DEFINE_list("columns", [
    "exp", "host", "sys_size", "dev_name", "dev_port", "capacity", "buf_size", "leader_fixed",
    "policy", "duration", "testtime", "throughput", "lat_avg", "lat_50p", "lat_99p", "lat_99_9p", "lat_max",
    "p1_aborts", "p2_aborts", "attempts", "skipped", "proposed"
],
    "Columns to include in the generated .csv file",
    short_name="c")
flags.DEFINE_string("path",
                    None,
                    "Path to root directory containing experimental data.",
                    required=True,
                    short_name="p")
flags.DEFINE_string("exp", None, "Name of experiment folder.", required=True)
flags.DEFINE_string("testbed",
                    None,
                    "Testbed used during experiment.",
                    required=True)


# A mapping from column name to regular expression that correspond to the line containing the measurement.
column_re_map = {
    "exp": r".*\[CONF\].*exp_name=(.*)",
    "host": r".*\[CONF\].*host=(.*)",
    "sys_size": r".*\[CONF\].*sys_size=(\d*)",
    "dev_name": r".*\[CONF\].*dev_name=(.*)",
    "dev_port": r".*\[CONF\].*dev_port=(.*)",
    "capacity": r".*\[CONF\].*capacity=(\d*)",
    "buf_size": r".*\[CONF\].*buf_size=(\d*)",
    "leader_fixed": r".\[CONF\].*leader_fixed=(.*)",
    "policy": r".\[CONF\].*policy=(.*)",
    "duration": r".\[CONF\].*duration=(\d*.\d*)ms",
    "testtime": r".\[CONF\].*testtime=(\d*.\d*)ms",
    "throughput": r".*\[THRU\].*throughput=(\d*.\d*)",
    "lat_avg": r".*\[LAT\].*lat_avg=(\d*.\d*)",
    "lat_50p": r".*\[LAT\].*lat_50p=(\d*.\d*)",
    "lat_99p": r".*\[LAT\].*lat_99p=(\d*.\d*)",
    "lat_99_9p": r".*\[LAT\].*lat_99_9p=(\d*.\d*)",
    "lat_max": r".*\[LAT\].*lat_max=(\d*.\d*)",
    "p1_aborts": r".*!> p1_aborts=(\d*)",
    "p2_aborts": r".*!> p2_aborts=(\d*)",
    "attempts": r".*!> attempts=(\d*)",
    "skipped": r".*!> skipped=(\d*)",
    "proposed": r".!> proposed=(\d*)",
}


def main(argv):
    # Write the column names
    lines = []
    line = ""
    for c in FLAGS.columns:
        line += (c + "," if c != FLAGS.columns[-1] else c)
    line += "\n"
    lines.append(line)
    # Read every data file. The directory is structured by testbed --> host --> algorithm --> datafiles
    root = os.path.join(FLAGS.path, FLAGS.exp, FLAGS.testbed)
    for host in os.listdir(root):
        for algo in os.listdir(os.path.join(root, host)):
            for config in os.listdir(os.path.join(root, host, algo)):
                line = ""
                # Parse the columns
                with open(os.path.join(root, host, algo, config)) as datafile:
                    contents = datafile.read()
                    for c in FLAGS.columns:
                        is_last = c != FLAGS.columns[-1]
                        match = re.search(column_re_map[c], contents)
                        if match is not None:
                            line += (match.group(1) +
                                     "," if is_last else match.group(1))
                        else:
                            line += "," if is_last else ""
                    line += "\n"
                    lines.append(line)
    for l in lines:
        print(l, end='')


if __name__ == "__main__":
    app.run(main)
