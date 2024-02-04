# compress data from graphs into a text file
import sys
import os
sys.path.insert(1, '..')
import protos.experiment_pb2 as protos
import configparser
import google.protobuf.text_format as text_format
import matplotlib.pyplot as plt

# Read the configuration
config = configparser.ConfigParser()
config.read('plot.ini')
prefix = eval(config.get('dirs', 'prefix'))
suffix_s = eval(config.get('dirs', 'paths'))
output = eval(config.get('dirs', 'output_file'))

nodes = eval(config.get('desc', 'node_count'))
threads = eval(config.get('desc', 'thread_count'))

title = eval(config.get('desc', 'title'))
subtitle = eval(config.get('desc', 'subtitle'))
x_axis = eval(config.get('desc', 'x_axis'))
x_label = eval(config.get('desc', 'x_label'))

# Get the protos into a list
proto = {}
for suffix in suffix_s:
    exp_node_c = 0
    exp_thread_c = 0
    for file in os.listdir(os.path.join(prefix, suffix)):
        if file.__contains__("node0") and file.__contains__("pbtxt"):
            f = open(os.path.join(prefix, suffix, file), "rb")
            p = text_format.Parse(f.read(), protos.ResultProto())
            exp_node_c =  p.params.node_count
            exp_thread_c =  p.params.thread_count
            f.close()
    for file in os.listdir(os.path.join(prefix, suffix)):
        if not file.__contains__("pbtxt"):
            continue
        f = open(os.path.join(prefix, suffix, file), "rb")
        p = text_format.Parse(f.read(), protos.ResultProto())
        node_c =  p.params.node_count
        thread_c =  p.params.thread_count
        if node_c not in proto:
            proto[node_c] = {}
        if thread_c not in proto[node_c]:
            proto[node_c][thread_c] = {}
        if node_c > exp_node_c and thread_c > exp_thread_c:
            continue
        proto[node_c][thread_c][file] = p
        f.close()

# Graph the nodes
x, y = [], []
for n in nodes:
    for t in threads:
        one_tick = x_axis
        one_tick = one_tick.replace("$node$", str(n))
        one_tick = one_tick.replace("$thread$", str(t))
        x.append(one_tick)
        mean_total = 0
        for node_file, node_data in proto[n][t].items():
            count = 0
            for it in node_data.driver:
                count += 1
                mean_total += it.qps.summary.mean
            assert(count == t)
        assert(len(proto[n][t]) == n)
        y.append(mean_total)

# Save the graph as a text file
txt_file_path = input("Name of db:\n")
with open(txt_file_path, "w") as f:
    str_build = [title, subtitle, x_label, "Total Throughput (ops/second)", str(x), str(y)]
    str_save = "\n".join(str_build)
    f.write(str_save)