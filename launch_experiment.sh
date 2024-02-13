#!/bin/bash
set -e # Halt the script on any error

# launch_experiment.sh -  Completely set up an experiment.  This involves
#                         copying code to remote machines, running configuration
#                         scripts on those machines, and then building screenrc
#                         configurations for either running the experiments, or
#                         connecting to the remote machines.
#
#                         Note that this script assumes you're doing your
#                         development on a Linux system that is not very
#                         different than the target CloudLab machine.  In
#                         particular, it is assumed that you can build locally,
#                         copy the executable file to cloudlab, and run it
#                         remotely.
#
#                         Lastly, note the "-no-prep" flag will skip running the
#                         "prepare_to_run.sh" script on the remote node.

#
# This section has the customization points for the script.  For now, you need
# to edit these by hand.
#

# The actual machine names that were allocated by cloudlab.  We assume the
# domain is the same for all, and specify it second, for brevity
#
# WARNING:  Order matters for the machines, because there is an implicit
#           conversion to names like node0, node1, etc.  Be sure to follow the
#           order in the CloudLab "List View" for your experiment.
machines=(apt137 apt148) #apt149) #apt085)
domain=apt.emulab.net

# The user who is going to be using ssh/scp to connect to cloudlab.  It is
# expected that keys are already set up.
user=depaulm

for machine in ${machines[@]}; do
  ssh $user@$machine.$domain echo "Connected"
done

# The executable file that needs to be sent over to cloudlab
exefile="./build/examples/iht/iht_rome"
exename=$(basename ${exefile})

# Configuration command... we're going to make this file on the remote machines
config_command=prepare_to_run.sh

# Configuration for the experiment
port=9999

# Names of screenrc files.  If the trailing prefix is '.screenrc', then our
# '.gitignore' file will be happy.
screenrc_run=run.screenrc
screenrc_dev=dev.screenrc

# Names of packages that we need in order to run the code
#
# NB: It's a big string, because that's easier later :)
package_deps="protobuf-compiler librdmacm-dev ibverbs-utils libspdlog-dev libfmt-dev"

#
# This section has internal configuration code.  You shouldn't need to edit it.
#

# The 0-indexed number of nodes
last_valid_index=$((${#machines[@]}-1)) # one less than ${#machines[@]}

# The 1-indexed number of nodes
num_nodes=${#machines[@]}

# construct "node_listing", a comma-separated, ordered list of node/port pairs.
node_listing="node0:${port}"
for i in `seq $last_valid_index`
do
    node_listing="${node_listing},node${i}:${port}"
done

#
# This section starts calling programs to do real work
#

# Build the binary
echo "Building executable file(s)"
make
echo -e "Done\n\n"

# Build the script for configuring the machines
tmp_script_file="$(mktemp)" || exit 1
echo 'echo `hostname`' > ${tmp_script_file}
echo "sudo apt update" >> ${tmp_script_file}
echo "sudo apt upgrade -y" >> ${tmp_script_file}
echo "sudo apt install -y ${package_deps}" >> ${tmp_script_file}

# Send the script and executable to the cloudlab machines.  `scp` has clean
# output, so we can do this in a simple loop.
echo "Transferring files to ${machines[*]}"
for m in ${machines[*]}
do
    scp ${exefile}         ${user}@${m}.${domain}: &
    scp ${tmp_script_file} ${user}@${m}.${domain}:${config_command} &
done
wait
echo -e "Done\n\n"
rm ${tmp_script_file}

# Run the configuration script on each machine... Since `apt` has ugly output,
# we use `screen`
if [[ $1 == "-no-prep" ]]; then
    echo -e "'-no-prep' flag detected... skipping machine configuration\n\n"
else
    echo "Configuring machines"
    tmp_screen="$(mktemp)" || exit 1
    echo 'startup_message off' > ${tmp_screen}
    echo 'defscrollback 10000' >> ${tmp_screen}
    echo 'autodetach on' >> ${tmp_screen}
    echo 'escape ^jj' >> ${tmp_screen}
    echo 'defflow off' >> ${tmp_screen}
    echo 'hardstatus alwayslastline "%w"' >> ${tmp_screen}
    for i in `seq 0 ${last_valid_index}`
    do
        echo "screen -t nodes${i} ssh ${user}@${machines[$i]}.${domain} bash ${config_command}" >> ${tmp_screen}
    done
    screen -c ${tmp_screen}
    echo -e "Done\n\n"
    rm ${tmp_screen}
fi

#
# We are not actually going to run the program from this script, but we will
# make some nice screen scripts for running the program on CloudLab
#

# Default config: No copyright page, big scrollback buffer
echo 'startup_message off' > ${screenrc_run}
echo 'defscrollback 10000' >> ${screenrc_run}
echo 'autodetach on' >> ${screenrc_run}
echo 'escape ^jj' >> ${screenrc_run}
echo 'defflow off' >> ${screenrc_run}
echo 'hardstatus alwayslastline "%w"' >> ${screenrc_run}
cp ${screenrc_run} ${screenrc_dev}

# The screenrcs differ in terms of what they do upon connecting to a node:
for i in `seq 0 ${last_valid_index}`
do
    echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain} ./${exename} --node_count ${num_nodes} --node_id ${i} --runtime 10 --op_count 0 --contains 80 --insert 10 --remove 10 --key_lb 0 --key_ub 10000 --region_size 25 --thread_count 1 --qp_max 1 --unlimited_stream; bash" >> ${screenrc_run}
    echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain}" >> ${screenrc_dev}
done

# Provide some instructions
echo "run 'screen -c ${screenrc_run}' to launch the experiment"
echo "run 'screen -c ${screenrc_dev}' to log in to all machines"
