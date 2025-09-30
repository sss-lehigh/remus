#!/bin/bash

# cl.sh
#
# A script for uploading and running Remus applications on CloudLab

set -e  # Halt the script on any error

# Print usage information
function usage {
    echo "cl.sh is a tool for running Remus applications on CloudLab"
    echo ""
    echo "Usage: cl.sh <cfg> <command> [arg]"
    echo ""
    echo "cfg is a configuration file, based on the cl.config template,"
    echo "that describes the CloudLab machines to be used"
    echo ""
    echo "The commands are:"
    echo ""
    echo "    install-deps    Perform one-time installation of dependencies"
    echo "                    onto CloudLab machines"
    echo "    build-run [arg] Upload an executable to CloudLab and run it"
    echo "    run       [arg] Run an experiment without re-uploading"
    echo "    run-debug [arg] Run an experiment with gdb"
    echo "    connect         Connect to CloudLab (i.e., for debugging)"
    echo ""
    echo "For commands that require an argument, arg is a configuration file,"
    echo "based on the cl.experiment template, that provides the executable"
    echo "name and the arguments to provide to the executable."
}

# Source the config file given as $1
function load_cfg {
    if [ ! -f $1 ]; then
        echo "Error: configuration file '$1' not found"; exit
    fi
    source $1
}

# SSH into machines once, to fix known_hosts
function cl_first_connect {
    echo "Performing one-time connection to CloudLab machines, to get known_hosts right"
    for machine in ${machines[@]}; do
        ssh $user@$machine.$domain echo "Connected"
    done
}

# Append the default configuration of a screenrc to the given file
function make_screen {
    echo 'startup_message off' >> $1
    echo 'defscrollback 10000' >> $1
    echo 'autodetach on' >> $1
    echo 'escape ^jj' >> $1
    echo 'defflow off' >> $1
    echo 'hardstatus alwayslastline "%w"' >> $1
}

# Check the status of IBV on the target machines
function check_ibv {
    echo "Checking ibv status:"
    for machine in ${machines[@]}; do
    echo "$machine:"
    ssh $user@$machine.$domain ibv_devinfo
    done
}

#  Configure the set of CloudLab machines
function cl_install_deps() {
    config_command=prepare_to_run.sh        # The script to put on remote nodes
    last_valid_index=$((${#machines[@]}-1)) # The 0-indexed number of nodes

    # Names of packages that we need to install on CloudLab
    package_deps="librdmacm-dev ibverbs-utils libnuma-dev gdb"

    # First-time SSH
    cl_first_connect

    # Build a script to run on all the machines
    tmp_script_file="$(mktemp)" || exit 1
    echo 'echo `hostname`' > ${tmp_script_file}
    echo "sudo apt update" >> ${tmp_script_file}
    echo "sudo apt upgrade -y" >> ${tmp_script_file}
    echo "sudo apt install -y ${package_deps}" >> ${tmp_script_file}

    # Send the script to all machines via parallel SCP
    echo "Sending configuration script to ${machines[*]}"
    for m in ${machines[*]}; do
        scp ${tmp_script_file} ${user}@${m}.${domain}:${config_command} &
    done
    wait
    rm ${tmp_script_file}

    # Use screen to run the script in parallel
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    for i in `seq 0 ${last_valid_index}`; do
        echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain} bash ${config_command}" >> ${tmp_screen}
    done
    screen -c ${tmp_screen}; rm ${tmp_screen}

    # Check the status of the IBV library on the target machines
    check_ibv
}

#  Build the binary and send it to CloudLab machines
function cl_build() {
    exename=$(basename ${exefile})  # The exe to send to CloudLab

    # Build the binary, send it to CloudLab
    make
    for m in ${machines[*]}; do
        scp ${exefile} ${user}@${m}.${domain}:${exename} &
    done
    wait
}

# Run a binary on the CloudLab machines
function cl_run() {
    exename=$(basename ${exefile})          # The exe to send to CloudLab
    last_valid_index=$((${#machines[@]}-1)) # The 0-indexed number of nodes

    # Set up a screen script for running the program on all machines
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    for i in `seq 0 ${last_valid_index}`; do
        echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain} ./${exename} --node-id ${i} ${experiment_args}; bash" >> ${tmp_screen}
    done
    screen -c $tmp_screen; rm $tmp_screen
}

# Run a binary on the CloudLab machines
function cl_run_no_interactive() {
    exename=$(basename ${exefile})          # The exe to send to CloudLab
    last_valid_index=$((${#machines[@]}-1)) # The 0-indexed number of nodes

    # Set up a screen script for running the program on all machines
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    for i in `seq 0 ${last_valid_index}`; do
        echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain} ./${exename} --node-id ${i} ${experiment_args}; exit" >> ${tmp_screen}
    done
    screen -c $tmp_screen; rm $tmp_screen
}

# Load a binary into gdb on the CloudLab machines
function cl_debug() {
    exename=$(basename ${exefile})          # The exe to send to CloudLab
    last_valid_index=$((${#machines[@]}-1)) # The 0-indexed number of nodes

    # Set up a screen script for running the program on all machines
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    # Add scrollback buffer to make screen scrollable
    echo "defscrollback 10000" >> ${tmp_screen}
    # Enable mouse scrolling and preserve terminal scrollback
    echo "mouse on" >> ${tmp_screen}
    echo "termcapinfo xterm* ti@:te@" >> ${tmp_screen}  
    # Disable flow control to avoid Ctrl-S freezing the terminal
    echo "defflow off" >> ${tmp_screen}      
    for i in `seq 0 ${last_valid_index}`; do
        echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain} gdb -ex \"r\" --args ./${exename} --node-id ${i} ${experiment_args}; bash" >> ${tmp_screen}
    done
    screen -c $tmp_screen; rm $tmp_screen
}

# Connect to CloudLab nodes (e.g., for debugging)
function cl_connect() {
    last_valid_index=$((${#machines[@]}-1)) # The 0-indexed number of nodes

    # Set up a screen script for connecting
    tmp_screen="$(mktemp)" || exit 1
    make_screen $tmp_screen
    for i in `seq 0 ${last_valid_index}`; do
        echo "screen -t node${i} ssh ${user}@${machines[$i]}.${domain}" >> ${tmp_screen}
    done
    screen -c $tmp_screen; rm $tmp_screen
}

# Get the important stuff out of the command-line args
cfg=$1      # The config file
cmd=$2      # The requested command
opt=$3      # The options to the command, if any
count=$#    # The number of command-line args

# Find the right command and do it
if [[ "$cmd" == "install-deps" && "$count" -eq 2 ]]; then
    load_cfg $cfg
    cl_install_deps
elif [[ "$cmd" == "build-run" && "$count" -eq 3 ]]; then
    load_cfg $cfg
    load_cfg $opt
    cl_build
    cl_run
elif [[ "$cmd" == "run" && "$count" -eq 3 ]]; then
    load_cfg $cfg
    load_cfg $opt
    cl_run
elif [[ "$cmd" == "run-debug" && "$count" -eq 3 ]]; then
    load_cfg $cfg
    load_cfg $opt
    cl_build
    cl_debug
elif [[ "$cmd" == "connect" && "$count" -eq 2 ]]; then
    load_cfg $cfg
    cl_connect
elif [[ "$cmd" == "build-run-no-interactive" && "$count" -eq 3 ]]; then
    load_cfg $cfg
    load_cfg $opt
    cl_build
    cl_run_no_interactive
else
    usage
fi
