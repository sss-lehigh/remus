# Overview

Remus is a collection of useful utilities for research-oriented programming. 
This is the ethos of this project, to accumulate helpful tools that can serve as the basis for development.

Remus Supports:
<!-- TODO: OLD -->

* Workload driver library (see `remus/workload`) for experimental evaluation
* Logging utilities (see `remus/logging`)
* Measurements library (see `remus/metrics`)
* Sundry other utilities that we don't know where to place yet (see `remus/util`)
* RDMA (see `remus/rdma`)
* NUMA (see `remus/numa`)

## Building
<!-- TODO: OLD -->
We have tested the following configurations:

|OS           |  Compiler            |
|-------------|----------------------|
|Ubuntu 22.04 | gcc-11               |
|Ubuntu 22.04 | gcc-12               |
|Ubuntu 22.04 | clang-15             |
|Ubuntu 22.04 | clang-14             |
|Ubuntu 22.04 | clang-15 & nvcc-12.3 |

To build/run your machine requires:
* protobuf-compiler 
* librdmacm-dev 
* ibverbs-utils 
* libspdlog-dev 
* libfmt-dev
* nlohmann-json3-dev
* libnuma-dev (if compiling NUMA support)
* CUDA 12.3 (if compiling GPU support)
* doxygen (for building DOCS)
* cmake (3.18 or later)

Your GPU must be Volta or later.

We have the following configuration options/flags:
* GPU (ON or OFF) will compile with GPU support
* KEEP (ON or OFF) will keep ptx and cubin files
* DOCS (ON or OFF) will create documentation
* NUMA (ON or OFF) will enable remus::numa
* LOG\_LEVEL (TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, or OFF) for logging
* CXX\_STANDARD (20 or 23) for the C++ standard
* CUDA\_ARCHITECTURES (semicolon seperated list of SM numbers) 

## Using Remus

<!-- [abc] Make sure that the GLIBCXX standards are compatible between your build environment and the cloudlab env. -->

1. Launch experiment on RDMA-capable Cloudlab cluster
    - Deploy on Ubuntu 24.04 for native gcc-13 support. 
2. Edit `cloudlab_common.sh` 
    - Fill out machines, domain, user, and your cloudlab sshkey path.
3. `bash cloudlab_install_deps.sh`
    - This will install the dependencies on the remote machines
4. `bash cloudlab_rebuild.sh` 
    - Sends over the pre-compiled binary to the remote machines
    - Configuration for the multi-screen setup
    - run `screen -c run.screenrc` to launch the experiment 
    - run `screen -c dev.screenrc` to log in to all machines

**build.sh** can also be used to compile the binary locally. 
Here, you can configure the log level (DEBUG or RELEASE).