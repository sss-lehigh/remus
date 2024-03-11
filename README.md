*DO NOT MERGE INTO LIBROME*

# Overview

Rome is a collection of useful utilities for research-oriented programming. 
This is the ethos of this project, to accumulate helpful tools that can serve as the basis for development.

Remus is Romulus's brother that Romulus killed over disagreeing about where Rome should be.
Any features here may be killed off at some point before making it to rome, hence the name.

Remus Supports:
* Workload driver library (see `rome/workload`) for experimental evaluation
* Logging utilities (see `rome/logging`)
* Measurements library (see `rome/metrics`)
* Sundry other utilities that we don't know where to place yet (see `rome/util`)
* RDMA (see `rome/rdma`)
* NUMA (see `rome/numa`)

## Building

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
* libnuma (if compiling NUMA support)
* CUDA 12.3 (if compiling GPU support)
* doxygen (for building DOCS)
* cmake (3.18 or later)

Your GPU must be Volta or later.

We have the following configuration options/flags:
* GPU (ON or OFF) will compile with GPU support
* KEEP (ON or OFF) will keep ptx and cubin files
* DOCS (ON or OFF) will create documentation
* NUMA (ON or OFF) will enable rome::numa
* LOG\_LEVEL (TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, or OFF) for logging
* CXX\_STANDARD (20 or 23) for the C++ standard
* CUDA\_ARCHITECTURES (semicolon seperated list of SM numbers) 

## Using Remus

`tools/install.sh` is a script to install Remus to `/opt/remus` on your machine.

After installing you can include remus in any CMake project by setting:
`-DCMAKE_PREFIX_PATH=/opt/remus/lib/cmake -DCMAKE_MODULE_PATH=/opt/remus/lib/cmake`
when running cmake.

Then in your CMakeLists.txt you can write `find_package(rome REQUIRED)`.

Rome can be accessed by linking in CMake to any of these libraries: 
- `rome::workload` 
- `rome::logging`
- `rome::metrics` 
- `rome::rdma` 
- `rome::util` 
- `rome::protos` 
- `rome::hds`

