# Getting Started

- Install remus using `tools/install.sh` on your ubuntu machine

To use RDMA:

- Initialize logging using `REMUS_INIT_LOG()`
- Create `remus::rdma::Peer` objects, with node ids, the node, and the port number
- Create a `remus::rdma::rdma_capability` per each thread (Peer) on the node 
- Initialize the pool with `init_pool` for the memory size and a vector of peers
    - Note initialization must be done in parallel for each thread to not block
- Call `RegisterThread` on each thread that should participate in RDMA
- Use the APIs
- See `rdma/test/multinode.cc` for a reference

Common Errors:
- Make sure your user limits are set high enough with ulimit. You will need to create many files to use remus. The launch script does this automatically.
- If you receive errors about creating connections, you are likely choosing ports that are unavailable. Try different ports instead.

## Reading Options

Remus provides a way to parse command-line arguments (but if you prefer, you can do it manually).

- Define all your arguments in one map using remus::util
    - The first string is the flag that denotes a certain argument
    - The second is a description of an argument
    - The third, if applicable, is a default value
```{c++}
using namespace remus::util;
auto ARGS = { 
    I64_ARG("--param1", "A required integer argument"),

    I64_ARG_OPT("--param2", "An optional integer argument with a default value of 0", 0),

    F64_ARG("--param3", "A required floating-point argument"),

    F64_ARG_OPT("--param4", "An optional floating-point argument with a default value of 0.0", 0.0),

    BOOL_ARG_OPT("--param5", "A boolean argument. Defaults to false, if the flag is set, argument is true"),

    STR_ARG("--param6", "A required string argument")

    STR_ARG_OPT("--param7", "A required string argument with the empty string as a default", "")
};
``` 
- Initialize the argument map by first importing via import_args and then parsing via parse_args.
    - Note: only call parse args once. If it fails, a mandatory arg was absent.
    - Can import args structure multiple times, merging if able. Otherwise returning an error.
```{c++}
remus::util::ArgMap args;
auto res = args.import_args(ARGS);
if (res) {
    REMUS_FATAL(res.value());
}
res = args.parse_args(argc, argv);
if (res) {
    args.usage();
    REMUS_FATAL(res.value());
}
```

- Read the parsed arguments
    - Use iget (integer), fget (float), sget (string), bget (boolean)
    - Pass the flag you want to read
```{c++}
int iparam = args.iget("--int_param");
double fparam = args.fget("--float_param");
bool bparam = args.bget("--bool_param");
std::string sparam = args.sget("--string_param");
```

## rdma_capability API

- Memory Allocation
    - Allocate\<T>(size)
        - Will allocate an array of "size" elements of type T.
            - Size defaults to 1
        - Will align to 64-bytes
    - Deallocate\<T>(ptr, size)
        - Deallocates the ptr. If size is not equal to that of the original allocation, will silently leak memory
            - Size defaults to 1
- Read, PartialRead, ExtendedRead
    - Copies memory from the provided pointer to a local buffer. The difference between each function is how much it copies.
    - Will internally allocate memory and return it to you. You have to free it to avoid leaks OR
        - If you use prealloc, then it will avoid allocation and copy into prealloc.
- Write
    - Will write a value to the provided pointer.
    - Will internally allocate RDMA-accessible memory to send from, unless prealloc is used
        - If it does internally allocate, it will also handle de-allocation.
- CompareAndSwap
    - Will atomically compare-and-swap 8 bytes of data and return the previous value
- AtomicSwap
    - Will atomically swap 8 bytes of data and return the previous value. 
        - Is CAS-loop under the hood

## Benchmarking

- Remus provides a small suite of benchmarking tools (again, optional to use)
- The Workload Driver
    - Accepts a ClientAdapter and a Stream in constructor
    - Call Run, executes the stream until it generates a StreamTerminatedStatus
    - driver->ToMetrics() then generates the results
- ClientAdapter
    - A class that has Start, Apply, Stop
        - Start is called in initialization
        - For each operation from the stream, it is passed in to the Apply function.
        - Stop is called in termination
        - An operation is part of the ClientAdapter's template
```{c++}
template <class Operation> class Client {
  // static_assert(::remus::IsClientAdapter<Client, Operation>);
public:
  Client() {}

  // Initialization
  remus::util::Status Start() {
    return remus::util::Status::Ok();
  }

  // Runs the next operation
  remus::util::Status Apply(const Operation &op) {
    return remus::util::Status::Ok();
  }

  // Cleanup and termination
  remus::util::Status Stop() {
    return remus::util::Status::Ok();
  }  
};
```
- The streams
    - PrefilledStream: a vector of operations to feed the application
    - FixedLengthStream: calls a generator function to feed the application. Limited by a length value. 
        - Useful to avoid the overhead of a large vector of operations if the operation is trivially computable.

## Two-Sided RDMA

This needs to be documented
