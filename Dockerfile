FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install libprotobuf-dev protobuf-compiler  -y
RUN apt-get install cmake -y
RUN apt-get install clang-15 libabsl-dev librdmacm-dev libibverbs-dev libgtest-dev libbenchmark-dev libfmt-dev libspdlog-dev libgmock-dev -y
RUN apt-get install libc6-dev-i386 -y
