FROM ubuntu:22.04
# Set environment variables
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get install libprotobuf-dev protobuf-compiler  -y
RUN apt-get install cmake -y
RUN apt-get install clang-15 libabsl-dev librdmacm-dev libibverbs-dev libgtest-dev libbenchmark-dev libfmt-dev libspdlog-dev libgmock-dev -y
RUN apt-get install libc6-dev-i386-cross -y
RUN apt-get install git sudo nlohmann-json3-dev gdb -y

WORKDIR /remus

COPY . /remus
