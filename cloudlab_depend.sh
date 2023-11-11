export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install libprotobuf-dev protobuf-compiler  -y
sudo apt-get install cmake -y
sudo apt-get install clang-15 libabsl-dev librdmacm-dev libibverbs-dev libgtest-dev libbenchmark-dev libfmt-dev libspdlog-dev libgmock-dev -y
sudo apt-get install libc6-dev-i386 -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib