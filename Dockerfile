FROM ubuntu:24.04

# Get latest software (standard ubuntu package manager)
RUN apt-get update -y
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3 cmake git man curl libnuma-dev librdmacm-dev libibverbs-dev screen lsb-release software-properties-common openssh-client

# Get LLVM 18
RUN curl -O https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 18
RUN rm ./llvm.sh

# Start in the `/root` folder
WORKDIR "/root"

# 1. Build the Docker image
# sudo docker build -t remusv2-spaa25 .

# 2. Run the Docker container
# sudo docker run -it --rm -v "$(pwd)":/root remusv2-spaa25

# NB: If you are a member of the SPAA25 team, please obtain the SSH key from
#     the team and place it in local/ssh-key.spaa
# eval "$(ssh-agent -s)" && ssh-add local/ssh-key.spaa
