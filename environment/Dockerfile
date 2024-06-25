# Ubuntu 22.04
FROM ubuntu:jammy 

RUN apt-get update -y
RUN apt-get upgrade -y

# Move to the working directory
COPY . .

# Configure dpkg certs
RUN dpkg --configure -a

# Install the required packages
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y $(cat req.list)

WORKDIR "/root"