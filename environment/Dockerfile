# Ubuntu 22.04
FROM ubuntu:jammy 

RUN apt-get update -y && \
    apt-get upgrade -y 

# Move to the working directory
COPY . .
ENV DEBIAN_FRONTEND=noninteractive
# Install the required packages + isolating ca-certificates for debugging purposes
RUN apt-get install --no-install-recommends -y ca-certificates && \
    apt-get install --no-install-recommends -y $(cat req.list) && \
# Clean up to free up additional disk space
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR "/root"