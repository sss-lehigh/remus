# Notes for building your environment

The **build.sh** script will automatically adjust to your computer architecture - no need to modify this script. 

## For MAC (M1/M2, M3)

**Disable the Roseta Desktop feature in Docker**
As of 06-2024 there is a bug in the experimental features of Rosetta Desketop that causes an illegal instruction in 
the x86_64 emulation of ca-certificates. The dpkg post-installation triggers will cause the build to fail if this feature is not disabled. 

Settings > General > `Use Rosetta for x86_64/amd64 emulation on Apple Silicon`
Uncheck this. 

Now build your environment:
`chmod +x build.sh && ./build.sh`

## For x86_64

Build your environment:
`chmod +x build.sh && ./build.sh`

## Running the container 

This should give you root access to the build image. 
`chmod +x run.sh && ./run.sh`

## Building the source code

`chmod +x build_local.sh && ./build_local.sh`

## Debugging

Modify the following  to the Dockerfile if you run into issues with the build: 
`RUN apt-get -o Debug::pkgProblemResolver=true install --no-install-recommends -y ca-certificates`

This will automatically be flushed to stdout. To see more Docker-specific issues, please see:
`docker logs <container-id>`