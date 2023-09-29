FROM ubuntu:focal
ENV TERM xterm-color

# Configure a user.
ARG USER=develop
RUN useradd -ms /bin/bash -G conda,sudo -k /etc/skel ${USER}
RUN echo "${USER}:${USER}" | chpasswd

RUN apt-get install -y python

USER ${USER}
COPY scripts /tmp/scripts
RUN cd /tmp/scripts && python setup/run.py --resources all 

# This image exposes a port that hosts an ssh server for users to connect to.
# The container port will be 22, but it must be mapped to host port when running
# the container. If you are using the command line interface, this can be
# accomplished by running the image with the command `docker run -p 8022:22 <image_name>`.
# Then, ssh-ing into the container using `ssh -p 8022 bundledrefs@localhost` and the 
# password `bundledrefs`.
EXPOSE 22
RUN service ssh start
CMD ["/usr/sbin/sshd", "-D"]