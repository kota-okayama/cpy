FROM nvidia/cuda:12.3.1-base-ubuntu22.04
# https://hub.docker.com/r/nvidia/cuda

USER root

RUN apt update \
    && apt -y install locales \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && apt install -y vim less fontconfig git swig curl \
    && apt install --no-install-recommends --yes build-essential

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

# Install Miniconda
COPY install_miniconda.sh environment.yml /tmp
RUN /tmp/install_miniconda.sh

ARG USERNAME=user
ARG GROUPNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $GROUPNAME \
    && useradd -m -s /bin/bash -u $UID -g $GID $USERNAME \
    && mkdir -p /home/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
RUN echo '. /opt/miniconda/bin/activate && conda activate saturn' >> /home/user/.bashrc
