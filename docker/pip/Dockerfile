
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
# Ubuntu 24.04.1 LTS
# Python 3.12.3
# Cuda 12.6.3

WORKDIR /root
USER root

RUN apt update \
    && apt -y install locales \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && apt install -y python3-pip vim less fontconfig git curl swig

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=JST-9
ENV TERM=xterm

# Base python packages
COPY docker/pip/__upgrade_base.txt ./
RUN pip install --break-system-packages -r __upgrade_base.txt

# MKL
COPY docker/pip/install_mkl.sh ./
RUN ./install_mkl.sh

# Numpy and Scipy with MKL
COPY docker/pip/install_mkl_numpy_scipy.sh ./
RUN ./install_mkl_numpy_scipy.sh

# Tensorflow (v2.19.0)
# 対応していない numpy をインストールすると tensorflow が numpy を自動でインストールし直すので留意
RUN pip install --break-system-packages tensorflow[and-cuda]==2.19.0 --upgrade-strategy only-if-needed

# CMAKE
COPY docker/pip/install_cmake.sh ./
RUN ./install_cmake.sh

# FAISS
COPY docker/pip/install_faiss.sh ./
RUN ./install_faiss.sh

# Fasttext
COPY docker/pip/install_fasttext.sh ./
RUN ./install_fasttext.sh

# ===== The following are not executed by GitHub Actions. =====

# Other python packages
COPY docker/pip/__required.txt ./
RUN pip install --break-system-packages -r __required.txt

# ubuntuユーザーのUIDとGIDを変更
ARG UID=1000
ARG GID=1000
RUN usermod -u $UID ubuntu && \
    groupmod -g $GID ubuntu

WORKDIR /home/ubuntu/workspace
COPY docker/pip/docker-entrypoint.sh /usr/local/bin
USER ubuntu
RUN mkdir -p /home/ubuntu/workspace
ENTRYPOINT ["docker-entrypoint.sh"]
