FROM nvidia/cuda:9.1-cudnn7-devel-centos7

RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm && \
    yum update -y && \
    yum install -y which openssh-clients blas-devel make zlib-devel openssl-devel

RUN cd /tmp && \
    curl -O https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tar.xz && \
    tar -xf Python-3.6.8.tar.xz && \
    cd Python-3.6.8 && \
    ./configure --prefix=/usr/local/python3.6 && make -j4 && make install && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    /usr/local/python3.6/bin/python3 get-pip.py && \
    /usr/local/python3.6/bin/python3 -m pip install -U pip

# RUN apt-get update -y && \
#     python3-dev \
#     python3-pip \
#     python3-wheel \
#     python3-setuptools \
#     git \
#     cmake \
#     libblas3 \
#     libblas-dev \
#     && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN CHAINER_BUILD_CHAINERX=1 CHAINERX_BUILD_CUDA=1 /usr/local/python3.6/bin/pip3 install --no-cache-dir cupy-cuda91==5.1.0 chainer==5.1.0
