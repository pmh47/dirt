ARG CUDA_BASE_VERSION
ARG UBUNTU_VERSION
ARG CUDNN_VERSION

# use CUDA + OpenGL
FROM nvidia/cudagl:${CUDA_BASE_VERSION}-devel-ubuntu${UBUNTU_VERSION}
MAINTAINER Domhnall Boyle (domhnallboyle@gmail.com)

# arguments from command line
ARG CUDA_BASE_VERSION
ARG UBUNTU_VERSION
ARG CUDNN_VERSION
ARG TENSORFLOW_VERSION

# set environment variables
ENV CUDA_BASE_VERSION=${CUDA_BASE_VERSION}
ENV CUDNN_VERSION=${CUDNN_VERSION}
ENV TENSORFLOW_VERSION=${TENSORFLOW_VERSION}

# install apt dependencies
RUN apt-get update && apt-get install -y \
	python \
	python-pip \
	git \
	vim \
	wget

# install newest cmake version
RUN apt-get purge cmake && cd ~ && wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz && tar -xvf cmake-3.14.5.tar.gz
RUN cd ~/cmake-3.14.5 && ./bootstrap && make && make install

# setting up cudnn
RUN apt-get install -y --no-install-recommends \             
	libcudnn7=$(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION) \             
	libcudnn7-dev=$(echo $CUDNN_VERSION)-1+cuda$(echo $CUDA_BASE_VERSION) 
RUN apt-mark hold libcudnn7 && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install tensorflow-gpu==$(echo $TENSORFLOW_VERSION)

# install dirt
ENV CUDAFLAGS='-DNDEBUG=1'
RUN cd ~ && git clone https://github.com/pmh47/dirt.git && \ 
 	pip install dirt/

# run dirt test command
RUN python ~/dirt/tests/square_test.py
