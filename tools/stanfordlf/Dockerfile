# Define Base Image
# FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# Install ceres solver
RUN cd /opt
WORKDIR /opt
RUN apt-get install -y libatlas-base-dev libsuitesparse-dev libgoogle-glog-dev libeigen3-dev libsuitesparse-dev
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver
WORKDIR /opt/ceres-solver
RUN mkdir build
RUN cd build
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
RUN make
RUN make install

# Install Colmap
RUN cd /opt
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap
RUN cd colmap
WORKDIR /opt/colmap
RUN mkdir build
RUN cd build
WORKDIR /opt/colmap/build
RUN cmake ..
RUN make
RUN make install
RUN cd /

# RUN mkdir -p /home/app
# RUN mkdir /home/app/data

# # Specify working directory

# WORKDIR /home/app

# # Copy script from Host machine to working directory of the container
# COPY colmap.sh .

# # Specify entry point of the container

# ENTRYPOINT ["sh", "colmap.sh"]

# Pick up some TF dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libzmq3-dev \
        ffmpeg \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN pip3 --no-cache-dir install --upgrade ipython && \
    pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        jupyterlab \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        scikit-image \
        imageio==2.4.0 \
        && \
    python3 -m ipykernel.kernelspec

# Install OpenCV
RUN apt-get update && apt-get install -y libopencv-dev python-opencv && \
    echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc

# Install GLFW
RUN apt-get update && apt-get install -y libglfw3-dev
RUN apt-get install -y imagemagick

RUN mkdir -p /workspace
WORKDIR /workspace
RUN chmod -R a+w /workspace