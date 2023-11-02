FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV LLVM_VERSION=16

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc

RUN apt-get update && \
    env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y curl \
                       vim \
                       git \
                       build-essential \
                       python3-dev \
                       python-is-python3 \
                       wget \
                       ca-certificates \
                       lsb-release \
                       software-properties-common \
                       gpg-agent \
                       doxygen \
                       graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN git config --global user.name  "ubuntu" && \
    git config --global --add safe.directory "*"

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} all && \
    rm llvm.sh && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && rm -f get-pip.py

RUN pip3 install lit==15.0.6 cmake

WORKDIR /opt
RUN git clone https://github.com/Tencent/ncnn && \
    cd ncnn && \
    mkdir -p build; cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j8; make install 

WORKDIR /home 
RUN git clone --recursive https://github.com/digital-nomad-cheng/tvm && \
    cd tvm; ./docker/install/ubuntu_download_arm_compute_lib_binaries.sh && \
    mkdir -p build; cd build && \
    cmake -DUSE_NCNN_CODEGEN=ON -DUSE_LLVM=ON -DUSE_ARM_COMPUTE_LIB=ON .. && \
    make -j12

RUN echo "export TVM_HOME=/home/tvm" >> ~/.bashrc && \
    echo 'export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}' >> ~/.bashrc 

RUN pip install --user numpy decorator attrs typing-extensions psutil scipy tornado 'xgboost>=1.1.0' cloudpickle && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/digital-nomad-cheng/tvm_project_course

