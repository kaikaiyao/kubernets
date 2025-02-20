# Use a newer PyTorch image with CUDA 12.1 and PyTorch 2.1.0
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Install system dependencies with combined RUN to minimize layers
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y \
    git \
    cmake \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    gcc-10 \
    g++-10 \
    curl \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA paths (already included in base image, but verify)
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Use a writable directory for torch extensions (avoid /root permission issues)
ENV TORCH_EXTENSIONS_DIR=/workspace/torch_extensions
RUN mkdir -p ${TORCH_EXTENSIONS_DIR} && chmod -R 777 ${TORCH_EXTENSIONS_DIR}

# Install Python dependencies (including ninja==1.10.2)
RUN pip install --no-cache-dir \
    ninja==1.10.2 \
    torchmetrics \
    torch-fidelity \
    matplotlib \
    pandas \
    click \
    requests \
    tqdm \
    pyspng \
    scikit-learn \
    statsmodels \
    seaborn \
    pycryptodome \
    cryptography \
    lpips \
    pycryptodome \
    imageio-ffmpeg==0.4.3

# Download pre-trained model weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L -o /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth \
    https://download.pytorch.org/models/vgg16-397923af.pth && \
    curl -L -o /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth \
    https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth

ARG CACHEBUST=1

# Clone repositories
RUN git clone https://github.com/kaikaiyao/kubernets.git /workspace/kubernets \
    && git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /workspace/kubernets/stylegan2-ada-pytorch

# Create the parent directory and subdirectories for results
RUN mkdir -p /workspace/kubernets/results && \
    for i in {0..7}; do \
        mkdir -p /workspace/kubernets/results/$i; \
    done

# Set working directory and default command
WORKDIR /workspace/kubernets
CMD ["true"]