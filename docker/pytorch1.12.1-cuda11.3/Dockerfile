FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory for installing packages
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app && chown -R user:user /opt
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the conda environment including all the pytorch-related packages (using Miniforge)
ENV PATH=$HOME/mambaforge/bin:$PATH
COPY base_environment.yml /app/environment.yml
RUN sudo chmod -R 777 /app
RUN curl -sLo ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
 && chmod +x ~/mambaforge.sh \
 && ~/mambaforge.sh -b -p ~/mambaforge \
 && rm ~/mambaforge.sh \
 && mamba env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && mamba clean -ya

ENV TZ=America/New_York
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 freeglut3 freeglut3-dev -y

# C++ libaries and tools
RUN sudo apt-get update && sudo apt-get install cmake libeigen3-dev pv pigz llvm-6.0 autoconf zlib1g-dev -y

# Set up additional packages
COPY environment.yml /app/environment.yml
RUN sudo chmod -R 777 /app
RUN mamba env update -n base -f /app/environment.yml \
 && rm /app/environment.yml

# setup work directory for project
RUN mkdir /opt/project
WORKDIR /opt/project

# Install BOP renderer
RUN sudo chmod -R 777 /opt
ENV OSMESA_PREFIX=/opt/osmesa LLVM_PREFIX=/lib/llvm-6.0 LLVM_BUILD=0 LLVM_VERSION=6.0.1
RUN git clone https://github.com/jingnanshi/bop_renderer.git && mkdir bop_renderer/osmesa-install/build \
    && cd bop_renderer/osmesa-install/build && bash ../osmesa-install.sh

# Activate mamba base environment by default
RUN mamba init
RUN echo "mamba activate base" >> ~/.bashrc
CMD ["bash"]
