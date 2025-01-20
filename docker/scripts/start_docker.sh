#!/bin/bash

CODE_DIR=${1:-"/home/$USER/code/CRISP"}
docker run -it --user="$(id -u $USER)":"$(id -g $USER)" \
--gpus all \
--shm-size 8G \
--env="DISPLAY=:1" \
--env="CUDA_VISIBLE_DEVICES=0" \
--env="PYTHONPATH=$PYTHONPATH:/opt/project/CRISP/src" \
--volume="$CODE_DIR:/opt/project/CRISP" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" crisp:latest bash
