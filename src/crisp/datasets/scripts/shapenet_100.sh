#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
blenderproc run gen_unified.py --bop_path=${SCRIPT_DIR}/../../../../data/bop/bop_datasets --replicacad_path=${SCRIPT_DIR}/../../../../data/replica_cad --spe3r_path=${SCRIPT_DIR}/../../../../data/spe3r --shapenet_path=${SCRIPT_DIR}/../../../../data/shapenet --objects_config_to_load=${SCRIPT_DIR}/configs/shapenet_100.yml --num_frames 1 --resolution=[224,224]
