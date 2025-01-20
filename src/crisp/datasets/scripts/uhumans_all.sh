#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
blenderproc run gen_unified.py --bop_path=${SCRIPT_DIR}/../../../../data/bop/bop_datasets --replicacad_path=${SCRIPT_DIR}/../../../../data/replica_cad --spe3r_path=${SCRIPT_DIR}/../../../../data/spe3r --uhumans_path=${SCRIPT_DIR}/../../../../data/uhumans --objects_config_to_load=${SCRIPT_DIR}/configs/uhumans_all.yml --num_frames 200 --resolution=[224,224]  --camera_sample_max_radius=10 --camera_sample_elevation_min=-90 --camera_sample_elevation_max=90
