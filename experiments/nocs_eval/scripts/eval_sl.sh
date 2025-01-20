CHECKPOINT_PATH=/home/jnshi/code/Hydra-Objects/experiments/nocs/model_ckpts/202407092229_929WL/checkpoint.pth

export CORE_USE_TORCH_COMPILE=false
cd /home/jnshi/code/Hydra-Objects/experiments/ycbv_object_slam
PYTHONPATH=/home/jnshi/code/Hydra-Objects/src:/home/jnshi/code/Hydra-Objects python train.py \
--checkpoint_path=${CHECKPOINT_PATH} \
--pose_noise_scale=0 \
--gen_mesh_for_test=False \
--visualize_rgb=False \
--objects_info_path=/mnt/jnshi_data/datasets/hydra_objects_data/ycbv_objects_info/objects_info.yaml \
--unified_ds_dir=/mnt/jnshi_data/datasets/hydra_objects_data/unified_renders/ycbv_0830_v2 \
--test_only=True \
--normalized_recons=False \
--pipeline_output_degen_condition_numbers=True \
--cert_depths_quantile=0.98 \
--cert_depths_eps=0.01 \
--use_corrector=False \
--pipeline_nr_downsample_before_corrector=2000 \
--use_mf_geometric_shape_corrector=False \
--use_mf_shape_code_corrector=False