# Joint model
All commands below assume you are in the directory `experiments/unified_model`.

## Training joint model with synthetic data
For example, to train on YCBV data (assume rendered data in the folder named `ycbv_0830_v2`), run the following:
```bash 
train.py --dataset_dir=DATASETS_PATH/CRISP_data/unified_renders/ycbv_0830_v2  --bop_data_dir=DATASETS_PATH/casper-data/bop/bop_datasets --preload_to_mem=False
```

Note that make sure the relevant original dataset directories are set: if you are training with synthetic data generated from ReplicaCAD, you need to specify the path to replicaCAD dataset. 
If you are training with synthetic data generated from multiple datasets, you need to specify the paths to all the datasets (currently only supports ReplicaCAD and YCBV). 

Additional parameters can be found in `ExpSettings` in `train.py`.
In particular, checkpoints by default are saved in `./model_ckpts`, and each run will generate a folder inside `./model_ckpts` that has a datetime prefix and a suffix of 5 alphanumerical characters separated by an underscore.
Tensorboard logs will be dumped at two places: in the same folder where model checkpoints are saved, and in `./logs` folder.
Visualization can be turned on by setting the relevant parameter to `True` via CLI arguments.

## Testing joint model 
You can use the same `train.py` to test the trained joint model (assume model trained with `ycbv_0830_v2`, with checkpoint at `./model_ckpts/202309082214_YLCCT/checkpoint.pth`):
```bash
train.py --dataset_dir=DATASETS_PATH/unified_renders/ycbv_0830_v2 --bop_data_dir=DATASETS_PATH/bop/bop_datasets --replicacad_data_dir=DATASETS_PATH/habitat_data/replica_cad --preload_to_mem=False --test_only=True --export_all_pred_recons_mesh=True --export_average_pred_recons_mesh=True --checkpoint_path=./model_ckpts/202309082214_YLCCT/checkpoint.pth --batch_size=1 --vis_sdf_sample_points=True
```

## Datasets sanity checks
If the model is not behaving correctly, it's possible that the synthetic datasets generated are wrong. 
You can run `dataset_checks.py` to ensure that the dataset makes sense (replace `DATASET_NAME` with the dataset you want to test):
```bash 
dataset_checks.py --dataset_dir=DATASETS_PATH/unified_renders/DATASET_NAME --bop_data_dir=DATASETS_PATH/bop/bop_datasets --preload_to_mem=False
```
