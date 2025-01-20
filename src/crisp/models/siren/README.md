# SDF Test

Train an SDF model:
```bash
python train_sdf.py --model_type=sine --point_cloud_path=./data/tless_obj_000001.xyz --experiment_name=test_tless_obj_2
```

Test the trained SDF model:
```bash 
python test_sdf.py --checkpoint_path=./logs/test_tless_obj_1/checkpoints/model_final.pth --experiment_name=test_tless_obj_1 
```

Note: in SIREN training the input model mesh is rescaled into [-1, 1], and the generated mesh is not rescaled back to the
original scale of the model.