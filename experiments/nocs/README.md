## Preparing relevant files

- precompute_gt_scales.py: This script is used to generate the ground truth scales, rotations and translations for the
  camera and real datasets.
- gen_obj_scales.py: After running precompute_gt_scales.py, this script is used to generate the object scales used for
  training the SDF model (unnormalized).
- gen_nocs_stub.py (in src/datasets/scripts): This script is used to generate the stub dataset (objects_info.yml) for
  the joint model. This is needed for our unified objects class to compute the SDF values for each object.

## Notes on NOCS dataset

NOCs consist of the following datasets:

- CAMERA: These are synthetic images generated from rendering. Here are its folders:
    - camera_val25k: validation/test set
    - train: training images
    - ikea
    - composed_depths
      Real: These are real images recorded. Here are its folders:
    - real_train:
    - real_test:
      GT Annotations: Annotations for camera_val25k and real_test
    - val: camera_val25k
    - real_test: Real test
      Obj models: Models for the datasets
    - real_test
    - real_train
    - train
    - val

Folder organization: It appears from their repo that they expect the following organization:

- camera
    - train
    - val
- camera_full_depths
    - Note: this is not used in the paper, but they are foreground object depths composed with background depths
    - train
    - val
- real
    - train
    - val
- coco

Coco seems to be used for classes.