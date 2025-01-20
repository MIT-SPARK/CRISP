import os
import dataclasses
import numpy as np
import torch
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser
import lightning as L
import time
import lightning.fabric as LF
import datetime
from tqdm import tqdm
import torch.utils.data as tchdata
import pyvista as pv
import seaborn

# local lib imports
from crisp.datasets.bop_nocs import BOPNOCSDataset
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.registration import align_nocs_to_depth
# from crisp.models.shape import create_sdf_samples_generic
from crisp.models.nocs import *
from crisp.models.loss_functions import nocs_loss, siren_udf_loss, siren_sdf_fast_loss, metric_sdf_loss
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.math import se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista
from crisp.utils.evaluation_metrics import rotation_error, translation_error
from crisp.utils import diff_operators
import crisp.utils.sdf

from experiments.unified_model.dataset_checks import single_batch_sanity_test

# print(len(next(os.walk('../datasets/ycbv_train_real/train_real/'))[1]))

# fileList=os.listdir('../datasets/ycbv_train_real/train_real/000000/depth')
# print(len(fileList))

# breakpoint()

shape_ds = BOPNOCSDataset(
        folder_path='../datasets/ycbv_train_real/train_real',
        ycbv_dataset_path="../datasets/ycbv_0830_v2",
        shapenet_dataset_path="../datasets/shapenet_renders_0801/",
        bop_dataset_path="../datasets/bop_datasets/",
        replicacad_dataset_path="../datasets/chair_nocs_v5",
        preload_to_mem=False,
        pc_size=60000,
        sample_surface_points_count=1500,
        sample_local_nonmnfld_points_count=1500,
        sample_global_nonmnfld_points_count=4000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=(-1.0, 1.0),
        debug_vis=False,
        data_to_output=[
            "rgb",
            "nocs",
            "coords",
            "instance_segmap",
            "metadata",
            "normals",
            "depth",
            "cam_intrinsics",
            "cam_pose",
            "sdf",
            "object_pc",
        ],
    )


train_dataloader = tchdata.DataLoader(shape_ds, shuffle=True, num_workers=2, batch_size=10)

for i, batch in tqdm(enumerate(train_dataloader)):
            # nocs related
            rgb_img, segmap, gt_nocs, coords, gt_normals, gt_sdf = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["coords"],
                batch["normals"],
                batch["sdf"],
            )