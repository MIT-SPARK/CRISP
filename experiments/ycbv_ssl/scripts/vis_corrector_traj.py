import copy
import cv2 as cv
import csv
import numpy as np
import os
from jsonargparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import lightning as L
import lightning.fabric as LF
from lightning.fabric.strategies import SingleDeviceStrategy
import datetime
from tqdm import tqdm
import torchvision.transforms.functional as F
from torchvision.transforms import functional as tvf
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data as tchdata

# project local imports
from crisp.backend.slam import PoseNode, PoseEdge
from crisp.datasets import bop
from crisp.datasets.bop import is_bop_obj_det_valid
from crisp.utils.math import se3_inverse_numpy
from crisp.utils.constants import *
import crisp.utils.sdf
from crisp.utils.visualization_utils import imgs_show
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.file_utils import safely_make_folders
from crisp.backend.bop_to_g2o import save_g2o, save_object_index, read_object_index
from crisp.utils.sdf import create_sdf_samples_generic
from crisp.models.pipeline import Pipeline
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.models.certifier import FrameCertifier
from crisp.models.registration import align_nocs_to_depth


@dataclass
class ExpSettings:
    # dataset
    bop_ds_dir: str = "../../data/bop/bop_datasets"
    ds_name: str = "ycbv"
    split: str = "train_real"
    model_ckpts_save_dir: str = "./model_ckpts"
    scenes_to_load: tuple = (0,)

    # backbone model
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    freeze_pretrained_backbone_weights: bool = True
    log_root_dir: str = "logs"

    # implicit recons model
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"

    # framewise certification parameters
    use_frame_certifier: bool = True
    cert_depths_clamp_thres: float = 10
    cert_depths_quantile: float = 0.9
    cert_depths_eps: float = 0.1

    # corrector parameters
    corrector_lr: float = 0.05
    corrector_max_iters: int = 25
    corrector_log_loss_traj: bool = True
    correcotr_log_dump_dir: str = "./corrector_logs"

    # loading model & testing
    visualize_rgb: bool = False
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    checkpoint_path: str = None

    # automatically populated if missing
    exp_id: str = None
    pose_noise_scale: float = 0


if __name__ == "__main__":
    print("Visualize corrector trajectory")
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    # generate random timestamped experiment ID
    if opt.exp_id is None:
        opt.exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"

    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    safely_make_folders([exp_dump_path])
    parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))

    opt = ExpSettings(**opt)

    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        freeze_pretrained_weights=True,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
    ).cuda()
    model.eval()

    print("Loading model checkpoint.")
    state = torch.load(opt.checkpoint_path)
    model.load_state_dict(state["model"])

    # load corrector traj file
    corrector_traj_files = [
        f for f in os.listdir(opt.correcotr_log_dump_dir) if os.path.isfile(os.path.join(opt.correcotr_log_dump_dir, f))
    ]

    if len(corrector_traj_files) == 0:
        raise ValueError(f"{os.path.abspath(opt.correcotr_log_dump_dir)} is empty.")
    else:
        for traj_f in corrector_traj_files:
                data = np.load(os.path.join(opt.correcotr_log_dump_dir, traj_f), allow_pickle=True)

                B = 1
                if data[0]["shape_code"].ndim != 1:
                    B = data[0]["shape_code"].shape[0]

                for bid in range(B):
                    meshes = []
                    for iter in [0, len(data) - 1]:
                        print(f"iter={iter}")

                        if torch.tensor(data[iter]["shape_code"], device="cuda").ndim == 1:
                            shp_code = torch.tensor(data[iter]["shape_code"], device="cuda").unsqueeze(0)
                        else:
                            shp_code = torch.tensor(data[iter]["shape_code"], device="cuda")[bid, ...].unsqueeze(0)

                        # gen shape
                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shp_code, coords=coords)

                        with torch.no_grad():
                            sdf_grid, voxel_size, voxel_grid_origin = create_sdf_samples_generic(
                                model_fn=model_fn,
                                N=256,
                                max_batch=64**3,
                                cube_center=np.array([0, 0, 0]),
                                cube_scale=2.5,
                            )
                            pred_mesh = crisp.utils.sdf.convert_sdf_samples_to_mesh(
                                sdf_grid=sdf_grid,
                                voxel_grid_origin=voxel_grid_origin,
                                voxel_size=voxel_size,
                                offset=None,
                                scale=None,
                            )
                            meshes.append(pred_mesh)

                    # dump meshes
                    for m in [meshes[0], meshes[-1]]:
                        m.show()
