import pickle
import math
from PIL import Image
import os
import argparse
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from jsonargparse import ArgumentParser
import lightning as L
import lightning.fabric as LF
import datetime
import itertools
from tqdm import tqdm
import torch.utils.data as tchdata

# local lib imports
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *

from crisp.utils.sdf import create_sdf_samples_generic
import utils
import utils.sdf
from crisp.utils.math import (
    depth_to_point_cloud_map_batched,
    SO3_from_axis_angle,
    make_se3_batched,
    make_scaled_se3_batched,
    se3_inverse_batched_torch,
)
from crisp.utils.visualization_utils import visualize_pcs_pyvista, visualize_meshes_pyvista


@dataclass
class ExpSettings:
    # dataset
    batch_size = 10
    dataset_dir: str
    bop_data_dir: str
    shapenet_data_dir: str = None
    replicacad_data_dir: str = None
    model_ckpts_save_dir: str = "./exp_results"
    preload_to_mem: bool = False
    scenes_to_load: tuple = (0,)
    pc_size: int = 60000
    per_batch_sample_surface_points_count: int = 1500
    per_batch_sample_global_nonmnfld_points_count: int = 3000
    dataset_debug_vis: bool = False

    # backbone model
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    freeze_pretrained_backbone_weights: bool = True
    backbone_model_path: str = None
    log_root_dir: str = "logs"

    # implicit recons model
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"

    # nocs model
    nocs_network_type: str = "xyz"

    # loading model & testing
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    checkpoint_path: str = None

    # automatically populated if missing
    exp_id: str = None
    pose_noise_scale: float = 0

    visualize: bool = True


def get_mesh_from_shape_code(shp_code, model):
    def model_fn(coords):
        return model.recons_net.forward(shape_code=shp_code, coords=coords)

    sdf_grid, voxel_size, voxel_grid_origin = create_sdf_samples_generic(
        model_fn=model_fn, N=128, max_batch=64**3, cube_center=np.array([0, 0, 0]), cube_scale=2.5
    )

    pred_mesh = utils.sdf.convert_sdf_samples_to_mesh(
        sdf_grid=sdf_grid,
        voxel_grid_origin=voxel_grid_origin,
        voxel_size=voxel_size,
        offset=None,
        scale=None,
    )
    return pred_mesh


def main(opt):
    LF.seed_everything(42)
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)

    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        local_backbone_model_path=opt.backbone_model_path,
        freeze_pretrained_weights=True,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
    )
    print("Generating ground truth latent codes for YCB-Video objects")

    print("Loading model checkpoint.")
    state = torch.load(opt.checkpoint_path)
    model.load_state_dict(state["model"])
    model = model.cuda()

    shape_ds = UnifiedObjects(
        folder_path=opt.dataset_dir,
        shapenet_dataset_path=opt.shapenet_data_dir,
        bop_dataset_path=opt.bop_data_dir,
        replicacad_dataset_path=opt.replicacad_data_dir,
        preload_to_mem=opt.preload_to_mem,
        pc_size=opt.pc_size,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
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
            "object_pc",
        ],
    )
    dl = tchdata.DataLoader(shape_ds, shuffle=True, num_workers=2, batch_size=opt.batch_size)

    raw_obj_code_db = {}
    if os.path.exists(os.path.join("./data/ycbv", "raw_obj_codes.npy")):
        print("Loading raw object codes from file.")
        raw_obj_code_db = np.load(os.path.join("./data/ycbv", "raw_obj_codes.npy"), allow_pickle=True).item()
    else:
        print("Generating new object codes.")
        # load synthetic dataset
        # go through dataset and log latent code <-> obj name
        # tsne projection
        # robust centroid and save to file
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            print(f"Currently at {i} out of {len(dl)}.")
            (rgb_img, depth, segmap, gt_nocs, object_pc, cam_intrinsics, gt_world_T_cam, coords, metadata) = (
                batch["rgb"].cuda(),
                batch["depth"].cuda(),
                batch["instance_segmap"].cuda(),
                batch["nocs"].cuda(),
                batch["object_pc"].cuda(),
                batch["cam_intrinsics"].cuda(),
                batch["cam_pose"].cuda(),
                batch["coords"].cuda(),
                batch["metadata"],
            )
            bs = rgb_img.shape[0]
            mask = (segmap == 1).unsqueeze(1)

            # run the corrector which takes the SDF,
            # perturbed GT NOCS and depths and give us corrected NOCS, and calculate registration.
            shape_code = model.forward_shape_code(img=rgb_img)

            for j in range(bs):
                if metadata["obj_name"][j] not in raw_obj_code_db.keys():
                    raw_obj_code_db[metadata["obj_name"][j]] = [shape_code[j, ...].numpy(force=True)]
                else:
                    raw_obj_code_db[metadata["obj_name"][j]].append(shape_code[j, ...].numpy(force=True))

    # now we have the shape code db, proceed with robust centroid & projection
    processed_obj_code_db = {}
    for k, codes in raw_obj_code_db.items():
        processed_obj_code_db[k] = geometric_median(np.array(codes))

    # save
    np.save(os.path.join("./data/ycbv", "processed_obj_codes.npy"), processed_obj_code_db)

    # visualize the shape codes
    sorted_obj_labels = sorted(list(processed_obj_code_db.keys()))
    for obj_label in sorted_obj_labels:
        code = processed_obj_code_db[obj_label]
        print(f"Visualizing {obj_label}...")
        # load shape
        shp_code = torch.tensor(code).cuda().unsqueeze(0).float()
        pred_mesh = get_mesh_from_shape_code(shp_code, model)

        # load CAD
        obj_data = dl.dataset.distinct_objects["ycbv"][obj_label]
        obj_pc = obj_data["surface_points"]

        if opt.visualize:
            # interactive visualize
            visualize_meshes_pyvista(
                [pred_mesh, obj_pc],
                plotter_shape=(1, 1),
                mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                subplots=None,
                show_axes=True,
                show_bounds=True,
            )
        else:
            # save images
            pl = visualize_meshes_pyvista(
                [pred_mesh, obj_pc],
                plotter_shape=(1, 1),
                mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                subplots=None,
                show_axes=True,
                show_bounds=True,
                off_screen=True,
            )
            image = pl.screenshot(None, return_img=True)
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(exp_dump_path, f"{obj_label}.jpeg"), "JPEG")

    return


if __name__ == "__main__":
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
    main(opt)
