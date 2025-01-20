import pathlib
from enum import Enum
import random
import pickle
import math
import numpy as np
from jsonargparse import ArgumentParser
import lightning.fabric as LF
import datetime
from tqdm import tqdm
import pyvista as pv
import torch.utils.data as tchdata

# local lib imports
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.datasets.spe3r import SatReconNOCSDataset
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh
import crisp.utils as utils
from crisp.utils.math import (
    depth_to_point_cloud_map_batched,
    SO3_from_axis_angle,
    make_se3_batched,
    make_scaled_se3_batched,
    se3_inverse_batched_torch,
)
from crisp.utils.visualization_utils import visualize_pcs_pyvista, visualize_meshes_pyvista
from experiments.satellite.train import ExpSettings

SPLIT="train"
#SPLIT = "test"


class TestMode(Enum):
    INTERPOLATE = 1
    EXTRAPOLATE = 2
    SIMPLEX_SAMPLE = 3
    SAVE_MESH = 4
    SAVE_RANDOM_MESH = 5


def get_mesh_from_shape_code(shp_code, model):
    def model_fn(coords):
        return model.recons_net.forward(shape_code=shp_code, coords=coords)

    sdf_grid, voxel_size, voxel_grid_origin = create_sdf_samples_generic(
        model_fn=model_fn, N=128, max_batch=64**3, cube_center=np.array([0, 0, 0]), cube_scale=5
    )

    pred_mesh = utils.sdf.convert_sdf_samples_to_mesh(
        sdf_grid=sdf_grid,
        voxel_grid_origin=voxel_grid_origin,
        voxel_size=voxel_size,
        offset=None,
        scale=None,
    )
    return pred_mesh


def save_screenshots_from_shape_codes(model, interp_shp_codes, export_names):
    # generate meshes for the shape codes
    for j in tqdm(range(interp_shp_codes.shape[0])):
        shp_code = torch.tensor(interp_shp_codes[j, ...]).unsqueeze(0).float().cuda()

        def model_fn(coords):
            return model.recons_net.forward(shape_code=shp_code, coords=coords)

        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
            model_fn=model_fn,
            N=128,
            max_batch=64**3,
            cube_center=np.array([0, 0, 0]),
            cube_scale=2,
        )

        pred_mesh = convert_sdf_samples_to_mesh(
            sdf_grid=sdf_grid,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=None,
            scale=None,
        )

        # mesh_name = f"{o1}_{o2}_step_{j}.png"
        # safely_make_folders([f"./exports/{prefix}_{model_name}/{o1}_{o2}"])
        # file_name = f"./exports/{prefix}_{model_name}/{o1}_{o2}/{mesh_name}"
        containing_folder = pathlib.Path(export_names[j]).parent
        safely_make_folders([containing_folder])
        pv_mesh = pv.wrap(pred_mesh)
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(pv_mesh)
        pl.show(screenshot=export_names[j])
    return


def save_mesh_from_shape_codes(model, shape_code_library, export_folder):
    # generate meshes for the shape codes
    for obj_name, shp_cd in shape_code_library.items():
        shp_code = torch.tensor(shp_cd).unsqueeze(0).float().cuda()

        def model_fn(coords):
            return model.recons_net.forward(shape_code=shp_code, coords=coords)

        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
            model_fn=model_fn,
            N=128,
            max_batch=64**3,
            cube_center=np.array([0, 0, 0]),
            cube_scale=2,
        )

        pred_mesh = convert_sdf_samples_to_mesh(
            sdf_grid=sdf_grid,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=None,
            scale=None,
        )

        # mesh_name = f"{o1}_{o2}_step_{j}.png"
        # safely_make_folders([f"./exports/{prefix}_{model_name}/{o1}_{o2}"])
        # file_name = f"./exports/{prefix}_{model_name}/{o1}_{o2}/{mesh_name}"
        safely_make_folders([export_folder])
        pred_mesh.export(os.path.join(export_folder, f"{obj_name}.obj"))

    return


def interpolate_shape_code(shp_code_1, shp_code_2, steps=10):
    x = np.linspace(0, 1, steps)
    delta = shp_code_2 - shp_code_1
    interpolated_shape_codes = np.tile(shp_code_1.reshape((1, -1)), (steps, 1)) + x.reshape((steps, 1)) * delta.reshape(
        (1, -1)
    )
    return interpolated_shape_codes


def extrapolate_shape_code(shp_code_1, shp_code_2, steps=20):
    x = np.linspace(-3, 4, steps)
    delta = shp_code_2 - shp_code_1
    extrapolated_shape_codes = np.tile(shp_code_1.reshape((1, -1)), (steps, 1)) + x.reshape((steps, 1)) * delta.reshape(
        (1, -1)
    )
    return extrapolated_shape_codes


def simplex_sample_shape_code(shape_code_library, samples=20):
    sampled_shape_codes = np.zeros((samples, shape_code_library.shape[0]))
    for i in range(samples):
        shape_coeffs = np.random.uniform(shape_code_library.shape[1], 1, size=shape_code_library.shape[1])
        shape_coeffs = shape_coeffs / np.sum(shape_coeffs)
        sampled_shape_codes[i, :] = (shape_code_library @ shape_coeffs.reshape((-1, 1))).squeeze()
    return sampled_shape_codes


def main(opt):
    LF.seed_everything(42)
    TEST_MODE = TestMode.SAVE_RANDOM_MESH
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    print("Generating ground truth latent codes for SPE3R objects")

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
        lateral_layers_type=opt.nocs_lateral_layers_type,
        backbone_input_res=opt.backbone_input_res,
        normalize_shape_code=opt.recons_shape_code_normalization,
        recons_shape_code_norm_scale=opt.recons_shape_code_norm_scale,
    )

    print("Loading model checkpoint.")
    state = torch.load(opt.checkpoint_path)
    ckpt_name = opt.checkpoint_path.split("/")[-2]
    model.load_state_dict(state["model"])
    model = model.cuda()

    train_data_to_output = [
        "rgb",
        "nocs",
        "coords",
        "normals",
        "sdf",
        "instance_segmap",
        "model_name",
    ]
    shape_ds = SatReconNOCSDataset(
        # SatReconDataset params
        opt.dataset_dir,
        split=SPLIT,
        model_name=None,
        input_H=opt.image_size[0],
        input_W=opt.image_size[1],
        # Unified objects_params
        unified_objects_dataset_path=opt.spe3r_unified_objects_dataset_dir,
        normalized_recons=opt.normalized_recons,
        preload_to_mem=False,
        data_to_output=train_data_to_output,
    )

    dl = tchdata.DataLoader(shape_ds, shuffle=True, num_workers=2, batch_size=opt.batch_size)

    shp_code_save_folder = os.path.join(f"./data/spe3r/{SPLIT}/spe3r_{ckpt_name}")
    raw_obj_code_db = {}
    if os.path.exists(os.path.join(shp_code_save_folder, "raw_obj_codes.npy")):
        print("Loading raw object codes from file.")
        raw_obj_code_db = np.load(os.path.join(shp_code_save_folder, "raw_obj_codes.npy"), allow_pickle=True).item()
    else:
        print("Generating new object codes.")
        # load synthetic dataset
        # go through dataset and log latent code <-> obj name
        # tsne projection
        # robust centroid and save to file
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            (rgb_img, segmap, gt_nocs, coords, model_name) = (
                batch["rgb"].cuda(),
                batch["instance_segmap"].cuda(),
                batch["nocs"].cuda(),
                batch["coords"].cuda(),
                batch["model_name"],
            )
            bs = rgb_img.shape[0]
            mask = (segmap == 1).unsqueeze(1)

            # run the corrector which takes the SDF,
            # perturbed GT NOCS and depths and give us corrected NOCS, and calculate registration.
            shape_code = model.forward_shape_code(img=rgb_img)

            for j in range(bs):
                if model_name[j] not in raw_obj_code_db.keys():
                    raw_obj_code_db[model_name[j]] = [shape_code[j, ...].numpy(force=True)]
                else:
                    raw_obj_code_db[model_name[j]].append(shape_code[j, ...].numpy(force=True))

        safely_make_folders([shp_code_save_folder])
        np.save(os.path.join(shp_code_save_folder, "raw_obj_codes.npy"), raw_obj_code_db, allow_pickle=True)

    # shape codes collected
    all_obj_names = sorted(raw_obj_code_db.keys())
    if TEST_MODE == TestMode.SIMPLEX_SAMPLE:
        shape_code_dict = {}
        for obj_name in raw_obj_code_db.keys():
            shape_code_dict[obj_name] = np.mean(np.array(raw_obj_code_db[obj_name]), axis=0)
        shape_code_library = np.zeros(
            (shape_code_dict[list(shape_code_dict.keys())[0]].shape[0], len(shape_code_dict.keys()))
        )
        for i, obj_name in enumerate(shape_code_dict.keys()):
            shape_code_library[:, i] = shape_code_dict[obj_name]

        simplex_sampled_shape_codes = simplex_sample_shape_code(shape_code_library, samples=100)
        mesh_screenshot_names = [
            f"./exports/spe3r/{SPLIT}/simplex_sample_{ckpt_name}/sample_{j}.png"
            for j in range(simplex_sampled_shape_codes.shape[0])
        ]
        save_screenshots_from_shape_codes(model, simplex_sampled_shape_codes, mesh_screenshot_names)
    elif TEST_MODE == TestMode.SAVE_MESH:
        shape_code_dict = {}
        for obj_name in raw_obj_code_db.keys():
            shape_code_dict[obj_name] = np.mean(np.array(raw_obj_code_db[obj_name]), axis=0)
        save_mesh_from_shape_codes(model, shape_code_dict, export_folder=f"./exports/spe3r/{SPLIT}/meshes_{ckpt_name}/")

    elif TEST_MODE == TestMode.SAVE_RANDOM_MESH:
        shape_code_dict = {}
        for obj_name in raw_obj_code_db.keys():
            # randomly select a shape code
            shape_code_dict[obj_name] = np.array(raw_obj_code_db[obj_name])[
                np.random.randint(0, len(raw_obj_code_db[obj_name])), ...
            ]
        save_mesh_from_shape_codes(
            model, shape_code_dict, export_folder=f"./exports/spe3r/{SPLIT}/random_meshes_{ckpt_name}/"
        )

    else:
        for i, obj_name in tqdm(enumerate(all_obj_names[:-1]), total=len(all_obj_names) - 1):
            o1 = obj_name
            o2 = all_obj_names[i + 1]
            shp_code_1, shp_code_2 = random.choice(raw_obj_code_db[o1]), random.choice(raw_obj_code_db[o2])

            if TEST_MODE == TestMode.INTERPOLATE:
                interp_shp_codes = interpolate_shape_code(shp_code_1, shp_code_2, steps=10)
                prefix = "interpolate"
            elif TEST_MODE == TestMode.EXTRAPOLATE:
                interp_shp_codes = extrapolate_shape_code(shp_code_1, shp_code_2, steps=20)
                prefix = "extrapolate"
            else:
                raise ValueError(f"Unsupported test mode: {TEST_MODE}")

            mesh_screenshot_names = [
                f"./exports/spe3r/{SPLIT}/{prefix}_{ckpt_name}/{o1}__{o2}/{o1}__{o2}_step_{j}.png"
                for j in range(interp_shp_codes.shape[0])
            ]
            save_screenshots_from_shape_codes(model, interp_shp_codes, mesh_screenshot_names)

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
