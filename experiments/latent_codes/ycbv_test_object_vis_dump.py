from enum import Enum
import random
from jsonargparse import ArgumentParser
import lightning.fabric as LF
import datetime
from tqdm import tqdm
import torch.utils.data as tchdata

# local lib imports
from crisp.datasets.bop_nocs import BOPNOCSDataset
from crisp.utils.file_utils import id_generator
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *

from crisp.utils.sdf import create_sdf_samples_generic
import crisp.utils as utils
from crisp.utils.visualization_utils import save_joint_visualizations, save_screenshots_from_shape_codes
from experiments.unified_model.train import ExpSettings


EXTRAPOLATE_OBJ_1 = "obj_000017"
EXTRAPOLATE_OBJ_2 = "obj_000009"


class TestMode(Enum):
    DUMP_NOCS_AND_RECONS_MESH = 0


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


def interpolate_shape_code(shp_code_1, shp_code_2, steps=10):
    x = np.linspace(0, 1, steps)
    delta = shp_code_2 - shp_code_1
    interpolated_shape_codes = np.tile(shp_code_1.reshape((1, -1)), (steps, 1)) + x.reshape((steps, 1)) * delta.reshape(
        (1, -1)
    )
    return interpolated_shape_codes


def extrapolate_shape_code(shp_code_1, shp_code_2, steps=20, start=-3, end=4, x=None):
    if x is None:
        x = np.linspace(start, end, steps)
    else:
        steps = len(x)
    delta = shp_code_2 - shp_code_1
    extrapolated_shape_codes = np.tile(shp_code_1.reshape((1, -1)), (steps, 1)) + x.reshape((steps, 1)) * delta.reshape(
        (1, -1)
    )
    return extrapolated_shape_codes, x


def simplex_sample_shape_code(shape_code_library, samples=20):
    sampled_shape_codes = np.zeros((samples, shape_code_library.shape[0]))
    for i in range(samples):
        shape_coeffs = np.random.uniform(shape_code_library.shape[1], 1, size=shape_code_library.shape[1])
        shape_coeffs = shape_coeffs / np.sum(shape_coeffs)
        sampled_shape_codes[i, :] = (shape_code_library @ shape_coeffs.reshape((-1, 1))).squeeze()
    return sampled_shape_codes


def main(opt):
    LF.seed_everything(42)
    TEST_MODE = TestMode.DUMP_NOCS_AND_RECONS_MESH
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    print("Generating ground truth latent codes for YCB-Video objects")

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
    model_name = opt.checkpoint_path.split("/")[-2]
    model.load_state_dict(state["model"])
    model = model.cuda()

    # dataloaders
    shape_ds = BOPNOCSDataset(
        ds_name="ycbv",
        split="test",
        bop_ds_dir=opt.bop_data_dir,
        unified_objects_dataset_path=opt.dataset_dir,
        preload_to_mem=opt.preload_to_mem,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        sample_bounds=(-1.0, 1.0),
        pc_size=opt.pc_size,
        input_H=opt.backbone_input_res[0],
        input_W=opt.backbone_input_res[1],
        normalized_recons=opt.normalized_recons,
        debug_vis=opt.dataset_debug_vis,
    )
    shape_ds.data_to_output.append("metadata")
    shape_ds.data_to_output.append("frame_info")
    dl = tchdata.DataLoader(shape_ds, shuffle=True, num_workers=2, batch_size=opt.batch_size)

    # load synthetic dataset
    # go through dataset and log latent code <-> obj name
    # tsne projection
    # robust centroid and save to file
    export_folder = f"./exports/ycbv/nocs_mesh_dump_{model_name}/"
    for i, batch in tqdm(enumerate(dl), total=len(dl)):
        # nocs related
        rgb_img, segmap, gt_nocs, coords, gt_normals, gt_sdf, metadata, frame_info = (
            batch["rgb"].cuda(),
            batch["instance_segmap"].cuda(),
            batch["nocs"].cuda(),
            batch["coords"].cuda(),
            batch["normals"].cuda(),
            batch["sdf"].cuda(),
            batch["metadata"],
            batch["frame_info"],
        )
        bs = rgb_img.shape[0]

        # the mask should be already for the correct object
        mask = segmap.unsqueeze(1)

        # run the corrector which takes the SDF,
        # perturbed GT NOCS and depths and give us corrected NOCS, and calculate registration.
        nocs_map, shape_code = model.forward_nocs_and_shape_code(img=rgb_img, mask=mask)

        # save nocs
        nocs_heatmap_names = []
        for i in range(bs):
            label = metadata["label"][i]
            name = f"nocs_{frame_info['frame_id'][i]}_{frame_info['view_id'][i]}_{label}.jpg"
            nocs_heatmap_names.append(
                os.path.join(export_folder, str(int(frame_info["scene_id"][i].item())), str(name))
            )
        save_joint_visualizations(rgb_img, nocs_map, mask, nocs_heatmap_names)

        # save mesh
        mesh_screenshot_names = []
        for i in range(bs):
            label = metadata["label"][i]
            name = f"mesh_{frame_info['frame_id'][i]}_{frame_info['view_id'][i]}_{label}.jpg"
            mesh_screenshot_names.append(os.path.join(export_folder, frame_info["scene_id"][i].item(), name))
        save_screenshots_from_shape_codes(model, shape_code, mesh_screenshot_names)

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
            f"./exports/simplex_sample_{model_name}/sample_{j}.png" for j in range(simplex_sampled_shape_codes.shape[0])
        ]
        save_screenshots_from_shape_codes(model, simplex_sampled_shape_codes, mesh_screenshot_names)

    else:
        if TEST_MODE == TestMode.EXTRAPOLATE_BETWEEN_SPECIFIED_OBJS:
            print("Extrapolating between two specified objects.")
            o1 = EXTRAPOLATE_OBJ_1
            o2 = EXTRAPOLATE_OBJ_2
            shp_code_1, shp_code_2 = random.choice(raw_obj_code_db[o1]), random.choice(raw_obj_code_db[o2])
            alphas = np.array([-1.5, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5])
            interp_shp_codes, _ = extrapolate_shape_code(shp_code_1, shp_code_2, x=alphas)
            prefix = "extrapolate_specific"
            mesh_screenshot_names = [
                f"./exports/ycbv/{prefix}_{model_name}/{o1}_{o2}/{o1}_{o2}_alpha_{a}.png" for a in alphas
            ]
            save_screenshots_from_shape_codes(model, interp_shp_codes, mesh_screenshot_names)

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
                    f"./exports/{prefix}_{model_name}/{o1}_{o2}/{o1}_{o2}_step_{j}.png"
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
