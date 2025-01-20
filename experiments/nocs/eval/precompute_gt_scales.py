import trimesh
import pickle
from collections import defaultdict
import h5py
from tqdm import tqdm
import json
import numpy as np
from jsonargparse import ArgumentParser
from dataclasses import dataclass, field
import datetime
import os
from pathlib import Path
import time

import torch
from crisp.datasets import nocs, nocs_config
from crisp.datasets.nocs_utils import backproject, align
from crisp.utils.math import depth_to_point_cloud
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.visualization_utils import visualize_meshes_pyvista
from crisp.models.registration import umeyama, umeyama_ransac, umeyama_ransac_batched


@dataclass
class ExpSettings:
    dataset_dir: str = "./data/NOCS"

    # subsets options: ["camera", "real"]
    subsets: list = field(default_factory=lambda: ["camera"])
    split: str = "train"

    gt_results_dump_dir: str = "./data/gt_results"
    visualize: bool = False

    dry_run: bool = True


def align_gt(
    class_ids,
    masks,
    coords,
    depth,
    intrinsics,
    synset_names,
    image_path,
    if_norm=False,
):
    num_instances = len(class_ids)
    error_messages = ""
    elapses = []
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses

    rotations, translations, scales, best_inlier_ratios = [], [], [], []
    bbox_scales = np.ones((num_instances, 3))

    for i in range(num_instances):
        class_id = class_ids[i]
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        abs_coord_pts = np.abs(coord[mask == 1] - 0.5)
        bbox_scales[i, :] = 2 * np.amax(abs_coord_pts, axis=0)

        pts, idxs = backproject(depth, intrinsics, mask)
        if pts.shape[0] == 0:
            rotations.append(None)
            translations.append(None)
            scales.append(None)
            continue

        coord_pts = coord[idxs[0], idxs[1], :] - 0.5

        if if_norm:
            scale = np.linalg.norm(bbox_scales[i, :])
            bbox_scales[i, :] /= scale
            coord_pts /= scale

        # try:
        scale, R, t, T, best_inlier_ratio = umeyama_ransac(
            torch.tensor(coord_pts.T).float(), torch.tensor(pts.T).float() / 1000.0
        )
        R, t, scale = R.numpy(), t.numpy(), scale.item()
        # except Exception as e:
        #    message = "[ Error ] aligning instance {} in {} fails. Message: {}.".format(
        #        synset_names[class_id], image_path, str(e)
        #    )
        #    print(message)
        #    error_messages += message + "\n"
        #    R = np.identity(3, dtype=np.float32)
        #    t = np.zeros(3, dtype=np.float32)
        #    scale = 1.0

        rotations.append(R)
        translations.append(t)
        scales.append(scale)
        best_inlier_ratios.append(best_inlier_ratio)

    return scales, rotations, translations, best_inlier_ratios


def eval_gt(opt, subset, dataset: nocs.Dataset, intrinsics, visualize=False):
    """Iterate through all the samples, generate gt scales, rotations and translations"""
    image_ids = dataset.image_ids
    save_dir = os.path.join(opt.gt_results_dump_dir, subset, opt.split)
    safely_make_folders(os.path.join(save_dir))
    gt_dump_path = os.path.join(save_dir, "gt_results.pkl")

    # We need the following values:
    # GT and image info indexed by image_folder/image_name (within the subset and split)
    # a dictionary indexed by the object_id with values equal to all the scales that particular object has appeared in
    output_data = {"object_scale_index": defaultdict(list), "gt_results": {}, "gt_results_per_image_inst": {}}
    total_images, skipped_images = 0, 0
    print(f"Generating ground truth results for a total of {len(image_ids)} images.")
    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        total_images += 1
        image_path = dataset.image_info[image_id]["path"]

        result = {}
        # Calculate GT RT and scale
        # https://github.com/hughw19/NOCS_CVPR2019/blob/dd58dbf68feede04c3d7bbafeb9212af1a43422f/detect_eval.py#L171

        # loading ground truth
        image = dataset.load_image(image_id)
        depth = dataset.load_depth(image_id)

        # gt_scales are (N, 3)
        gt_inst_mask, gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label, _ = dataset.load_mask(image_id)

        if len(gt_class_ids) == 0:
            skipped_images += 1
            continue

        inst_dict, inst2mesh_dict = (
            dataset.image_info[image_id]["inst_dict"],
            dataset.image_info[image_id]["inst2mesh_dict"],
        )

        # real set and camera subsets use different object conventions
        sorted_inst_ids = sorted(inst_dict.keys())
        if subset == "real":
            mesh_names = [Path(inst2mesh_dict[inst_id]).parts[-1].split(".")[0] for inst_id in sorted_inst_ids]
        elif subset == "camera":

            def mname_func(inst_id):
                pparts = Path(inst2mesh_dict[inst_id]).parts
                return f"{pparts[-3]}_{pparts[-2]}"

            mesh_names = [mname_func(inst_id) for inst_id in sorted_inst_ids]
        else:
            raise ValueError(f"Unknown subset: {subset}")

        result["image_id"] = image_id
        result["image_path"] = image_path
        result["gt_class_ids"] = gt_class_ids
        result["gt_scales"] = gt_scales

        gt_s, gt_R, gt_t, gt_inlier_ratios = align_gt(
            gt_class_ids,
            gt_mask,
            gt_coord,
            depth,
            intrinsics,
            dataset.synset_names,
            image_path,
        )
        result["gt_trans_s"] = gt_s
        result["gt_trans_R"] = gt_R
        result["gt_trans_t"] = gt_t

        # visualize
        if visualize:
            print(f"Visualizing {image_path} point clouds.")
            print("Depth: blue, gt transformed: red")
            num_instances = len(gt_class_ids)
            instance_ids = sorted(inst_dict.keys())
            for jj in range(num_instances):
                if gt_s[jj] is None:
                    continue
                mask = gt_mask[:, :, jj]
                coord = gt_coord[:, :, jj, :]
                depths_pts, idxs = backproject(depth, intrinsics, mask)
                coord_pts = coord[idxs[0], idxs[1], :] - 0.5
                gt_transformed_pts = gt_s[jj] * gt_R[jj] @ coord_pts.T + gt_t[jj].reshape(3, 1)

                visualize_meshes_pyvista(
                    meshes=[depths_pts / 1000.0, gt_transformed_pts],
                    plotter_shape=(1, 1),
                    mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                )

                print(f"Compare CAD model (scaled by gt scale) with NOCS points.")
                print(f"The two should be overlapping with each other.")
                print(f"NOCS points: blue, CAD model: red")
                mesh_path = inst2mesh_dict[instance_ids[jj]]
                mesh = trimesh.load(mesh_path, force="mesh", skip_materials=True)
                visualize_meshes_pyvista(
                    meshes=[coord_pts, mesh],
                    plotter_shape=(1, 1),
                    mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                )

        # save the result
        image_path = Path(dataset.image_info[image_id]["path"])
        local_img_dir, image_name = image_path.parts[-2], image_path.parts[-1]

        output_data["gt_results"][local_img_dir + "/" + image_name] = []
        for mi, mname in enumerate(mesh_names):
            if gt_s[mi] is None or gt_R[mi] is None or gt_t[mi] is None:
                continue
            output_data["object_scale_index"][mname].append(gt_s[mi])
            output_data["gt_results"][local_img_dir + "/" + image_name].append(
                {
                    "mesh_name": mname,
                    "gt_s": gt_s[mi],
                    "gt_R": gt_R[mi],
                    "gt_t": gt_t[mi],
                    "gt_best_inlier_ratio": gt_inlier_ratios[mi],
                }
            )

            inst_id = sorted_inst_ids[mi]
            output_data["gt_results_per_image_inst"][f"{image_id}_{inst_id}"] = {
                "image_id": image_id,
                "inst_id": inst_id,
                "mesh_name": mname,
                "gt_s": gt_s[mi],
                "gt_R": gt_R[mi],
                "gt_t": gt_t[mi],
                "gt_best_inlier_ratio": gt_inlier_ratios[mi],
            }

        if i % 1000 == 0:
            with open(gt_dump_path, "wb") as f:
                pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)

    print(f"Total images: {total_images}, skipped images: {skipped_images}")
    print(f"Dry run status: {opt.dry_run}")
    if not opt.dry_run:
        print("Saving results...")
        with open(gt_dump_path, "wb") as f:
            pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)

    return


def main(opt: ExpSettings):
    """Iterate through all the samples, generate gt scales"""
    # prepare and load NOCS dataset
    allowed_subsets = ["camera", "real"]
    assert set(opt.subsets) <= set(allowed_subsets)

    config = nocs_config.Config()
    config.ROOT_DIR = opt.dataset_dir
    config.OBJ_MODEL_DIR = os.path.join(config.ROOT_DIR, "obj_models")

    for subset in opt.subsets:
        camera_dir = os.path.join(opt.dataset_dir, "camera")
        real_dir = os.path.join(opt.dataset_dir, "real")
        obj_dir = os.path.join(opt.dataset_dir, "obj_models")

        dataset = nocs.Dataset(subset=opt.split, make_yaml=False, config=config)
        # https://github.com/hughw19/NOCS_CVPR2019/blob/dd58dbf68feede04c3d7bbafeb9212af1a43422f/detect_eval.py#L171
        intrinsics = None
        if subset == "camera":
            dataset.load_camera_scenes(camera_dir, obj_dir)
            intrinsics = np.array([[577.5, 0, 319.5], [0.0, 577.5, 239.5], [0.0, 0.0, 1.0]])
        elif subset == "real":
            dataset.load_real_scenes(real_dir, obj_dir)
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        class_map = {
            "bottle": "bottle",
            "bowl": "bowl",
            "cup": "mug",
            "laptop": "laptop",
        }
        dataset.prepare(class_map)
        eval_gt(opt, subset, dataset, intrinsics, visualize=opt.visualize)


if __name__ == "__main__":
    """NOCS dataset experiment"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()
    main(ExpSettings(**opt))
