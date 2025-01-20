import pickle
import trimesh
import csv
import yaml
import os
import numpy as np
import sys
import json
from jsonargparse import ArgumentParser
from dataclasses import dataclass, field

from crisp.utils.file_utils import safely_make_folders

# fmt: off
blender_R_cad = np.array([[1.0, 0, 0],
                          [0, 1.0, 0],
                          [0, 0, 1.0]])
cad_R_blender = blender_R_cad.T
# fmt: on


@dataclass
class NOCSGenSettings:
    nocs_path: str = None
    # subsets options: ["camera_train", "camera_val", "real_train", "real_test"]
    subset_with_splits: list = field(default_factory=lambda: ["camera_train"])
    output_dir: str = "./output"


def load_norms_file(filepath):
    """Load norms from a .txt file where norms are given in three separate lines."""
    with open(filepath, "r") as file:
        lines = file.readlines()
        if len(lines) < 3:
            raise ValueError(f"File {filepath} does not contain exactly 3 lines.")
        x = float(lines[0].strip())
        y = float(lines[1].strip())
        z = float(lines[2].strip())
        return (x, y, z)


def gen_real_subset_objects_info(folder_path):
    """Create yaml file for REAL subset objects"""
    files = os.listdir(folder_path)
    obj_files = [f for f in files if f.endswith(".obj")]

    objects_yaml_data = {}
    for obj_file in sorted(obj_files):
        base_name = obj_file[:-4]  # Remove the .obj extension
        txt_filepath = os.path.join(folder_path, base_name + ".txt")

        norms = load_norms_file(txt_filepath)
        objects_yaml_data[base_name] = {
            "blender_R_cad": blender_R_cad.tolist(),
            "blender_s_cad": [1.0, 1.0, 1.0],
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_s_blender": np.array(2 / np.linalg.norm(np.array(norms))).tolist(),
            "recons_t_blender": [0.0, 0.0, 0.0],
        }
    return objects_yaml_data


def gen_camera_subset_objects_info(dataset_name, nocs_path):
    """Create yaml file for CAMERA subset objects"""
    obj_folder_path = os.path.join(nocs_path, "obj_models", dataset_name)
    objects_data = load_camera_subset_objects(obj_folder_path)

    split = dataset_name.split("_")[-1]
    obj_index_path = os.path.join(nocs_path, "gt_results", "camera", split, "gt_results.pkl")
    assert os.path.exists(obj_index_path), f"Object index path not found: {obj_index_path}"
    object_index = pickle.load(open(obj_index_path, "rb"))
    # we use median scales for the camera objects
    scale_index = object_index["object_median_scales"]

    yaml_data = load_camera_subset_scales(objects_data, scale_index)
    return yaml_data


def load_camera_subset_scales(obj_data, scale_index):
    objects_yaml_data = {}

    for obj_name, obj_info in obj_data.items():
        # generate an entry for each scale
        if obj_name not in scale_index:
            # these are the distractor objects
            continue

        scale = scale_index[obj_name]

        # for NOCS camera objects, the assumption is
        # that the NOCS and the CAD objects are the same scale
        #
        # the scale we have loaded in the scale index is the scaling between NOCS and depth:
        # transformed_pts = s * gt_R @ coord_pts.T + gt_t
        #
        # since we assume the same scale between NOCS and CAD, we have
        # coord_pts.T = non_metric_cad_pts
        # transformed_pts = s * gt_R @ non_metric_cad_pts.T + gt_t
        # transformed_pts = gt_R @ (s * non_metric_cad_pts.T) + gt_t
        # transformed_pts = gt_R @ metric_cad_pts.T + gt_t
        # so
        # metric_cad_pts = s * non_metric_cad_pts
        objects_yaml_data[obj_name] = {
            "cad_s_nonmetric_cad": float(scale),
            "blender_R_cad": blender_R_cad.tolist(),
            "blender_s_cad": [1.0, 1.0, 1.0],
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_s_blender": float(2 / scale),
            "recons_t_blender": [0.0, 0.0, 0.0],
        }

    return objects_yaml_data


def load_camera_subset_objects(folder_path):
    """Create yaml file for CAMERA subset objects
    Steps:
    1. load all objects
    2. read through all files

    folder_path should point to
    """
    # load all objects
    objects_data = {}
    for category in sorted(os.listdir(folder_path)):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            # Loop through each instance directory in the category directory
            for instance in os.listdir(category_path):
                instance_path = os.path.join(category_path, instance)
                if os.path.isdir(instance_path):
                    # Get the path to the mesh file
                    obj_name = f"{category}_{instance}"
                    mesh_file_path = os.path.join(instance_path, "model.obj")
                    bbox_path = os.path.join(instance_path, "bbox.txt")
                    bbox = np.loadtxt(bbox_path)
                    scale = bbox[0, :] - bbox[1, :]

                    if os.path.exists(mesh_file_path):
                        objects_data[obj_name] = {"mesh_path": mesh_file_path, "scale": scale}
                    else:
                        raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

    return objects_data


def main(opt: NOCSGenSettings):
    # prepare and load NOCS dataset
    allowed_subsets = ["camera_train", "camera_val", "real_train", "real_test"]
    assert set(opt.subset_with_splits) <= set(allowed_subsets)
    obj_models_path = os.path.join(opt.nocs_path, "obj_models")

    all_yaml_data = {}
    for subset in opt.subset_with_splits:
        print(f"Generating objects_info.yml for {subset}")
        if subset == "real_train":
            yaml_data = gen_real_subset_objects_info(os.path.join(obj_models_path, "real_train"))
            all_yaml_data["nocs_real_train"] = yaml_data
        elif subset == "real_test":
            yaml_data = gen_real_subset_objects_info(os.path.join(obj_models_path, "real_test"))
            all_yaml_data["nocs_real_test"] = yaml_data
        elif subset == "camera_train" or subset == "camera_val":
            yaml_data = gen_camera_subset_objects_info(subset, opt.nocs_path)
            all_yaml_data[f"nocs_{subset}"] = yaml_data

    safely_make_folders([opt.output_dir])
    with open(os.path.join(opt.output_dir, "objects_info.yaml"), "w") as objects_yaml:
        yaml.dump(all_yaml_data, objects_yaml)


if __name__ == "__main__":
    """Generate stub datasets objects_info.yml for all the objects in the NOCS
      datasets.
      It will generate the following objects_info.yml files:
      - For real set only
        - train objects
    - test objects
      - For camera set only
        - train objects
        - test objects
      - For real + camera objects
    """
    parser = ArgumentParser()
    parser.add_class_arguments(NOCSGenSettings)
    opt = parser.parse_args()
    main(NOCSGenSettings(**opt))
