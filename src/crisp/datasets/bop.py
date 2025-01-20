import os.path

import copy
import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import pickle
import time
from tqdm import tqdm
import trimesh
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as vF
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, default_convert
from torchvision.utils import draw_segmentation_masks

from bop_toolkit_lib import inout

# from cosypose.lib3d import Transform
# from crisp.datasets.object_pose.bop_constants import BOP_MODEL_INDICES
from crisp.utils.general import pos_tensor_to_o3d
from crisp.utils.visualization_utils import visualize_model_n_keypoints, visualize_point_cloud_and_mesh

BOP_DS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bop" / "bop_datasets"
BOP_DS_DIR = BOP_DS_DIR.resolve()

BOP_NUM_OBJECTS = {"ycbv": 21, "tless": 30}

"""
Constants
"""

BOP_CATEGORIES = {"ycbv": 21, "tless": 30}

BOP_SPLIT_DIRS = {
    "ycbv": ["train_real", "train_synt", "train_pbr", "test"],
    "tless": ["train_primesense", "test_primesense", "train_pbr"],
}

YCBV = {
    "obj_000001": "002_master_chef_can",
    "obj_000002": "003_cracker_box",
    "obj_000003": "004_sugar_box",
    "obj_000004": "005_tomato_soup_can",
    "obj_000005": "006_mustard_bottle",
    "obj_000006": "007_tuna_fish_can",
    "obj_000007": "008_pudding_box",
    "obj_000008": "009_gelatin_box",
    "obj_000009": "010_potted_meat_can",
    "obj_000010": "011_banana",
    "obj_000011": "019_pitcher_base",
    "obj_000012": "021_bleach_cleanser",
    "obj_000013": "024_bowl",
    "obj_000014": "025_mug",
    "obj_000015": "035_power_drill",
    "obj_000016": "036_wood_block",
    "obj_000017": "037_scissors",
    "obj_000018": "040_large_marker",
    "obj_000019": "051_large_clamp",
    "obj_000020": "052_extra_large_clamp",
    "obj_000021": "061_foam_brick",
}

TLESS = {
    "obj_000001": "obj_000001",
    "obj_000002": "obj_000002",
    "obj_000003": "obj_000003",
    "obj_000004": "obj_000004",
    "obj_000005": "obj_000005",
    "obj_000006": "obj_000006",
    "obj_000007": "obj_000007",
    "obj_000008": "obj_000008",
    "obj_000009": "obj_000009",
    "obj_000010": "obj_000010",
    "obj_000011": "obj_000011",
    "obj_000012": "obj_000012",
    "obj_000013": "obj_000013",
    "obj_000014": "obj_000014",
    "obj_000015": "obj_000015",
    "obj_000016": "obj_000016",
    "obj_000017": "obj_000017",
    "obj_000018": "obj_000018",
    "obj_000019": "obj_000019",
    "obj_000020": "obj_000020",
    "obj_000021": "obj_000021",
    "obj_000022": "obj_000022",
    "obj_000023": "obj_000023",
    "obj_000024": "obj_000024",
    "obj_000025": "obj_000025",
    "obj_000026": "obj_000026",
    "obj_000027": "obj_000027",
    "obj_000028": "obj_000028",
    "obj_000029": "obj_000029",
    "obj_000030": "obj_000030",
}

BOP_MODEL_INDICES = dict(ycbv=YCBV, tless=TLESS)

BOP_LABEL_TO_CATEGORY_ID = dict(
    ycbv={
        "background": 0,
        "obj_000001": 1,
        "obj_000002": 2,
        "obj_000003": 3,
        "obj_000004": 4,
        "obj_000005": 5,
        "obj_000006": 6,
        "obj_000007": 7,
        "obj_000008": 8,
        "obj_000009": 9,
        "obj_000010": 10,
        "obj_000011": 11,
        "obj_000012": 12,
        "obj_000013": 13,
        "obj_000014": 14,
        "obj_000015": 15,
        "obj_000016": 16,
        "obj_000017": 17,
        "obj_000018": 18,
        "obj_000019": 19,
        "obj_000020": 20,
        "obj_000021": 21,
    },
    tless={
        "background": 0,
        "obj_000001": 1,
        "obj_000002": 2,
        "obj_000003": 3,
        "obj_000004": 4,
        "obj_000005": 5,
        "obj_000006": 6,
        "obj_000007": 7,
        "obj_000008": 8,
        "obj_000009": 9,
        "obj_000010": 10,
        "obj_000011": 11,
        "obj_000012": 12,
        "obj_000013": 13,
        "obj_000014": 14,
        "obj_000015": 15,
        "obj_000016": 16,
        "obj_000017": 17,
        "obj_000018": 18,
        "obj_000019": 19,
        "obj_000020": 20,
        "obj_000021": 21,
        "obj_000022": 22,
        "obj_000023": 23,
        "obj_000024": 24,
        "obj_000025": 25,
        "obj_000026": 26,
        "obj_000027": 27,
        "obj_000028": 28,
        "obj_000029": 29,
        "obj_000030": 30,
    },
)

"""
Utils to be replaced / moved
"""


def get_transform(R, t):
    # breakpoint()
    T = np.concatenate((R, t.reshape(3, 1)), axis=1)
    e = np.concatenate((np.zeros((1, 3)), np.ones((1, 1))), axis=1)
    T = np.concatenate((T, e), axis=0)

    return T


def inv_transform(T):
    R = T[:3, :3]
    t = T[:3, 3:]
    Rinv = R.transpose()
    tinv = -Rinv @ t

    return get_transform(Rinv, tinv)


"""
Core 
"""


def is_bop_obj_det_valid(entry, min_visib_fract=0.002, min_px_count_valid=10, min_px_count_visib=10, min_area=None):
    """Determine whether a given object dictionary in BOP format is a valid detection or not"""
    add = True
    # only keep visible objects
    if entry["visib_fract"] <= min_visib_fract:
        add = False

    # not enough valid pixels
    if entry["px_count_valid"] < min_px_count_valid:
        add = False

    # not enough visible pixels
    if entry["px_count_visib"] < min_px_count_visib:
        add = False

    # area check
    if add and min_area is not None:
        if entry["visib_area"] < min_area:
            add = False
    return add


def remap_bop_targets(targets):
    targets = targets.rename(columns={"im_id": "view_id"})
    targets["label"] = targets["obj_id"].apply(lambda x: f"obj_{x:06d}")
    return targets


def _make_tless_dataset(split, bop_ds_dir=None, load_depth=False):
    ds_dir = bop_ds_dir / "tless"
    ds = BOPDataset(ds_dir, split=split, load_depth=load_depth)
    return ds


def keep_bop19(ds):
    targets = pd.read_json(ds.ds_dir / "test_targets_bop19.json")
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ["scene_id", "view_id"]].drop_duplicates()
    index = ds.frame_index.merge(targets, on=["scene_id", "view_id"]).reset_index(drop=True)
    #assert len(index) == len(targets)
    ds.frame_index = index
    return ds


def make_scene_dataset(ds_name, n_frames=None, bop_ds_dir=None, load_depth=False):
    """
    Credit: https://github.com/ylabbe/cosypose
    """
    # TLESS
    if ds_name == "tless.primesense.train":
        ds = _make_tless_dataset("train_primesense", bop_ds_dir=bop_ds_dir, load_depth=load_depth)

    elif ds_name == "tless.primesense.test":
        ds = _make_tless_dataset("test_primesense", bop_ds_dir=bop_ds_dir, load_depth=load_depth)

    elif ds_name == "tless.primesense.test.bop19":
        ds = _make_tless_dataset("test_primesense", bop_ds_dir=bop_ds_dir, load_depth=load_depth)
        ds = keep_bop19(ds)

    # YCBV
    elif ds_name == "ycbv.train.real":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="train_real", load_depth=load_depth)

    elif ds_name == "ycbv.train.synt":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="train_synt", load_depth=load_depth)

    elif ds_name == "ycbv.test":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)

    elif ds_name == "ycbv.test.keyframes":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        keyframes_path = ds_dir / "keyframe.txt"
        ls = keyframes_path.read_text().split("\n")[:-1]
        frame_index = ds.frame_index
        ids = []
        for l_n in ls:
            scene_id, view_id = l_n.split("/")
            scene_id, view_id = int(scene_id), int(view_id)
            mask = (frame_index["scene_id"] == scene_id) & (frame_index["view_id"] == view_id)
            ids.append(np.where(mask)[0].item())
        ds.frame_index = frame_index.iloc[ids].reset_index(drop=True)

    # BOP challenge
    elif ds_name == "hb.bop19":
        ds_dir = bop_ds_dir / "hb"
        ds = BOPDataset(ds_dir, split="test_primesense", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "icbin.bop19":
        ds_dir = bop_ds_dir / "icbin"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "itodd.bop19":
        ds_dir = bop_ds_dir / "itodd"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "lmo.bop19":
        ds_dir = bop_ds_dir / "lmo"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "tless.bop19":
        ds_dir = bop_ds_dir / "tless"
        ds = BOPDataset(ds_dir, split="test_primesense", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "tudl.bop19":
        ds_dir = bop_ds_dir / "tudl"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        ds = keep_bop19(ds)
    elif ds_name == "ycbv.bop19":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="test", load_depth=load_depth)
        ds = keep_bop19(ds)

    elif ds_name == "hb.pbr":
        ds_dir = bop_ds_dir / "hb"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "icbin.pbr":
        ds_dir = bop_ds_dir / "icbin"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "itodd.pbr":
        ds_dir = bop_ds_dir / "itodd"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "lm.pbr":
        ds_dir = bop_ds_dir / "lm"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "tless.pbr":
        ds_dir = bop_ds_dir / "tless"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "tudl.pbr":
        ds_dir = bop_ds_dir / "tudl"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)
    elif ds_name == "ycbv.pbr":
        ds_dir = bop_ds_dir / "ycbv"
        ds = BOPDataset(ds_dir, split="train_pbr", load_depth=load_depth)

    elif ds_name == "hb.val":
        ds_dir = bop_ds_dir / "hb"
        ds = BOPDataset(ds_dir, split="val_primesense", load_depth=load_depth)
    elif ds_name == "itodd.val":
        ds_dir = bop_ds_dir / "itodd"
        ds = BOPDataset(ds_dir, split="val", load_depth=load_depth)
    elif ds_name == "tudl.train.real":
        ds_dir = bop_ds_dir / "tudl"
        ds = BOPDataset(ds_dir, split="train_real", load_depth=load_depth)

    # Synthetic datasets
    elif "synthetic." in ds_name:
        raise NotImplementedError

    else:
        raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_object_dataset(ds_name, bop_ds_dir=None):
    """Helper function for generating a BOP object dataset
    Credit: https://github.com/ylabbe/cosypose
    """
    ds = None
    if ds_name == "tless.cad":
        ds = BOPObjectDataset(bop_ds_dir / "tless/models_cad")
    elif ds_name == "tless.eval" or ds_name == "tless.bop":
        ds = BOPObjectDataset(bop_ds_dir / "tless/models_eval")

    # YCBV
    elif ds_name == "ycbv.bop":
        ds = BOPObjectDataset(bop_ds_dir / "ycbv/models")
    elif ds_name == "ycbv.bop-compat":
        # BOP meshes (with their offsets) and symmetries
        # Replace symmetries of objects not considered symmetric in PoseCNN
        ds = BOPObjectDataset(bop_ds_dir / "ycbv/models_bop-compat")
    elif ds_name == "ycbv.bop-compat.eval":
        # PoseCNN eval meshes and symmetries, WITH bop offsets
        ds = BOPObjectDataset(bop_ds_dir / "ycbv/models_bop-compat_eval")

    # Other BOP
    elif ds_name == "hb":
        ds = BOPObjectDataset(bop_ds_dir / "hb/models")
    elif ds_name == "icbin":
        ds = BOPObjectDataset(bop_ds_dir / "icbin/models")
    elif ds_name == "itodd":
        ds = BOPObjectDataset(bop_ds_dir / "itodd/models")
    elif ds_name == "lm":
        ds = BOPObjectDataset(bop_ds_dir / "lm/models")
    elif ds_name == "tudl":
        ds = BOPObjectDataset(bop_ds_dir / "tudl/models")

    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name, local_data_dir=None, asset_dir=None):
    """Helper function for making a URDF dataset

    Credit: https://github.com/ylabbe/cosypose
    """
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset

    # BOP
    if ds_name == "tless.cad":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "tless.cad")
    elif ds_name == "tless.reconst":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "tless.reconst")
    elif ds_name == "ycbv":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "ycbv")
    elif ds_name == "hb":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "hb")
    elif ds_name == "icbin":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "icbin")
    elif ds_name == "itodd":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "itodd")
    elif ds_name == "lm":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "lm")
    elif ds_name == "tudl":
        ds = BOPUrdfDataset(local_data_dir / "urdfs" / "tudl")

    # Custom scenario
    elif "custom" in ds_name:
        scenario = ds_name.split(".")[1]
        ds = BOPUrdfDataset(local_data_dir / "scenarios" / scenario / "urdfs")

    elif ds_name == "camera":
        ds = OneUrdfDataset(asset_dir / "camera/model.urdf", "camera")
    else:
        raise ValueError(ds_name)
    return ds


def build_index(ds_dir, save_file, split, all_objects_labels, save_file_annotations, save_file_obj_index):
    """Build indices for faster accessing/search
    Edited to also build object-to-(scene_id, view_id) index
    Credit: cosypose
    """
    scene_ids, cam_ids, view_ids = [], [], []

    annotations = dict()
    base_dir = ds_dir / split

    for scene_dir in tqdm(base_dir.iterdir()):
        scene_id = scene_dir.name
        annotations_scene = dict()
        for f in ("scene_camera.json", "scene_gt_info.json", "scene_gt.json"):
            path = scene_dir / f
            if path.exists():
                annotations_scene[f.split(".")[0]] = json.loads(path.read_text())
        annotations[scene_id] = annotations_scene
        # for view_id in annotations_scene['scene_gt_info'].keys():
        for view_id in annotations_scene["scene_camera"].keys():
            cam_id = "cam"
            scene_ids.append(int(scene_id))
            cam_ids.append(cam_id)
            view_ids.append(int(view_id))

    # save dataset frame index
    frame_index = pd.DataFrame({"scene_id": scene_ids, "cam_id": cam_ids, "view_id": view_ids, "cam_name": cam_ids})
    # Note: sort to ensure consistencies across machines
    frame_index.sort_values("view_id", kind="stable", inplace=True)
    frame_index.sort_values("scene_id", kind="stable", inplace=True)
    frame_index.reset_index(drop=True, inplace=True)
    frame_index.to_feather(save_file)
    save_file_annotations.write_bytes(pickle.dumps(annotations))

    # build the object / images indices
    # this is built after the dataframe index to keep the index consistent across them
    # the assumption is: the order of objects in a particular image stays consistent across
    # json files loaded from the dataset
    object_image_index = {k: [] for k in sorted(all_objects_labels)}

    for i, (s_id, v_id) in enumerate(zip(frame_index["scene_id"], frame_index["view_id"])):
        s_id_str = f"{s_id:06d}"
        objs_data = annotations[s_id_str]["scene_gt"][str(v_id)]
        visib = annotations[s_id_str]["scene_gt_info"][str(v_id)]
        for j in range(len(objs_data)):
            o = objs_data[j]
            obj_label = f"obj_{o['obj_id']:06d}"
            bbox_visib = np.array(visib[j]["bbox_visib"])
            _, _, w, h = bbox_visib
            payload = dict(
                index_row=i,
                scene_id=s_id,
                view_id=v_id,
                obj_index_in_view=j,
                visib_area=w * h,
                visib_fract=visib[j]["visib_fract"],
                px_count_all=visib[j]["px_count_all"],
                px_count_valid=visib[j]["px_count_valid"],
                px_count_visib=visib[j]["px_count_visib"],
            )
            object_image_index[obj_label].append(payload)

    # save dataset object images index
    save_file_obj_index.write_bytes(pickle.dumps(object_image_index))

    return


def build_object_images_index(ds_dir, split, all_objects_labels, frame_index, save_file=None):
    """Build an index from object IDs to the indices

    Returns:
        dict: keys=object labels; items=tuple containing (scene_id, view_id)
    """
    object_image_index = {k: [] for k in all_objects_labels}

    base_dir = ds_dir / split
    for scene_dir in base_dir.iterdir():
        scene_id = scene_dir.name
        path = scene_dir / "scene_gt.json"
        obj_annotations = json.loads(path.read_text())
        for view_id in obj_annotations.keys():
            objs_data = obj_annotations[view_id]
            for o in objs_data:
                obj_label = f"obj_{o['obj_id']:06d}"
                object_image_index[obj_label].append((int(scene_id), int(view_id)))

    if save_file is not None:
        file = open(save_file, "wb")
        pickle.dump(object_image_index, file)
        file.close()

    return object_image_index


def get_cad_model_mesh(object_label, ds_name, ds_dir=BOP_DS_DIR):
    """
    :param object_label: object label
    :param ds_name: bop dataset name
    :return: cad model mesh of the object
    """

    # setting up the objects dataset
    if ds_name.split(".")[0] == "ycbv":
        ds_objects = make_object_dataset(ds_name="ycbv.bop", bop_ds_dir=ds_dir)
    elif ds_name.split(".")[0] == "tless":
        ds_objects = make_object_dataset(ds_name="tless.cad", bop_ds_dir=ds_dir)

    # getting object id from the object_label (e.g. obj_000014 --> 14)
    obj_id = int(object_label.split("_")[1])
    cad_model_info = ds_objects[obj_id - 1]

    # reading the cad_model_mesh from the path
    _cad_model_mesh = o3d.io.read_triangle_model(cad_model_info["mesh_path"])
    cad_model_mesh = _cad_model_mesh.meshes[0].mesh
    cad_model_mesh = o3d.geometry.TriangleMesh(cad_model_mesh)

    return cad_model_mesh


def _get_depth_and_pose_from_single_batch(rgb, mask, obs, object_label, model=None, visualize_segmask=False):
    # re-formating rgb, mask to our format (wxhx3 --> 3xwxh)
    rgb = rgb.squeeze(0)
    mask = mask.squeeze(0)
    rgb = torch.stack([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])

    # extracting model outputs
    if model is not None:
        # ToDo: have to extract obs and mask from the pre-trained model, by feeding in rgb
        raise NotImplementedError

    # extracting labels.
    # boxes = [torch.stack(obs["objects"][i]["bbox"]).reshape(1, 4) for i in range(len(obs["objects"]))]
    # boxes = torch.stack(boxes).squeeze(1)
    labels = [obs["objects"][i]["label"][0] for i in range(len(obs["objects"]))]

    # checking if the object_label is in lables
    if object_label not in labels:
        print("Error: object_label not found in the scene and view id")
        return None, None, None, None

    else:
        # getting the object index for the mask
        obj_idx = labels.index(object_label)

        # extracting segmentation masks. putting them in a visualizable format for torchvision.
        observed_categories = torch.unique(mask)
        instance_masks = []
        for c in observed_categories:
            if c != 0:
                instance_masks.append((mask == c).to(dtype=torch.uint8))

        instance_masks = torch.stack(instance_masks)
        instance_masks = instance_masks.to(dtype=torch.bool)

        obj_mask = instance_masks[obj_idx, ...]

        # drawing segmentation mask
        if visualize_segmask:
            images_with_obj_mask = draw_segmentation_masks(rgb, masks=obj_mask, alpha=0.7)
            im = F.to_pil_image(images_with_obj_mask)
            plt.imshow(im)
            plt.show()

        # extracting fraction of object visible
        visible_fraction = [obs["objects"][i]["visib_fract"] for i in range(len(obs["objects"]))]
        visible_fraction = torch.stack(visible_fraction)
        object_visible_fraction = visible_fraction[obj_idx, ...]

        # extracting scene and object depth
        camera = obs["camera"]
        K = camera["K"].squeeze(0)
        # K[:2, :2] /= 1000
        T0C = camera["T0C"].squeeze(0)
        T0C[:3, 3] *= 1000
        TWC = camera["TWC"].squeeze(0)
        TWC[:3, 3] *= 1000
        res = camera["resolution"]

        u = torch.arange(res[1].item())
        u = u.repeat(res[0].item(), 1)
        v = torch.arange(res[0].item())
        v = v.repeat(res[1].item(), 1)
        v = v.T
        z = 1000 * camera["depth"].squeeze(0)

        ux = u.reshape(1, -1)
        vy = v.reshape(1, -1)
        zz = z.reshape(1, -1)
        ione = torch.ones(1, zz.shape[-1])
        obj_mask_pts = obj_mask.reshape(1, -1)
        img_coord = torch.concat([ux * zz, vy * zz, zz, ione], 0)

        _K = torch.eye(4)
        invK = torch.linalg.inv(K)

        _K[:3, :3] = invK
        invT0C = torch.linalg.inv(T0C)
        scene_depth_pc = (
            TWC.to(dtype=torch.double)
            @ invT0C.to(dtype=torch.double)
            @ _K.to(dtype=torch.double)
            @ img_coord.to(dtype=torch.double)
        )
        scene_depth_pc = scene_depth_pc[:3, :]
        object_depth_pc = scene_depth_pc * obj_mask_pts

        # extracting ground truth pose
        TWO = obs["objects"][obj_idx]["TWO"].squeeze(0)
        TWO[:3, 3] *= 1000
        # invTWO = torch.linalg.inv(TWO)
        # object_pose = invT0C @ invTWO
        object_pose = invT0C @ TWO
        object_pose = object_pose.to(device="cpu").detach().numpy()

        return scene_depth_pc, object_depth_pc, object_pose, object_visible_fraction


def get_batch_data(rgb, mask, obs, b_idx):
    """ """
    _rgb = rgb[b_idx, ...]
    _mask = mask[b_idx, ...]
    _obs = copy.deepcopy(obs)

    for idx in range(len(obs["objects"])):
        _obs["objects"][idx]["label"] = [obs["objects"][idx]["label"][b_idx]]
        _obs["objects"][idx]["name"] = [obs["objects"][idx]["name"][b_idx]]
        # breakpoint()
        _obs["objects"][idx]["TWO"] = obs["objects"][idx]["TWO"][b_idx, ...].unsqueeze(0)
        _obs["objects"][idx]["T0O"] = obs["objects"][idx]["T0O"][b_idx, ...].unsqueeze(0)
        _obs["objects"][idx]["visib_fract"] = obs["objects"][idx]["visib_fract"][b_idx].unsqueeze(0)
        _obs["objects"][idx]["id_in_segm"] = obs["objects"][idx]["id_in_segm"][b_idx].unsqueeze(0)
        _obs["objects"][idx]["bbox"] = [
            obs["objects"][idx]["bbox"][0][b_idx].unsqueeze(0),
            obs["objects"][idx]["bbox"][1][b_idx].unsqueeze(0),
            obs["objects"][idx]["bbox"][2][b_idx].unsqueeze(0),
            obs["objects"][idx]["bbox"][3][b_idx].unsqueeze(0),
        ]

    _obs["camera"]["T0C"] = obs["camera"]["T0C"][b_idx, ...].unsqueeze(0)
    _obs["camera"]["K"] = obs["camera"]["K"][b_idx, ...].unsqueeze(0)
    _obs["camera"]["TWC"] = obs["camera"]["TWC"][b_idx, ...].unsqueeze(0)
    # breakpoint()
    _obs["camera"]["resolution"] = [obs["camera"]["resolution"][0][b_idx], obs["camera"]["resolution"][1][b_idx]]
    _obs["camera"]["depth"] = obs["camera"]["depth"][b_idx, ...].unsqueeze(0)

    for akey in obs["frame_info"].keys():
        _obs["frame_info"][akey] = [obs["frame_info"][akey][b_idx]]

    return _rgb, _mask, _obs


def get_depth_and_pose(rgb, mask, obs, object_label, model=None, visualize_segmask=False):
    """ """
    b = rgb.shape[0]
    if b == 1:
        scene_depth_pc, object_depth_pc, object_pose, object_visible_fraction = _get_depth_and_pose_from_single_batch(
            rgb, mask, obs, object_label, model=model, visualize_segmask=visualize_segmask
        )

        scene_depth_pc = scene_depth_pc.unsqueeze(0)
        object_depth_pc = object_depth_pc.unsqueeze(0)
        object_pose = np.expand_dims(object_pose, 0)

    else:
        # ToDo: This does not work. Remains to be done for batch_size b > 1.
        scene_depth_pc = []
        object_depth_pc = []
        object_pose = []
        object_visible_fraction = []

        for b_idx in range(b):
            _rgb, _mask, _obs = get_batch_data(rgb, mask, obs, b_idx)
            # breakpoint()
            (
                _scene_depth_pc,
                _object_depth_pc,
                _object_pose,
                _object_visible_fraction,
            ) = _get_depth_and_pose_from_single_batch(
                _rgb, _mask, _obs, object_label, model=model, visualize_segmask=visualize_segmask
            )

            if _scene_depth_pc is None:
                continue

            scene_depth_pc.append(_scene_depth_pc)
            object_depth_pc.append(_object_depth_pc)
            object_pose.append(_object_pose)
            object_visible_fraction.append(_object_visible_fraction[0])

        if len(scene_depth_pc) != 0:
            scene_depth_pc = torch.stack(scene_depth_pc)
            object_depth_pc = torch.stack(object_depth_pc)
            object_pose = np.stack(object_pose, 0)

    return scene_depth_pc, object_depth_pc, object_pose, object_visible_fraction


def get_keypoints(object_label, ds_name=None, ds_dir=BOP_DS_DIR):
    """
    :param ds_name: bop dataset name
    :param object_label: object label
    :return: annotated keypoints : torch.Tensor of shape (3, N)
    """
    if ds_name is not None:
        keypoint_file_location = ds_dir / ds_name.split(".")[0] / "annotations"
    else:
        keypoint_file_location = ds_dir / "annotations"

    keypoint_file_name = object_label + ".npy"
    keypoints_np = np.load(keypoint_file_location / keypoint_file_name)
    keypoints = torch.from_numpy(keypoints_np)

    return keypoints


def visualize_object_depth_point_cloud(ds_name, object_label, model=None, bop_ds_dir=BOP_DS_DIR):
    """
    :param ds_name: bop dataset name
    :param object_label: object label
    :param model: pre-trained model or None, to use ground-turth annotations
    :param visible_fraction_threshold: extracts objects that have min threshold visibility
    :return: visualization: image + segmentation mask, scene depth point cloud, object depth point cloud,
             registered cad model and keypoints
    """

    # ds_name = 'ycbv.train.real'
    # ds_name = 'ycbv.pbr'
    # ds_name = 'tless.pbr'
    # ds_name = 'tless.primesense.train'
    ds = make_scene_dataset(ds_name, load_depth=True, bop_ds_dir=bop_ds_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # keypoint location
    keypoints = get_keypoints(ds_name, object_label)

    for idx, data in enumerate(dl):
        # extracting data
        rgb, mask, obs = data
        scene_depth_pc, object_depth_pc, object_pose, visible_fraction = get_depth_and_pose(
            rgb, mask, obs, object_label, model=model, visualize_segmask=True
        )
        if scene_depth_pc is None:
            continue

        cad_model_mesh = get_cad_model_mesh(object_label, ds_name)

        registered_cad_model_mesh = copy.deepcopy(cad_model_mesh).transform(object_pose)
        registered_keypoints = torch.from_numpy(object_pose[:3, :3]) @ keypoints + torch.from_numpy(object_pose[:3, 3:])

        # visualizing the scene
        visualize_point_cloud_and_mesh(scene_depth_pc, registered_cad_model_mesh)

        # visualizing the segmented object
        point_cloud = pos_tensor_to_o3d(pos=object_depth_pc)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        registered_keypoints_np = registered_keypoints.numpy().transpose()
        visualize_model_n_keypoints(
            model_list=[point_cloud, registered_cad_model_mesh], keypoints_xyz=registered_keypoints_np
        )
        break

    return None


def get_bop_object_images_dataset(
    ds_name, object_label, visible_fraction_lb=0.9, visible_fraction_ub=1.0, bop_ds_dir=BOP_DS_DIR
):
    """
    :param ds_name: bop dataset name
    :param object_label: object label
    :param visible_fraction_threshold: number in [0, 1]
    :return: dataset, contains all images in ds_name that contain object: object_label
             creates an index list at location BOP_DS_DIR / ds_name.split('.')[0] / 'object_images'
    """

    # ds_name = 'ycbv.train.real'
    # ds_name = 'ycbv.pbr'
    # ds_name = 'tless.pbr'
    # ds_name = 'tless.primesense.train'
    ds = make_scene_dataset(ds_name, load_depth=True, bop_ds_dir=bop_ds_dir)
    # breakpoint()
    location = bop_ds_dir / ds_name.split(".")[0] / "object_images"
    if not location.exists():
        location.mkdir()

    # file_name = location / str(
    #    'index_' + ds_name + '_' + object_label + '_lbub_' + str(visible_fraction_lb) + '_' + str(
    #        visible_fraction_ub) + '.pkl')
    file_name = location / str(
        "index_"
        + ds_name
        + "_"
        + "allobjs"
        + "_lbub_"
        + str(visible_fraction_lb)
        + "_"
        + str(visible_fraction_ub)
        + ".pkl"
    )

    if file_name.exists():
        logging.info("Object image index exists. Loading index.")
        file = open(file_name, "rb")
        index_set = pickle.load(file)
        file.close()
    else:
        logging.info("Object image index does not exist. Building index.")
        index_set = build_object_images_index(
            ds.ds_dir, ds.split, BOP_MODEL_INDICES[ds_name.split(".")[0]], ds.frame_index, save_file=file_name
        )

    # convert index set to be consistent with frame_index of the scene dataset
    target_obj_scenes_views = index_set[object_label]
    ds_subset_ids = []
    for scene_id, view_id in target_obj_scenes_views:
        mask = (ds.frame_index["scene_id"] == scene_id) & (ds.frame_index["view_id"] == view_id)
        ds_subset_ids.append(np.where(mask)[0].item())

    dataset = torch.utils.data.Subset(ds, ds_subset_ids)
    logging.info("Object image dataset built.")
    return dataset


class BOPDataset(torch.utils.data.Dataset):
    """
    Main dataset class for all BOP datasets.

    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, ds_name, split="train", load_depth=False, bop_ds_dir=BOP_DS_DIR):
        ds_dir = bop_ds_dir / ds_name
        ds_dir = Path(ds_dir)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), "Dataset does not exists."

        self.split = split
        self.base_dir = ds_dir / split
        models_infos = json.loads((ds_dir / "models" / "models_info.json").read_text())
        self.all_labels = [f"obj_{int(obj_id):06d}" for obj_id in models_infos.keys()]
        self.load_depth = load_depth

        # build indices to retrieve images from view / frame ids
        save_file_index = self.ds_dir / f"index_{split}.feather"
        save_file_annotations = self.ds_dir / f"annotations_{split}.pkl"
        save_file_obj_index = self.ds_dir / f"obj2imgs_index_{split}.pkl"
        if (
            not os.path.isfile(save_file_index)
            or not os.path.isfile(save_file_annotations)
            or not os.path.isfile(save_file_obj_index)
        ):
            logging.info(
                f"No existing dataset index. Building one at {save_file_index}, "
                f"{save_file_annotations}, and {save_file_obj_index}"
            )
            start_time = time.time()
            build_index(
                ds_dir=ds_dir,
                save_file=save_file_index,
                all_objects_labels=self.all_labels,
                save_file_annotations=save_file_annotations,
                save_file_obj_index=save_file_obj_index,
                split=split,
            )
            end_time = time.time()
            logging.info(f"Index creation took: {end_time - start_time}s for BOP-{ds_dir}-{split}")

        start_time = time.time()
        logging.info(f"Loading index from {save_file_index}, " f"{save_file_annotations}, and {save_file_obj_index}")
        self.frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
        self.annotations = pickle.loads(save_file_annotations.read_bytes())
        self.obj_img_index = pickle.loads(save_file_obj_index.read_bytes())
        end_time = time.time()
        logging.info(f"Index loading took: {end_time - start_time}s for BOP-{ds_dir}-{split}")

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, frame_id):
        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f"{view_id:06d}"
        scene_id_str = f"{int(scene_id):06d}"
        scene_dir = self.base_dir / scene_id_str

        rgb_dir = scene_dir / "rgb"
        if not rgb_dir.exists():
            rgb_dir = scene_dir / "gray"
        rgb_path = rgb_dir / f"{view_id_str}.png"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".jpg")
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".tif")

        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]

        # breakpoint()
        cam_annotation = self.annotations[scene_id_str]["scene_camera"][str(view_id)]
        if "cam_R_w2c" in cam_annotation:
            RC0 = np.array(cam_annotation["cam_R_w2c"]).reshape(3, 3)
            tC0 = np.array(cam_annotation["cam_t_w2c"]) * 0.001
            # breakpoint()
            # TC0 = Transform(RC0, tC0)
            TC0 = get_transform(RC0, tC0)
        else:
            # TC0 = Transform(np.eye(3), np.zeros(3))
            TC0 = get_transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation["cam_K"]).reshape(3, 3)
        # T0C = TC0.inverse()
        T0C = inv_transform(TC0)
        # T0C = T0C.toHomogeneousMatrix()
        camera = {"T0C": T0C, "K": K, "TWC": T0C, "resolution": rgb.shape[:2]}

        # T0C = TC0.inverse()

        objects = []
        mask = np.zeros((h, w), dtype=np.uint8)
        if "scene_gt_info" in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]["scene_gt"][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]["scene_gt_info"][str(view_id)]
            for n in range(n_objects):
                RCO = np.array(annotation[n]["cam_R_m2c"]).reshape(3, 3)
                tCO = np.array(annotation[n]["cam_t_m2c"]) * 0.001
                # breakpoint()
                # TCO = Transform(RCO, tCO)
                TCO = get_transform(RCO, tCO)
                # T^0/world_object
                T0O = T0C @ TCO
                # T0O = T0O.toHomogeneousMatrix()
                obj_id = annotation[n]["obj_id"]
                name = f"obj_{int(obj_id):06d}"
                bbox_visib = np.array(visib[n]["bbox_visib"])
                x, y, w, h = bbox_visib
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                # check depth validity
                px_count_all = visib[n]["px_count_all"]
                px_count_valid = visib[n]["px_count_valid"]
                px_count_visib = visib[n]["px_count_visib"]
                obj = {
                    "label": name,
                    "name": name,
                    "TWO": T0O,
                    "T0O": T0O,
                    "TCO": TCO,
                    "visib_fract": visib[n]["visib_fract"],
                    "px_count_all": px_count_all,
                    "px_count_valid": px_count_valid,
                    "px_count_visib": px_count_visib,
                    "id_in_segm": n + 1,
                    "bbox": [x1, y1, x2, y2],
                }
                objects.append(obj)

            mask_path = scene_dir / "mask_visib" / f"{view_id_str}_all.png"
            if mask_path.exists():
                mask = np.array(Image.open(mask_path))
            else:
                for n in range(n_objects):
                    mask_n = np.array(Image.open(scene_dir / "mask_visib" / f"{view_id_str}_{n:06d}.png"))
                    mask[mask_n == 255] = n + 1

        # (H, W); unique integers correspond to integers and background (0)
        mask = torch.as_tensor(mask)

        if self.load_depth:
            depth_path = scene_dir / "depth" / f"{view_id_str}.png"
            if not depth_path.exists():
                depth_path = depth_path.with_suffix(".tif")
            depth = np.array(inout.load_depth(depth_path))
            camera["depth"] = depth * cam_annotation["depth_scale"] / 1000

        # enable users to access the underlying frame ID to retrieve data from other datasets
        # for example, retrieve the GT NOCS from the BOPNOCSDataset
        frame_info = row.to_dict()
        frame_info["frame_id"] = frame_id

        obs = {"objects": objects, "camera": camera, "frame_info": frame_info}
        return rgb, mask, obs


class BOPSceneDataset(BOPDataset):
    """BOP dataset class that allows for selecting specific scenes"""

    def __init__(
        self, ds_name, split="train", load_depth=False, bop_ds_dir=BOP_DS_DIR, scenes_to_load=None, bop19_targets=False
    ):
        super().__init__(ds_name=ds_name, split=split, load_depth=load_depth, bop_ds_dir=bop_ds_dir)

        # create additional filter on the dataset index to only select the specified scene
        if scenes_to_load is not None:
            print(f"Selecting scenes to load: {scenes_to_load}")
            self.frame_index = self.frame_index.loc[self.frame_index["scene_id"].isin(scenes_to_load)]
        else:
            print(f"Using all scenes available: {set(self.frame_index['scene_id'])}.")

        self.bop19_targets = bop19_targets
        if self.bop19_targets:
            print("Using BOP19 targets.")
            keep_bop19(self)

    def collate_fn(self, batch):
        """Custom collate function to handle the per frame objects"""
        num_imgs = len(batch)
        rgbs = torch.zeros(num_imgs, 3, batch[0][0].shape[0], batch[0][0].shape[1])
        masks = torch.zeros(num_imgs, 1, batch[0][1].shape[0], batch[0][1].shape[1])
        depths = torch.zeros_like(masks)
        intrinsics = torch.zeros(num_imgs, 1, 3, 3)
        world_T_cam = torch.zeros(num_imgs, 1, 4, 4)
        objs, frames_info = [], []

        for i, (rgb, mask, obs) in enumerate(batch):
            rgbs[i, ...] = vF.to_tensor(rgb)
            masks[i, ...] = default_convert(mask)
            depths[i, ...] = default_convert(obs["camera"]["depth"])
            intrinsics[i, ...] = default_convert(obs["camera"]["K"])
            world_T_cam[i, ...] = default_convert(obs["camera"]["TWC"])
            # only add valid objects
            objs.append([entry for entry in obs["objects"] if is_bop_obj_det_valid(entry)])
            frames_info.append(obs["frame_info"])
        return rgbs, masks, depths, intrinsics, world_T_cam, objs, frames_info


class BOPObjectDataset:
    """
    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, ds_dir, preload_meshes=False):
        ds_dir = Path(ds_dir)
        infos_file = ds_dir / "models_info.json"
        infos = json.loads(infos_file.read_text())
        objects = []
        for obj_id, bop_info in infos.items():
            obj_id = int(obj_id)
            obj_label = f"obj_{obj_id:06d}"
            mesh_path = (ds_dir / obj_label).with_suffix(".ply").as_posix()
            obj = dict(label=obj_label, category=None, mesh_path=mesh_path, mesh_units="mm")
            is_symmetric = False
            for k in ("symmetries_discrete", "symmetries_continuous"):
                obj[k] = bop_info.get(k, [])
                if len(obj[k]) > 0:
                    is_symmetric = True
            obj["is_symmetric"] = is_symmetric
            obj["diameter"] = bop_info["diameter"]
            scale = 0.001 if obj["mesh_units"] == "mm" else 1.0
            obj["diameter_m"] = bop_info["diameter"] * scale
            objects.append(obj)

        self.objects = objects
        self.meshes = None
        if preload_meshes:
            self.preload_meshes()
        self.ds_dir = ds_dir

    def __getitem__(self, idx):
        return self.objects[idx]

    def preload_meshes(self):
        self.meshes = {}
        for oentry in self.objects:
            mesh = trimesh.load(oentry["mesh_path"], force="mesh", skip_texture=True)
            label = oentry["label"]
            self.meshes[label] = mesh

    def get_mesh(self, label):
        if self.meshes is not None:
            return self.meshes[label]
        else:
            for oentry in self.objects:
                if oentry["label"] == label:
                    return trimesh.load(oentry["mesh_path"], force="mesh", skip_texture=True)
            raise ValueError(f"Unknown label: {label}")

    def __len__(self):
        return len(self.objects)


class BOPObjectPCDataset(BOPObjectDataset):
    def __init__(self, ds_dir, preload_meshes=True, per_obj_pc_size=5000):
        super().__init__(ds_dir, preload_meshes=preload_meshes)
        self.object_pcs = None
        self.per_obj_pc_size = per_obj_pc_size
        self._sample_pcs()

    def _sample_pcs(self):
        self.object_pcs = {}
        for oentry in self.objects:
            mesh = self.get_mesh(oentry["label"])
            samples, _ = mesh.sample(self.per_obj_pc_size, return_index=True)
            self.object_pcs[oentry["label"]] = np.array(samples).transpose()

    def get_pc(self, label):
        return self.object_pcs[label]


def get_bop_objects(labels, ds_name="ycbv", bop_ds_dir=BOP_DS_DIR):
    ds_dir = bop_ds_dir / ds_name / "models"
    dataset = BOPObjectDataset(ds_dir)
    cad_models = dict()

    for obj in dataset:
        if obj["label"] in labels:
            cad_models[obj["label"]] = obj["mesh_path"]

    return cad_models


class UrdfDataset:
    """
    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, ds_dir):
        ds_dir = Path(ds_dir)
        index = []
        for urdf_dir in Path(ds_dir).iterdir():
            urdf_paths = list(urdf_dir.glob("*.urdf"))
            if len(urdf_paths) == 1:
                urdf_path = urdf_paths[0]
                infos = dict(label=urdf_dir.name, urdf_path=urdf_path.as_posix(), scale=1.0)
                index.append(infos)
        self.index = pd.DataFrame(index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]

    def __len__(self):
        return len(self.index)


class BOPUrdfDataset(UrdfDataset):
    """
    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, ds_dir):
        super().__init__(ds_dir)
        self.index["scale"] = 0.001


class OneUrdfDataset:
    """
    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, urdf_path, label, scale=1.0):
        index = [dict(urdf_path=urdf_path, label=label, scale=scale)]
        self.index = pd.DataFrame(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]


class UrdfMultiScaleDataset(UrdfDataset):
    """
    Credit: https://github.com/ylabbe/cosypose
    """

    def __init__(self, urdf_path, label, scales=[]):
        index = []
        for scale in scales:
            index.append(dict(urdf_path=urdf_path, label=label + f"scale={scale:.3f}", scale=scale))
        self.index = pd.DataFrame(index)
