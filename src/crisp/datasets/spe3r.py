"""
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)

Modified by Jingnan Shi
"""

import os
import yaml
import numpy as np
import random
import cv2
import json
from pathlib import Path
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DistributedSampler
import torchvision.transforms.v2 as tvt
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.io import load_objs_as_meshes

from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import crisp.utils.math
from crisp.utils.math import se3_inverse_torch

logger = logging.getLogger(__name__)


def load_spe3r_camera_intrinsics(camera_json):
    """Helper function to load the camera intrinsics JSON file for SPE3R dataset"""
    with open(camera_json) as f:
        cam = json.load(f)

    if "distCoeffs" in cam.keys():
        cam["distCoeffs"] = np.array(cam["distCoeffs"], dtype=np.float32)

    # Compute horizontal FOV
    cam["horizontalFOV"] = 2.0 * np.arctan2(0.5 * cam["ppx"] * cam["Nu"], cam["fx"])

    return cam


def _seed_worker(worker_id):
    """Set seeds for dataloader workers. For more information, see below
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseModelDataset(object):
    """Base python class to hold all info (e.g., images, meshes) for EACH model"""

    def __init__(
        self,
        tag,
        path_to_model_dir,
        image_dir="images",
        mask_dir="masks",
        depth_dir="depths",
        nocs_dir="nocs",
        mesh=None,
        image_size=(256, 256),
    ):
        self.tag = tag
        self.image_size = image_size

        # Mesh
        self.path_to_mesh_file = Path(path_to_model_dir) / "models" / "model_normalized.obj"
        self._mesh_gt = mesh
        self.model_name = path_to_model_dir.name

        # Image & pose paths
        self.path_to_image_dir = Path(path_to_model_dir) / image_dir
        self.path_to_mask_dir = Path(path_to_model_dir) / mask_dir
        self.path_to_depth_dir = Path(path_to_model_dir) / depth_dir
        if not self.path_to_depth_dir.exists():
            self.path_to_depth_dir.mkdir(parents=True)
        self.path_to_nocs_dir = Path(path_to_model_dir) / nocs_dir
        if not self.path_to_nocs_dir.exists():
            self.path_to_nocs_dir.mkdir(parents=True)

        path_to_pose_json = Path(path_to_model_dir) / "labels.json"

        # Other paths
        self.path_to_surface_points = Path(path_to_model_dir) / "surface_points.npz"
        self.path_to_occupancy = Path(path_to_model_dir) / "occupancy_points.npz"

        # Read .json
        if path_to_pose_json.exists():
            with open(str(path_to_pose_json), "r") as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    @property
    def mesh(self):
        if self._mesh_gt is None:
            self._mesh_gt = load_objs_as_meshes([self.path_to_mesh_file], load_textures=False)
        return self._mesh_gt

    def random_nocspath(self, ri=None):
        if ri is None:
            ri = np.random.choice(len(self))
        return str(self.path_to_nocs_dir / (self.labels[ri]["filename"] + ".png"))

    def random_depthpath(self, ri=None):
        if ri is None:
            ri = np.random.choice(len(self))
        return str(self.path_to_depth_dir / (self.labels[ri]["filename"] + ".png"))

    def random_imagepath(self, ri=None):
        if ri is None:
            ri = np.random.choice(len(self))
        return str(self.path_to_image_dir / (self.labels[ri]["filename"] + ".jpg"))

    def random_maskpath(self, ri=None):
        if ri is None:
            ri = np.random.choice(self.num_masks)
        return str(self.path_to_mask_dir / (self.labels[ri]["filename"] + ".png"))

    def random_pose(self, ri=None):
        if ri is None:
            ri = np.random.choice(self.num_masks)
        return self.labels[ri]["r_Vo2To_vbs_true"], self.labels[ri]["q_vbs2tango_true"]


class SPE3R:
    """Creates BaseModelDataset for each model"""

    def __init__(self, dataset_root, tags=None, img_dir="images", mask_dir="masks", img_size=(256, 256)):
        base_dir = Path(dataset_root)

        # List of PosixPath's (absolute path) for tags
        all_paths = sorted(x for x in Path(base_dir).iterdir() if x.is_dir())

        # List of tags
        tags_in_dir = [p.name for p in all_paths]

        if tags:
            # Custom list of tags (e.g., those in train & val)
            self._tags = [t for t in tags if t in tags_in_dir]
            self._paths = [p for p in all_paths if p.name in self._tags]
        else:
            self._tags = tags_in_dir
            self._paths = all_paths

        # Misc.
        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = img_size

        # Pre-load
        self.datasets = []
        for i in range(len(self)):
            self.datasets.append(
                BaseModelDataset(
                    self._tags[i],
                    self._paths[i],
                    image_dir=self.image_dir,
                    mask_dir=self.mask_dir,
                    mesh=None,
                    image_size=self.image_size,
                )
            )

    def _get_base_model_of_tag(self, i, mesh=None):
        if not self.datasets:
            return self.datasets[i]
        else:
            return BaseModelDataset(
                self._tags[i],
                self._paths[i],
                image_dir=self.image_dir,
                mask_dir=self.mask_dir,
                mesh=mesh,
                image_size=self.image_size,
            )

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, i):
        return self._get_base_model_of_tag(i)


class SatReconDataset(torch.utils.data.Dataset):
    """torch Dataset class to be loaded"""

    def __init__(
        self,
        dataset_root,
        split="train",
        rgb_transform=None,
        mask_transform=None,
        nocs_transform=None,
        depth_transform=None,
        output_mesh=False,
        output_occupancy=True,
        output_surface_points=True,
        model_name=None,
        split_csv="splits.csv",
        num_points_on_mesh=2000,
        num_points_in_mesh=10000,
        depth_z_far=100,
        nocs_scale=65536,
        depth_scale=65536,
    ):
        super(SatReconDataset, self).__init__()

        assert split in ["train", "validation", "test", "all"]

        self.root_dir = Path(dataset_root)
        self.split = split
        self.split_csv = split_csv
        self.is_train = split == "train"
        self.rgb_transform = rgb_transform
        self.mask_transform = mask_transform
        self.nocs_transform = nocs_transform
        self.depth_transform = depth_transform

        # Misc
        self.rgb = True
        self.output_mesh = output_mesh
        self.output_occupancy = output_occupancy
        self.output_surface_points = output_surface_points
        self.num_points_on_mesh = num_points_on_mesh
        self.num_points_in_mesh = num_points_in_mesh
        self.depth_z_far = depth_z_far
        self.nocs_scale = nocs_scale
        self.depth_scale = depth_scale

        # When there are splits, tags should only contain those in CSV file
        if split_csv is not None:
            csv = pd.read_csv(str(self.root_dir / split_csv), header=None)

            if split == "all":
                tags = [
                    csv.iloc[idx, 0]
                    for idx in range(len(csv))
                    if (model_name is None or csv.iloc[idx, 0] == model_name)
                ]
            else:
                temp = "train" if split == "train" or split == "validation" else "test"
                tags = [
                    csv.iloc[idx, 0]
                    for idx in range(len(csv))
                    if csv.iloc[idx, 1] == temp and (model_name is None or csv.iloc[idx, 0] == model_name)
                ]
        else:
            tags = None

        # hard filtering rejecting the aquarius_ud model due to incorrect GT pose labels
        tags = [x for x in tags if x not in ["aquarius_ud"]]

        # Dataset object
        self.original_image_size = (256, 256)
        self.datasets = SPE3R(dataset_root=dataset_root, tags=tags)

        # NOTE: SPE3R has 1,000 images total, set 200 apart for validation
        # - Train:      0001 ~ 0400, 0501 ~ 0900
        # - Validation: 0401 ~ 0500, 0901 ~ 1000
        if split == "train":
            self.image_indices = list(range(400)) + list(range(500, 900))
        elif split == "validation":
            self.image_indices = list(range(400, 500)) + list(range(900, 1000))
        else:
            self.image_indices = list(range(1000))

        # load camera intrinsics
        self.camera_intrinsics = load_spe3r_camera_intrinsics(self.root_dir / "camera.json")

        logger.info(f"Dataset: {dataset_root} ({split})")
        logger.info(f"   • Num. models:         {self.num_models}")
        logger.info(f"   • Num. images / model: {self.num_images_per_model}")
        logger.info(f"   • Num. GT pts ON mesh: {self.num_points_on_mesh}")
        logger.info(f"   • Num. GT pts IN mesh: {self.num_points_in_mesh}")

    @property
    def num_models(self):
        return len(self.datasets)

    @property
    def num_images_per_model(self):
        return len(self.image_indices)

    def __len__(self):
        # Number of models * number of images per model
        return len(self.datasets) * self.num_images_per_model

    def __getitem__(self, idx):
        modelidx = self._get_model_idex(idx)
        imageidx = idx % self.num_images_per_model

        batch = self._get_item(modelidx, imgidx=imageidx)

        return batch

    def _get_model_idex(self, idx):
        modelidx = int(idx / self.num_images_per_model)
        return modelidx

    def _get_item(self, modelidx, imgidx=None):
        # Grab idx'th MODEL
        dataset = self.datasets[modelidx]

        # ---------- Get image & mask
        if imgidx is None:
            imgidx = random.randrange(self.num_images_per_model)

        # Get correct image index
        imgidx = self.image_indices[imgidx]

        # NOTE: Just to make sure we got image indexing right
        if self.split == "train":
            assert (imgidx >= 0 and imgidx < 400) or (
                imgidx >= 500 and imgidx < 900
            ), f"Got imgidx = {imgidx} for {self.split} split"
        elif self.split == "validation":
            assert (imgidx >= 400 and imgidx < 500) or (
                imgidx >= 900 and imgidx < 1000
            ), f"Got imgidx = {imgidx} for {self.split} split"

        # Load image & mask
        image = self._load_image(dataset.random_imagepath(ri=imgidx))
        initial_mask = self._load_mask(dataset.random_maskpath(ri=imgidx))
        depth = self._load_depth(dataset.random_depthpath(ri=imgidx))
        nocs = self._load_nocs(dataset.random_nocspath(ri=imgidx))

        # Apply transform on everything
        image = self.rgb_transform(image)
        initial_mask = self.mask_transform(initial_mask)
        depth = self.depth_transform(depth)
        nocs = self.nocs_transform(nocs)
        assert depth.shape[1] == nocs.shape[1] == image.shape[1] == initial_mask.shape[1]
        assert depth.shape[2] == nocs.shape[2] == image.shape[2] == initial_mask.shape[2]

        # update mask to mask out all zero NOCS values, depth
        nonzero_mask = torch.logical_and(nocs != 0, depth != 0)
        nonzero_mask = torch.ge(torch.sum(nonzero_mask, dim=0), 1)
        mask = torch.logical_and(initial_mask, nonzero_mask).squeeze(0)

        # --------- Get pose
        trans, quat = dataset.random_pose(ri=imgidx)

        trans = torch.tensor(trans, dtype=torch.float32)
        quat = torch.tensor(quat, dtype=torch.float32)
        rot = quaternion_to_matrix(quat)

        batch = {
            "image": image,
            "mask": mask,
            "nocs": nocs,
            "depth": depth,
            "trans": trans,
            "rot": rot,
            "model_name": dataset.model_name,
        }

        # ---------- Get mesh surface points
        if self.output_surface_points:
            surface = np.load(dataset.path_to_surface_points, allow_pickle=True)
            pidx = random.sample(list(range(100000)), self.num_points_on_mesh)
            points_on_mesh = torch.tensor(surface["points"][0, pidx], dtype=torch.float32)
            batch["points_on_mesh"] = points_on_mesh

        # ---------- Get occupancy labels
        if self.output_occupancy:
            occupancy = np.load(dataset.path_to_occupancy, allow_pickle=True)  # [N,]

            points = occupancy["points"]
            occ_labels = occupancy["labels"]

            n_positive = occ_labels.sum()

            if n_positive < self.num_points_in_mesh / 2:
                # Not enough positive points -- use all
                idx_positive = np.where(occ_labels == 1)[0]
                idx_negative = np.random.choice(np.where(occ_labels == 0)[0], self.num_points_in_mesh - n_positive)
            else:
                # Enough positive points -- sample
                idx_positive = np.random.choice(np.where(occ_labels == 1)[0], int(self.num_points_in_mesh / 2))
                idx_negative = np.random.choice(np.where(occ_labels == 0)[0], int(self.num_points_in_mesh / 2))

            pidx = np.concatenate([idx_positive, idx_negative])
            assert len(pidx) == self.num_points_in_mesh

            points_in_mesh = torch.tensor(points[pidx], dtype=torch.float32)
            occ_labels = torch.tensor(occ_labels[pidx], dtype=torch.float32)
            occ_weights = torch.ones_like(occ_labels)

            batch["occ_labels"] = occ_labels
            batch["points_in_mesh"] = points_in_mesh
            batch["occ_weights"] = occ_weights

        if self.output_mesh:
            batch["mesh"] = dataset.mesh
            batch["model_idx"] = modelidx
            batch["base_filename"] = dataset.labels[imgidx]["filename"]

        return batch

    def _load_nocs(self, fn):
        data = cv2.imread(fn, cv2.IMREAD_UNCHANGED).astype(np.float32)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = data / self.nocs_scale
        return data

    def _load_depth(self, fn):
        data = cv2.imread(fn, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        data = data / self.depth_scale * self.depth_z_far
        return data

    def _load_image(self, fn):
        """Read image of given index from a folder, if specified"""
        data = cv2.imread(fn, cv2.IMREAD_COLOR)

        if self.rgb:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            # Force grayscale
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        return data

    def _load_mask(self, fn):
        """Read mask image"""
        data = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        # Clean up any intermediate values
        data[data > 128] = 255
        data[data <= 128] = 0

        return data[:, :, None]
        # return data


class SatReconNOCSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # SatReconDataset params
        dataset_root,
        split="train",
        model_name=None,
        split_csv="splits.csv",
        num_points_on_mesh=2000,
        num_points_in_mesh=10000,
        input_H=140,
        input_W=140,
        # UnifiedObjects params
        unified_objects=None,
        unified_objects_dataset_path=None,
        preload_to_mem=False,
        sample_surface_points_count=1000,
        sample_local_nonmnfld_points_count=1000,
        sample_global_nonmnfld_points_count=5000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=(-1.0, 1.0),
        normalized_recons=True,
        pc_size=10000,
        data_to_output=None,
        debug_vis=False,
    ):
        rgb_transform = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Resize((input_H, input_W), interpolation=tvt.InterpolationMode.BILINEAR),
                tvt.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        mask_transform = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Resize((input_H, input_W), interpolation=tvt.InterpolationMode.NEAREST),
            ]
        )
        nocs_transform = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Resize((input_H, input_W), interpolation=tvt.InterpolationMode.NEAREST),
            ]
        )
        depth_transform = tvt.Compose(
            [
                tvt.ToTensor(),
                tvt.Resize((input_H, input_W), interpolation=tvt.InterpolationMode.NEAREST),
            ]
        )

        self.sat_recon_dataset = SatReconDataset(
            dataset_root,
            split=split,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            nocs_transform=nocs_transform,
            depth_transform=depth_transform,
            output_mesh=False,
            output_surface_points=False,
            output_occupancy=False,
            model_name=model_name,
            split_csv=split_csv,
            num_points_on_mesh=num_points_on_mesh,
            num_points_in_mesh=num_points_in_mesh,
        )

        # for SDF
        if unified_objects is not None:
            self.unified_objects = unified_objects
        else:
            unified_data_to_output = ["coords", "normals", "sdf", "cam_intrinsics", "cam_pose"]
            self.unified_objects = UnifiedObjects(
                folder_path=unified_objects_dataset_path,
                spe3r_dataset_path=dataset_root,
                preload_to_mem=preload_to_mem,
                pc_size=pc_size,
                sample_surface_points_count=sample_surface_points_count,
                sample_local_nonmnfld_points_count=sample_local_nonmnfld_points_count,
                sample_global_nonmnfld_points_count=sample_global_nonmnfld_points_count,
                global_nonmnfld_points_voxel_res=global_nonmnfld_points_voxel_res,
                sample_bounds=sample_bounds,
                debug_vis=debug_vis,
                data_to_output=unified_data_to_output,
                normalized_recons=normalized_recons,
                force_recompute_sdf=False,
            )
        self.model_img_H, self.model_img_W = input_H, input_W
        with open(os.path.join(unified_objects_dataset_path, "objects_info.yaml"), "r") as f:
            objects_info = yaml.safe_load(f)
        self.objects_info = objects_info["spe3r"]

        # scale camera intrinsic matrix
        self.scaled_camera_K = np.array(self.sat_recon_dataset.camera_intrinsics["cameraMatrix"])
        x_scale = self.model_img_W / self.sat_recon_dataset.original_image_size[1]
        y_scale = self.model_img_H / self.sat_recon_dataset.original_image_size[0]
        self.scaled_camera_K[0, 0] *= x_scale
        self.scaled_camera_K[0, -1] *= x_scale
        self.scaled_camera_K[1, 1] *= y_scale
        self.scaled_camera_K[1, -1] *= y_scale
        self.scaled_camera_K = torch.tensor(self.scaled_camera_K).float()

        if data_to_output is None:
            data_to_output = [
                "rgb",
                "nocs",
                "coords",
                "normals",
                "sdf",
                "instance_segmap",
                "model_name",
            ]
        self.data_to_output = data_to_output

    def __len__(self):
        return self.sat_recon_dataset.__len__()

    def __getitem__(self, frame_id):
        # load frame data
        data = self.sat_recon_dataset.__getitem__(frame_id)

        rgb, mask, nocs, depth, trans, rot, model_name = (
            data["image"],
            data["mask"],
            data["nocs"],
            data["depth"],
            data["trans"],
            data["rot"],
            data["model_name"],
        )

        assert rgb.shape[0] == 3

        obj_meta = self.unified_objects.objects_info["spe3r"][model_name]
        blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender = (
            torch.tensor(obj_meta["blender_s_cad"]).double().to(nocs.device),
            torch.tensor(obj_meta["blender_R_cad"]).double().to(nocs.device),
            torch.tensor(obj_meta["recons_t_blender"]).double().to(nocs.device),
            obj_meta["recons_s_blender"],
        )

        # NOCS (cad frame) to NOCS (recons) frame
        # NOCS renders for SPE3R are shifted by 0.5 from the original CAD coordinates
        recons_nocs = nocs.reshape((3, -1)) - 0.5
        recons_nocs = blender_s_cad.reshape((3, 1)) * blender_R_cad @ recons_nocs
        recons_nocs = recons_s_blender * recons_nocs + recons_t_blender.reshape((3, 1))
        recons_nocs = recons_nocs.reshape((3, nocs.shape[1], nocs.shape[2]))
        if self.unified_objects.normalized_recons:
            recons_nocs = recons_nocs / 2.0 + 0.5
        else:
            recons_nocs /= obj_meta["recons_s_blender"]
        recons_nocs *= mask

        # get recons_T_camera
        cam_T_cad = torch.eye(4)
        cam_T_cad[:3, :3] = rot
        cam_T_cad[:3, -1] = trans
        cad_T_recons = torch.eye(4)
        cad_T_recons[:3, :3] = blender_R_cad.T
        cad_T_recons[:3, -1] = -recons_t_blender
        TOC = se3_inverse_torch(cam_T_cad @ cad_T_recons)

        # SDF
        obj_geom = self.unified_objects.distinct_objects["spe3r"][model_name]

        surface_rand_idxs = np.random.choice(
            obj_geom["surface_points"].shape[0], size=self.unified_objects.sample_surface_points_count
        )
        local_nonmnfld_rand_idxs = np.random.choice(
            obj_geom["nonmnfld_coords_local"].shape[0], size=self.unified_objects.sample_local_nonmnfld_points_count
        )
        global_nonmnfld_rand_idxs = np.random.choice(
            obj_geom["nonmnfld_coords_global"].shape[0], size=self.unified_objects.sample_global_nonmnfld_points_count
        )

        sdf = torch.zeros(
            self.unified_objects.sample_surface_points_count
            + self.unified_objects.sample_local_nonmnfld_points_count
            + self.unified_objects.sample_global_nonmnfld_points_count
        )
        nonmnfld_sdf_local = obj_geom["nonmnfld_coords_local_sdf"][local_nonmnfld_rand_idxs]
        sdf[
            self.unified_objects.sample_surface_points_count : self.unified_objects.sample_surface_points_count
            + self.unified_objects.sample_local_nonmnfld_points_count
        ] = nonmnfld_sdf_local

        nonmnfld_sdf_global = obj_geom["nonmnfld_coords_global_sdf"][global_nonmnfld_rand_idxs]
        sdf[
            self.unified_objects.sample_surface_points_count + self.unified_objects.sample_local_nonmnfld_points_count :
        ] = nonmnfld_sdf_global

        # coords
        on_surface_coords = obj_geom["surface_points"][surface_rand_idxs, :]
        nonmnfld_coords_local = obj_geom["nonmnfld_coords_local"][local_nonmnfld_rand_idxs, :]
        nonmnfld_coords_global = obj_geom["nonmnfld_coords_global"][global_nonmnfld_rand_idxs, :]
        coords = torch.cat((on_surface_coords, nonmnfld_coords_local, nonmnfld_coords_global), dim=0)

        normalized_mesh = [
            torch.tensor(obj_geom["normalized_mesh"].vertices).float(),
            torch.tensor(obj_geom["normalized_mesh"].faces).float(),
        ]

        data = {}
        if "rgb" in self.data_to_output:
            data["rgb"] = rgb

        if "depth" in self.data_to_output:
            data["depth"] = depth.float()

        if "normals" in self.data_to_output:
            data["normals"] = obj_geom["normals"]

        if "instance_segmap" in self.data_to_output:
            data["instance_segmap"] = mask

        if "cam_intrinsics" in self.data_to_output:
            data["cam_intrinsics"] = self.scaled_camera_K

        if "coords" in self.data_to_output:
            data["coords"] = coords.float()

        if "normalized_mesh" in self.data_to_output:
            data["normalized_mesh"] = normalized_mesh

        if "nocs" in self.data_to_output:
            data["nocs"] = recons_nocs.float()

        if "sdf" in self.data_to_output:
            data["sdf"] = sdf

        if "cam_pose" in self.data_to_output:
            data["cam_pose"] = TOC.float()

        if "object_pc" in self.data_to_output:
            data["object_pc"] = obj_geom["surface_points"]

        if "normalized_mesh" in self.data_to_output:
            data["normalized_mesh"] = [
                torch.tensor(obj_geom["normalized_mesh"].vertices).float(),
                torch.tensor(obj_geom["normalized_mesh"].faces).float(),
            ]

        if "sdf_grid" in self.data_to_output:
            data["sdf_grid"] = obj_geom["nonmnfld_coords_global_sdf"]

        if "model_name" in self.data_to_output:
            data["model_name"] = model_name

        return data


class DistributedObjClassBatchSampler(DistributedSampler):
    def __init__(
        self,
        per_obj_class_batch_size,
        num_classes_per_batch,
        sat_dataset: SatReconNOCSDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        self.per_obj_class_batch_size = per_obj_class_batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_size = self.num_classes_per_batch * self.per_obj_class_batch_size

        if isinstance(sat_dataset, torch.utils.data.dataset.Subset):
            self.sat_dataset = sat_dataset.dataset
            self.indices = sat_dataset.indices
        else:
            self.sat_dataset = sat_dataset
            self.indices = range(self.sat_dataset.__len__())

        # note: we always drop last
        super().__init__(
            dataset=sat_dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed, drop_last=True
        )

        # load and calculate indices
        self._make_batches()

    def _make_batches(self):
        # read the hdf5 files
        # make obj label to files mapping
        self.obj_cls_to_index = defaultdict(list)
        # In case we are using a Subset wrapped dataset, we need to get the actual file index from the
        # subset indices.
        # If we are using the dataset directly, self.indices is just a consecutive sequence of integers
        for i in tqdm(
            range(len(self.indices)), desc="DistributedObjClassBatchSampler building obj class to index mapping"
        ):
            model_idx = self.sat_dataset.sat_recon_dataset._get_model_idex(i)
            model_name = self.sat_dataset.sat_recon_dataset.datasets[model_idx].model_name
            self.obj_cls_to_index[model_name].append(i)

        # sorted obj classes
        self.all_obj_classes = sorted(list(self.obj_cls_to_index.keys()))

        # calculate how many batches
        # ideally, we want to at least cover each frame 1 time
        # This can be formulated as a coupon collector's problem
        # https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
        # T is the number of samples.
        # E(T) = n * H(n) where n is the number of frame
        num_frames = self.sat_dataset.__len__()
        T = num_frames * crisp.utils.math.H(num_frames)
        self.num_samples = int(np.floor(T / self.batch_size))
        # divide by the number of devices we are running training on
        self.num_samples = int(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_cls = self.all_obj_classes

        for i in range(self.num_samples):
            batch = []
            # sample classes
            shuffled_cls_idx = torch.randperm(len(all_cls), generator=g).tolist()[: self.num_classes_per_batch]
            for j in shuffled_cls_idx:
                cls = all_cls[j]
                # sample images
                cls_img_count = len(self.obj_cls_to_index[cls])
                # sampled_image_idx = np.random.choice(cls_img_count, size=self.per_obj_class_batch_size, replace=True)
                sampled_image_idx = torch.multinomial(
                    torch.tensor(range(cls_img_count)).float(),
                    self.per_obj_class_batch_size,
                    generator=g,
                    replacement=True,
                ).int()
                batch.extend([self.obj_cls_to_index[cls][k] for k in list(sampled_image_idx)])

            yield batch
