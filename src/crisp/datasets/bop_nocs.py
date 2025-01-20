import torch
import numpy as np
import random
import yaml
import os
import PIL
import cv2
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as tvf
import torchvision.transforms as transforms

from crisp.datasets.augmentations import invert_T, to_torch_uint8
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.datasets.bop import BOPDataset, keep_bop19
from crisp.utils.math import depth_to_point_cloud_map_batched, instance_depth_to_point_cloud_torch
from crisp.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class BOPNOCSDataset(Dataset):
    def __init__(
        self,
        # BOPDataset params
        ds_name,
        split="train",
        bop_ds_dir=None,
        # UnifiedObjects params
        unified_objects_dataset_path=None,
        preload_to_mem=False,
        sample_surface_points_count=1000,
        sample_local_nonmnfld_points_count=1000,
        sample_global_nonmnfld_points_count=5000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=(-0.5, 0.5),
        pc_size=10000,
        input_H=224,
        input_W=224,
        min_area=50,
        scenes_to_load=None,
        normalized_recons=True,
        data_to_output=None,
        debug_vis=False,
        bop19_targets=False,
    ):
        if bop_ds_dir is None:
            raise ValueError("bop_ds_dir must be provided.")

        self.bop_dataset = BOPDataset(ds_name=ds_name, split=split, load_depth=True, bop_ds_dir=Path(bop_ds_dir))
        if scenes_to_load is not None:
            print(f"Selecting scenes to load for BOPNOCSDataset: {scenes_to_load}")
            self.bop_dataset.frame_index = self.bop_dataset.frame_index.loc[
                self.bop_dataset.frame_index["scene_id"].isin(scenes_to_load)
            ]

        if bop19_targets:
            print("Using BOP19 targets for BOPNOCSDataset.")
            keep_bop19(self.bop_dataset)

        # for SDF and NOCS
        unified_data_to_output = ["rgb", "nocs", "coords", "normals", "sdf", "cam_intrinsics", "cam_pose"]
        self.unified_objects = UnifiedObjects(
            folder_path=unified_objects_dataset_path,
            shapenet_dataset_path=None,
            bop_dataset_path=bop_ds_dir,
            replicacad_dataset_path=None,
            preload_to_mem=preload_to_mem,
            pc_size=pc_size,
            sample_surface_points_count=sample_surface_points_count,
            sample_local_nonmnfld_points_count=sample_local_nonmnfld_points_count,
            sample_global_nonmnfld_points_count=sample_global_nonmnfld_points_count,
            global_nonmnfld_points_voxel_res=global_nonmnfld_points_voxel_res,
            sample_bounds=sample_bounds,
            normalized_recons=normalized_recons,
            debug_vis=debug_vis,
            data_to_output=unified_data_to_output,
        )
        self.min_area = min_area
        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )
        self.model_img_H, self.model_img_W = input_H, input_W
        with open(os.path.join(unified_objects_dataset_path, "objects_info.yaml"), "r") as f:
            objects_info = yaml.safe_load(f)
        self.objects_info = objects_info["ycbv"]

        if data_to_output is None:
            data_to_output = [
                "rgb",
                "nocs",
                "coords",
                "normals",
                "sdf",
                "cam_intrinsics",
                "cam_pose",
                "instance_segmap",
            ]
        self.data_to_output = data_to_output

        return

    def __len__(self):
        return len(self.bop_dataset)

    def __getitem__(self, frame_id, obj_idx=None):
        """Specify obj_idx to sample a specific object. Otherwise, sample randomly."""
        # load frame data
        rgb, mask, state = self.bop_dataset.__getitem__(frame_id)

        # sample object and pose
        mask = to_torch_uint8(mask)
        mask_uniqs = set(np.unique(mask))
        objects_visible = []
        for obj in state["objects"]:
            add = False
            if obj["id_in_segm"] in mask_uniqs and np.all(np.array(obj["bbox"]) >= 0):
                add = True

            if add and self.min_area is not None:
                bbox = np.array(obj["bbox"])
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area >= self.min_area:
                    add = True
                else:
                    add = False
            if add:
                objects_visible.append(obj)
        if len(objects_visible) == 0:
            raise ValueError("No object found. ")

        # rgb = torch.as_tensor(rgb).permute(2, 0, 1).to(torch.uint8)
        rgb = self.img_transform(rgb)
        assert rgb.shape[0] == 3

        if obj_idx is None:
            obj = random.sample(objects_visible, k=1)[0]
        else:
            obj = objects_visible[obj_idx]
        TWO = torch.as_tensor(obj["TWO"])
        TWC = torch.as_tensor(state["camera"]["TWC"])
        TCO = invert_T(TWC) @ TWO
        TOC = invert_T(TCO)

        # process depths & rgb (crop and interpolate)
        depth = torch.as_tensor(state["camera"]["depth"]).unsqueeze(-1).permute(2, 0, 1).float()
        depth_map_x_indices = torch.arange(depth.shape[-1])
        depth_map_y_indices = torch.arange(depth.shape[-2])
        depth_map_grid_x, depth_map_grid_y = torch.meshgrid(depth_map_x_indices, depth_map_y_indices, indexing="xy")
        pc = depth_to_point_cloud_map_batched(
            depth[0:1, ...].unsqueeze(0),
            torch.as_tensor(state["camera"]["K"]).unsqueeze(0).unsqueeze(0),
            grid_x=depth_map_grid_x,
            grid_y=depth_map_grid_y,
        )

        bbox = np.asarray(obj["bbox"])
        x, y, x2, y2 = bbox
        w = x2 - x
        h = y2 - y
        cropped_image = tvf.crop(rgb, top=y, left=x, width=w, height=h).contiguous()
        cropped_depth = tvf.crop(depth, top=y, left=x, width=w, height=h).contiguous()
        cropped_mask = tvf.crop(mask, top=y, left=x, width=w, height=h).unsqueeze(0).contiguous()
        cropped_pcs = tvf.crop(pc.squeeze(0), top=y, left=x, width=w, height=h).float().contiguous()

        processed_img = nn.functional.interpolate(
            cropped_image.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="bilinear"
        )
        processed_depth = nn.functional.interpolate(
            cropped_depth.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        )
        processed_mask = nn.functional.interpolate(
            cropped_mask.unsqueeze(0).float(), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        ).int()
        processed_mask = torch.eq(processed_mask, obj["id_in_segm"])

        pc = nn.functional.interpolate(
            cropped_pcs.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        )
        nonzero_pc_mask = torch.ne(torch.sum(pc[0, ...], dim=0, keepdim=True), 0)
        processed_mask = torch.logical_and(processed_mask, nonzero_pc_mask)

        # transform to NOCS
        pc_cad = pc.reshape(3, -1)
        pc_cad = torch.cat((pc_cad, torch.ones((1, pc_cad.shape[1]))), 0)
        pc_cad = invert_T(TCO.float()) @ pc_cad.float()
        pc_cad = pc_cad[:3, :].numpy()

        obj_meta = self.objects_info[obj["label"]]
        blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender = (
            np.array((1, 1, 1)),
            np.array(obj_meta["blender_R_cad"]),
            np.array(obj_meta["recons_t_blender"]),
            obj_meta["recons_s_blender"],
        )

        # these should bring models from CAD frame to blender frame (should be consistent with NOCS)
        blender_coords = blender_s_cad.reshape((3, 1)) * blender_R_cad @ pc_cad
        nocs = recons_s_blender * blender_coords + recons_t_blender.reshape((3, 1))
        if self.unified_objects.normalized_recons:
            nocs = nocs / 2 + 0.5
        else:
            # remove the normalization scale factor
            nocs /= obj_meta["recons_s_blender"]
        nocs = torch.tensor(nocs.reshape(3, self.model_img_H, self.model_img_W).astype(np.float32))
        nocs *= processed_mask.squeeze(0)

        # SDF
        obj_geom = self.unified_objects.distinct_objects["ycbv"][obj["label"]]

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
            data["rgb"] = processed_img.squeeze()

        if "depth" in self.data_to_output:
            data["depth"] = processed_depth.squeeze()

        if "normals" in self.data_to_output:
            data["normals"] = obj_geom["normals"]

        if "instance_segmap" in self.data_to_output:
            data["instance_segmap"] = processed_mask.squeeze()

        if "cam_intrinsics" in self.data_to_output:
            data["cam_intrinsics"] = torch.tensor(state["camera"]["K"].astype(np.float32))

        if "coords" in self.data_to_output:
            data["coords"] = coords.float()

        if "normalized_mesh" in self.data_to_output:
            data["normalized_mesh"] = normalized_mesh

        if "nocs" in self.data_to_output:
            data["nocs"] = nocs

        if "sdf" in self.data_to_output:
            data["sdf"] = sdf

        if "sdf_grid" in self.data_to_output:
            data["sdf_grid"] = obj_geom["nonmnfld_coords_global_sdf"]

        if "cam_pose" in self.data_to_output:
            data["cam_pose"] = TOC.float()

        if "object_pc" in self.data_to_output:
            data["object_pc"] = obj_geom["surface_points"]

        if "metadata" in self.data_to_output:
            data["metadata"] = obj

        if "frame_info" in self.data_to_output:
            data["frame_info"] = state["frame_info"]

        return data
