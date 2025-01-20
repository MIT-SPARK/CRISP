import os.path
import logging
import json
import numpy as np
import pickle
import yaml
from pathlib import Path
from quaternion import as_rotation_matrix
from scipy.spatial.transform import Rotation as R

import torch
import torchvision.transforms.functional as vF
from torch.utils.data import default_convert

# project imports
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.math import depth_to_point_cloud_map_batched
from crisp.utils.visualization_utils import imgs_show
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista


def parse_replicacad_traj(ds_dir, split):
    agent_poses = []
    rgbs = []
    depths = []
    sems = []
    object_poses = []
    for i in range(len(os.listdir(ds_dir))):
        saved_step = pickle.load(open(f"{ds_dir}/step_{i}.pkl", "rb"))
        agent_pos = saved_step["agent_pos"]
        agent_rot = as_rotation_matrix(saved_step["agent_rot"])
        agent_pose = np.eye(4)
        agent_pose[:3, :3] = agent_rot
        agent_pose[:3, -1] = agent_pos
        agent_poses.append(agent_pose)

        rgb = saved_step["observations"]["color_sensor"]
        depth = saved_step["observations"]["depth_sensor"]
        sem = saved_step["observations"]["semantic_sensor"]
        obj_pos = saved_step["object_poses"]

        rgbs.append(rgb)
        depths.append(depth)
        sems.append(sem)
        object_poses.append(obj_pos)

    split_len = int(0.8 * len(agent_poses))
    val_len = int(0.1 * len(agent_poses))
    if split == "train":
        agent_poses = agent_poses[:split_len]
        rgbs = rgbs[:split_len]
        depths = depths[:split_len]
        sems = sems[:split_len]
        object_poses = object_poses[:split_len]
    elif split == "test":
        agent_poses = agent_poses[split_len + val_len :]
        rgbs = rgbs[split_len + val_len :]
        depths = depths[split_len + val_len :]
        sems = sems[split_len + val_len :]
        object_poses = object_poses[split_len + val_len :]
    else:
        agent_poses = agent_poses[split_len : split_len + val_len]
        rgbs = rgbs[split_len : split_len + val_len]
        depths = depths[split_len : split_len + val_len]
        sems = sems[split_len : split_len + val_len]
        object_poses = object_poses[split_len : split_len + val_len]
    return agent_poses, rgbs, depths, sems, object_poses


def parse_obj_poses(obj_info_list):
    poses = {}
    for obj in obj_info_list:
        objname = obj["template_name"].split("/")[1]
        Tobj = np.eye(4)
        Tobj[:3, :3] = R.from_quat(obj["rotation"]).as_matrix()
        Tobj[:3, -1] = obj["translation"]
        poses[objname] = Tobj
    return poses


class ReplicaCADDataset(torch.utils.data.Dataset):
    """
    Main dataset class for ReplicaCAD trajectories.
    """

    def __init__(
        self,
        # trajectories
        rep_dir,
        split,
        # unified objects dataset
        rep_objects_dir,
        rep_dataset_dir,
        preload_to_mem=False,
        sample_surface_points_count=1000,
        sample_local_nonmnfld_points_count=1000,
        sample_global_nonmnfld_points_count=5000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=(-1.0, 1.0),
        pc_size=10000,
        input_H=600,
        input_W=800,
    ):
        ds_dir = rep_dir
        assert os.path.exists(ds_dir), f"Dataset does not exists at {ds_dir}."
        self.ds_dir = Path(ds_dir).resolve()
        self.root_ds_dir = Path(ds_dir).resolve().parent

        logging.info(f"Creating the dataset from saved ReplicaCAD trajectories at {self.ds_dir}.")
        self.split = split
        self.agent_poses, self.rgbs, self.depths, self.sems, self.object_poses = parse_replicacad_traj(
            ds_dir, self.split
        )
        data_name = rep_dir.split("/")[-1][:-2]
        id_name_mapping = json.load(open(self.root_ds_dir / f"{data_name}.scene_instance.json"))["object_instances"]
        self.obj_T_world = parse_obj_poses(id_name_mapping)
        self.id_name_mapping = [obj["template_name"] for obj in id_name_mapping]
        self.unified_objects = UnifiedObjects(
            folder_path=rep_objects_dir,
            shapenet_dataset_path=None,
            bop_dataset_path=None,
            replicacad_dataset_path=rep_dataset_dir,
            preload_to_mem=preload_to_mem,
            pc_size=pc_size,
            sample_surface_points_count=sample_surface_points_count,
            sample_local_nonmnfld_points_count=sample_local_nonmnfld_points_count,
            sample_global_nonmnfld_points_count=sample_global_nonmnfld_points_count,
            global_nonmnfld_points_voxel_res=global_nonmnfld_points_voxel_res,
            sample_bounds=sample_bounds,
            debug_vis=False,
            data_to_output=None,
        )

        self.objects_info = yaml.safe_load(open(os.path.join(rep_objects_dir, "objects_info.yaml"), "r"))["replicacad"]
        self.input_img_H, self.input_img_W = input_H, input_W
        # based on https://github.com/facebookresearch/OccupancyAnticipation/blob/
        # aea6a2c0d9701336c01c9c85f5c3b8565d7c52ba/habitat_extensions/sensors.py#L363
        hfov = float(90) * np.pi / 180
        vfov = 2 * np.arctan((self.input_img_H / self.input_img_W) * np.tan(hfov / 2.0))
        self.intrinsics_mat = np.array(
            [
                [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                [0.0, 1 / np.tan(vfov / 2.0), 0.0, 0.0],
                [0.0, 0.0, 1, 0],
                [0.0, 0.0, 0, 1],
            ]
        )
        self.inverse_intrinsic_mat = np.linalg.inv(self.intrinsics_mat)
        self.min_depth = 0
        self.max_forward_range = 10
        self._depth_map_x_indices = torch.tensor(np.linspace(-1, 1, input_W))
        self._depth_map_y_indices = torch.tensor(np.linspace(1, -1, input_H))
        self._depth_map_grid_x, self._depth_map_grid_y = torch.meshgrid(
            self._depth_map_x_indices, self._depth_map_y_indices, indexing="xy"
        )
        self.proj_xs, self.proj_ys = np.meshgrid(
            np.linspace(-1, 1, self.input_img_W), np.linspace(1, -1, self.input_img_H)
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array

        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = np.copy(self.proj_xs).reshape(-1)
        ys = np.copy(self.proj_ys).reshape(-1)
        depth_float = depth_float.reshape(-1)
        # Filter out invalid depths
        valid_depths = (depth_float != self.min_depth) & (depth_float <= self.max_forward_range)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]
        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack(
            (
                xs * depth_float,
                ys * depth_float,
                -depth_float,
                np.ones(depth_float.shape),
            )
        )
        inv_K = self.inverse_intrinsic_mat
        xyz_camera = np.matmul(inv_K, xys).T  # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def __len__(self):
        return len(self.agent_poses)

    def __getitem__(self, frame_id):
        rgb = self.rgbs[frame_id][:, :, :3]
        imgs_show([torch.permute(torch.tensor(rgb), (2, 0, 1))])
        depth = self.depths[frame_id]

        # pc2 = self.convert_to_pointcloud(np.expand_dims(depth, (-1)))
        # visualize_pcs_pyvista([pc2], colors=["lightblue"], pt_sizes=[5.0])

        # intr = torch.from_numpy(np.expand_dims(self.intrinsics_mat, (0, 1)))
        # dep = torch.from_numpy(np.expand_dims(depth, (0, 1)))
        # pc = depth_to_point_cloud_map_batched(
        #    dep, intr, grid_x=self._depth_map_grid_x, grid_y=self._depth_map_grid_y
        # )
        ## TODO: Habitat has -z pointing away from the camera
        # visualize_pcs_pyvista([pc.squeeze().reshape(3, -1)], colors=["lightblue"], pt_sizes=[5.0])

        mask = self.sems[frame_id]
        object_info = self.object_poses[frame_id]
        cam_T = self.agent_poses[frame_id]
        if frame_id == 0:
            cam_T_prev = np.eye(4)
        else:
            cam_T_prev = self.agent_poses[frame_id - 1]

        camera = {"K": self.intrinsics_mat, "resolution": rgb.shape[:2]}

        objects = []
        breakpoint()
        objects_seg_id = {}
        for n in object_info:
            if n == 0:
                continue
            name = n[:-6]
            for i, d in enumerate(self.id_name_mapping):
                if d.split("/")[-1] == name:
                    obj_id = i
                    break
            y1 = np.where(mask == obj_id)[0].min()
            y2 = np.where(mask == obj_id)[0].max()
            x1 = np.where(mask == obj_id)[1].min()
            x2 = np.where(mask == obj_id)[1].max()
            obj = {"label": name, "name": name, "id_in_segm": obj_id, "bbox": [x1, y1, x2, y2], "TCO": cam_T}
            objects.append(obj)
            objects_seg_id[name] = obj_id

        # (H, W); unique integers correspond to integers and background (0)
        camera["depth"] = depth
        camera["rel_pose"] = np.linalg.inv(cam_T_prev) @ cam_T
        camera["frame_id"] = frame_id

        obs = {"objects": objects, "camera": camera}
        all_meta_unified = self.unified_objects.distinct_objects["replicacad"]
        scene_sdfs = []
        scene_coords = []
        scene_nocs = []
        scene_normals = []
        scene_pcs = []
        # get sdf and nocs related info
        object_info_keys = list(object_info.keys())
        for rcad_obj in object_info_keys:
            # obtain object name
            objname = rcad_obj[:-12]
            meta_unified = all_meta_unified[objname]

            # get coords
            surface_rand_idxs = np.random.choice(
                meta_unified["surface_points"].shape[0], size=self.unified_objects.sample_surface_points_count
            )
            local_nonmnfld_rand_idxs = np.random.choice(
                meta_unified["nonmnfld_coords_local"].shape[0],
                size=self.unified_objects.sample_local_nonmnfld_points_count,
            )
            global_nonmnfld_rand_idxs = np.random.choice(
                meta_unified["nonmnfld_coords_global"].shape[0],
                size=self.unified_objects.sample_global_nonmnfld_points_count,
            )
            on_surface_coords = meta_unified["surface_points"][surface_rand_idxs, :]
            nonmnfld_coords_local = meta_unified["nonmnfld_coords_local"][local_nonmnfld_rand_idxs, :]
            nonmnfld_coords_global = meta_unified["nonmnfld_coords_global"][global_nonmnfld_rand_idxs, :]
            coords = torch.cat((on_surface_coords, nonmnfld_coords_local, nonmnfld_coords_global), dim=0)
            scene_coords.append(coords)

            # get sdf
            sdf = torch.zeros(
                self.unified_objects.sample_surface_points_count
                + self.unified_objects.sample_local_nonmnfld_points_count
                + self.unified_objects.sample_global_nonmnfld_points_count
            )
            nonmnfld_sdf_local = meta_unified["nonmnfld_coords_local_sdf"][local_nonmnfld_rand_idxs]
            sdf[
                self.unified_objects.sample_surface_points_count : self.unified_objects.sample_surface_points_count
                + self.unified_objects.sample_local_nonmnfld_points_count
            ] = nonmnfld_sdf_local

            nonmnfld_sdf_global = meta_unified["nonmnfld_coords_global_sdf"][global_nonmnfld_rand_idxs]
            sdf[
                self.unified_objects.sample_surface_points_count
                + self.unified_objects.sample_local_nonmnfld_points_count :
            ] = nonmnfld_sdf_global
            scene_sdfs.append(sdf)

            # get NOCS
            obj_meta = self.objects_info[objname]
            blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender = (
                np.array(obj_meta["blender_s_cad"]),
                np.array(obj_meta["blender_R_cad"]),
                np.array(obj_meta["recons_t_blender"]),
                obj_meta["recons_s_blender"],
            )

            # these should bring models from CAD frame to blender frame (should be consistent with NOCS)
            intr = torch.from_numpy(np.expand_dims(self.intrinsics_mat, (0, 1)))
            dep = torch.from_numpy(np.expand_dims(depth, (0, 1)))

            # get pcd
            pc = depth_to_point_cloud_map_batched(
                dep, intr, grid_x=self._depth_map_grid_x, grid_y=self._depth_map_grid_y
            )

            pc2 = self.convert_to_pointcloud(np.expand_dims(depth, (-1)))

            visualize_pcs_pyvista([pc2], colors=["lightblue"], pt_sizes=[5.0])

            obj_T_world = self.obj_T_world[rcad_obj["label"]]
            cam_T_obj = np.linalg.inv(np.linalg.inv(cam_T) @ obj_T_world)
            pc = pc.cpu().numpy()[0]
            # pc = pc[:, mask == rcad_obj["id_in_segm"]]
            pc = pc.reshape((-1, 3))
            pc = pc @ cam_T_obj[:3, :3] + cam_T_obj[:3, -1]
            blender_coords = blender_s_cad.reshape((1, 3)) * pc @ blender_R_cad.T

            # get nocs in depth-aligned frame
            nocs = recons_s_blender * blender_coords + recons_t_blender.reshape((1, 3))
            # downsample nocs
            rand_nocs_idx = np.random.randint(0, len(nocs), 5000)
            # nocs = nocs[rand_nocs_idx]
            nocs = nocs.reshape(3, 1200, 1600)
            nocs[:, mask != objects_seg_id[rcad_obj[:-6]]] = 0
            scene_nocs.append(nocs)
            scene_normals.append(meta_unified["normals"])
            scene_pcs.append(pc[rand_nocs_idx])
            # breakpoint()

        # randomly sample an object
        rand_idx_obj = np.random.randint(low=0, high=len(scene_nocs))
        # processed_mask = (mask == objects_seg_id[object_info_keys[rand_idx_obj][:-6]]).astype(int)
        return {
            "rgb": vF.to_tensor(rgb),
            "mask": default_convert(mask.astype(np.int32)),
            "obs": default_convert(obs),
            "sdf": default_convert(scene_sdfs[rand_idx_obj]).float(),
            "coords": default_convert(scene_coords[rand_idx_obj]).float(),
            "nocs": default_convert(scene_nocs[rand_idx_obj]).float(),
            "normals": default_convert(scene_normals[rand_idx_obj]).float(),
            "chosen_idx": rand_idx_obj,
            "intrinsics": default_convert(self.intrinsics_mat),
            "depth": default_convert(depth).float(),
            "pc": default_convert(scene_pcs[rand_idx_obj].T).float(),
            "T_cam": default_convert(cam_T).float(),
            "metadata": {"dataset_name": "replicacad", "obj_name": objects[rand_idx_obj]["label"]},
        }


class ReplicaCADSceneDataset(ReplicaCADDataset):
    """ReplicaCAD dataset class that allows for selecting specific scenes"""

    def collate_fn(self, batch):
        """Custom collate function to handle the per frame objects"""
        num_imgs = len(batch)
        rgbs = torch.zeros(num_imgs, 3, batch[0][0].shape[0], batch[0][0].shape[1])
        masks = torch.zeros(num_imgs, 1, batch[0][1].shape[0], batch[0][1].shape[1])
        sdfs = torch.zeros(num_imgs, 7000, 1)
        nocs = torch.zeros(num_imgs, 3, 5000)
        pcs = torch.zeros(num_imgs, 3, 5000)
        coords = torch.zeros(num_imgs, batch[0]["coords"].shape[0], batch[0]["coords"].shape[1])
        normals = torch.zeros(num_imgs, batch[0]["normals"].shape[0], batch[0]["normals"].shape[1])
        depths = torch.zeros_like(masks)
        intrinsics = torch.zeros(num_imgs, 1, 3, 3)
        T_cams = torch.zeros(num_imgs, 1, 4, 4)
        metas = []
        chosens = []

        for i, b in enumerate(batch):
            rgbs[i, ...] = vF.to_tensor(b["rgb"])
            masks[i, ...] = default_convert(b["mask"].astype(np.int32))
            depths[i, ...] = default_convert(b["camera"]["depth"])
            intrinsics[i, ...] = default_convert(b["camera"]["K"])
            T_cams[i, ...] = default_convert(b["T_cam"])
            sdfs[i, ...] = default_convert(b["sdf"])
            nocs[i, ...] = default_convert(b["nocs"])
            coords[i, ...] = default_convert(b["coords"])
            normals[i, ...] = default_convert(b["normals"])
            pcs[i, ...] = default_convert(b["pc"])
            metas.append(b["metadata"][0])
            chosens.append(b["chosen_idx"])
        return {
            "rgb": rgbs,
            "mask": masks,
            "sdf": sdfs,
            "coords": coords,
            "nocs": nocs,
            "normals": normals,
            "intrinsics": intrinsics,
            "depth": depths,
            "pc": pcs,
            "T_cam": T_cams,
            "metadata": metas,
        }


