import h5py
import os
import json
import torch
import csv
import torchvision.transforms as transforms
import pyvista as pv
import numpy as np
import trimesh
import trimesh.sample
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from collections import namedtuple
import crisp.utils.geometry


class ShapeNetRender(Dataset):
    """Dataloder for the ShapeNet renders dataset.
    Expected format: a folder with .hdf5 files containing depth, RGB and potentially more data.
    Another csv file that contains the synset IDs and source IDs of the objects in the dataset.
    """

    def __init__(
        self,
        folder_path,
        shapenet_dataset_path,
        preload_to_mem=False,
        sample_surface_points_count=1000,
        sample_global_nonmnfld_points_count=1000,
        pc_size=10000,
        rgb_key="colors",
        normals_key="normals",
        depth_key="depth",
        distance_key="distance",
        segmap_key="instance_segmaps",
        state_key="shapenet_state",
        nocs_key="nocs",
        cam_states_key="cam_states",
        cam_intrinsics_key="cam_K",
        cam_pose_key="cam2world",
        data_to_output=None,
        debug_vis=False,
    ):
        assert os.path.isdir(folder_path)
        assert os.path.isdir(shapenet_dataset_path)

        self.folder_path = folder_path
        self.shapenet_dataset_path = shapenet_dataset_path
        self.preload_to_mem = preload_to_mem
        self.pc_size = pc_size
        self.sample_surface_points_count = sample_surface_points_count
        self.sample_global_nonmnfld_points_count = sample_global_nonmnfld_points_count
        # start of the global nonmnfld points (for loss functions)
        self.global_nonmnfld_points_start = self.sample_surface_points_count * 2
        if data_to_output is None:
            self.data_to_output = [
                "rgb",
                "nocs",
                "coords",
                "normals",
                "sdf",
                "obj_shapenet_id",
                "cam_intrinsics",
                "cam_pose",
            ]
        else:
            self.data_to_output = data_to_output
        print(f"ShapeNet dataset will output: {self.data_to_output}")

        (
            self.rgb_key,
            self.normals_key,
            self.depth_key,
            self.distance_key,
            self.segmap_key,
            self.state_key,
            self.nocs_key,
            self.cam_states_key,
            self.cam_intrinsics_key,
            self.cam_pose_key,
        ) = (
            rgb_key,
            normals_key,
            depth_key,
            distance_key,
            segmap_key,
            state_key,
            nocs_key,
            cam_states_key,
            cam_intrinsics_key,
            cam_pose_key,
        )

        # RGB pre-processing
        # note: transforms.ToTensor() will convert (H x W x C) [0-255] to (C x H x W) [0, 1]
        self.rgb_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        self.hdf5files = []
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith(".hdf5"):
                self.hdf5files.append(file)

        self.data = None
        if preload_to_mem:
            self.data = []
            for file in self.hdf5files:
                self.data.append(self._make_data_entry(file))

        # load model metadata and mesh
        self.distinct_objects = {}
        self.normalized_cube_center = [0, 0, 0]
        self.normalized_cube_scale = 1
        with open(os.path.join(self.folder_path, "objects_info.csv"), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                synset_id, source_id = row[0], row[1]
                model_path = os.path.join(
                    self.shapenet_dataset_path, f"{synset_id}/{source_id}/models/model_normalized.obj"
                )
                # force='mesh' makes sure trimesh loads everything into the same mesh
                # skip_texture=True because we don't need colors
                mesh = trimesh.load(model_path, force="mesh", skip_texture=True)
                coords, normals = utils.geometry.sample_pts_and_normals(mesh=mesh, count=pc_size, interp=False)
                coords, obj_metainfo = utils.geometry.normalize_points_to_cube(
                    coords,
                    centroid=mesh.centroid,
                    keep_aspect_ratio=True,
                    center_at=np.array(self.normalized_cube_center),
                    cube_scale=self.normalized_cube_scale,
                )
                if debug_vis:
                    print("Visualizing sampled coordinates and normals...")
                    pl = pv.Plotter(shape=(1, 1))
                    rand_idx = np.random.choice(coords.shape[0], size=5000)
                    pl.add_arrows(coords[rand_idx, :], normals[rand_idx, :], mag=0.05)
                    pl.show()

                # calculate sigmas for near-surface nonmanifold point generation (sigmas are in normalized frame)
                ptree = cKDTree(coords)
                sigma_set = []
                for p in np.array_split(coords, 100, axis=0):
                    d = ptree.query(p, 50 + 1)
                    sigma_set.append(d[0][:, -1])
                sigmas = np.concatenate(sigma_set)

                self.distinct_objects[(synset_id, source_id)] = {
                    "surface_points": torch.tensor(coords).float(),
                    "surface_points_k50_sigmas": sigmas,
                    "normals": torch.tensor(normals).float(),
                    "og_scale": obj_metainfo["og_scale"],
                    "og_center": obj_metainfo["og_center"],
                }

    def __len__(self):
        return len(self.hdf5files)

    def __getitem__(self, idx):
        # in each file, it has the following key
        # data['shapenet_state']
        # [{'used_synset_id': '03001627', 'used_source_id': '1a6f615e8b1b5ae4dbbc9440457e303e'},
        if not self.preload_to_mem:
            file = self.hdf5files[idx]
            data = self._make_data_entry(file)
        else:
            data = self.data[idx]

        # append object info
        data["obj_geom"] = self.distinct_objects[(data["state"]["used_synset_id"], data["state"]["used_source_id"])]

        # depending on our settings, we may change the output data
        output_data = {}
        rand_idcs = np.random.choice(self.pc_size, size=self.sample_surface_points_count)
        off_surface_samples = self.sample_surface_points_count + self.sample_global_nonmnfld_points_count
        total_samples = self.sample_surface_points_count * 2 + self.sample_global_nonmnfld_points_count

        if "rgb" in self.data_to_output:
            output_data["rgb"] = data["rgb"]
        if "nocs" in self.data_to_output:
            output_data["nocs"] = data["nocs"]
        if "coords" in self.data_to_output:
            # samples
            on_surface_coords = data["obj_geom"]["surface_points"][rand_idcs, :]

            # randomly sampled off surface points
            # 1. global samples
            off_surface_coords_global = np.random.uniform(
                -self.normalized_cube_scale / 2,
                self.normalized_cube_scale / 2,
                size=(self.sample_global_nonmnfld_points_count, 3),
            )
            off_surface_coords_global += np.array(self.normalized_cube_center)

            # 2. local samples (near mesh points but perturbed)
            local_sigma = data["obj_geom"]["surface_points_k50_sigmas"][rand_idcs]
            off_surface_coords_local = on_surface_coords + (
                np.random.randn(*on_surface_coords.shape) * local_sigma[:, None]
            )
            coords = np.concatenate((on_surface_coords, off_surface_coords_local, off_surface_coords_global), axis=0)
            output_data["coords"] = torch.tensor(coords).float()

        if "object_pc" in self.data_to_output:
            output_data["object_pc"] = data["obj_geom"]["surface_points"]

        if "depth" in self.data_to_output:
            # output depth map
            output_data["depth"] = data["depth"]

        if "distance" in self.data_to_output:
            output_data["distance"] = data["distance"]

        if "instance_segmap" in self.data_to_output:
            output_data["instance_segmap"] = data["instance_segmap"]

        if "normals" in self.data_to_output:
            # off surface normals are all -1
            on_surface_normals = data["obj_geom"]["normals"][rand_idcs, :]
            off_surface_normals = np.ones((off_surface_samples, 3)) * -1
            normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)
            output_data["normals"] = torch.tensor(normals).float()

        if "sdf" in self.data_to_output:
            sdf = np.zeros((total_samples, 1))  # on-surface = 0
            sdf[self.sample_surface_points_count :, :] = -1  # off-surface = -1
            output_data["sdf"] = torch.tensor(sdf).float()

        if "obj_shapenet_id" in self.data_to_output:
            output_data["obj_shapenet_id"] = (data["state"]["used_synset_id"], data["state"]["used_source_id"])

        if "cam_intrinsics" in self.data_to_output:
            output_data["cam_intrinsics"] = torch.tensor(np.array(data["intrinsics"])).float()

        if "cam_pose" in self.data_to_output:
            output_data["cam_pose"] = torch.tensor(np.array(data["cam_pose"])).float()

        return output_data

    def _make_data_entry(self, hdf5ile):
        """load a single HDF5 file"""
        # HDF5/blenderproc file dimension: (H, W, C)
        # PyTorch convention: (C, H, W)
        with h5py.File(os.path.join(self.folder_path, hdf5ile), "r") as f:
            data_dict = {
                "depth": torch.tensor(np.array(f[self.depth_key])),
                "rgb": self.rgb_transform(np.array(f[self.rgb_key])),
                "normal": transforms.functional.to_tensor(np.array(f[self.normals_key])),
                "distance": torch.tensor(np.array(f[self.distance_key])),
                "instance_segmap": torch.tensor(np.array(f[self.segmap_key])),
                # from (H, W, C) to (C, H, W)
                # last channel for nocs is the object mask
                "state": json.loads(f[self.state_key][()]),
                # load cam states and pose
                # cam_states -> cam2world, cam_K
                "intrinsics": json.loads(f[self.cam_states_key][()])[self.cam_intrinsics_key],
                "cam_pose": json.loads(f[self.cam_states_key][()])[self.cam_pose_key],
            }

            if "nocs" in self.data_to_output:
                data_dict.update({"nocs": torch.tensor(np.transpose(np.array(f[self.nocs_key]), (2, 0, 1))).float()})

            return data_dict
