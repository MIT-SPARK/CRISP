import math
import logging
import h5py
import os
import multiprocessing
from multiprocessing import Pool, Process, Manager
import concurrent.futures
import copy
import json
import torch
import time
import pickle
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from collections import defaultdict
import csv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pyvista as pv
import numpy as np
import trimesh
import trimesh.sample
import yaml
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler, DistributedSampler
from scipy.spatial import cKDTree
from collections import namedtuple
from tqdm import tqdm
import crisp.utils.geometry
import crisp.utils.math
from crisp.utils.constants import *
from crisp.utils.file_utils import safely_make_folders
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista


def unified_objects_collate_fn(data):
    """Use this collate function if you want to output normalized meshes"""
    collated_batch = {}
    # Iterate over each key-value in the dictionary
    for key in data[0]:
        # Gather the data from the batch for the current key
        values = [item[key] for item in data]

        # Check if the key's values are tensors
        if isinstance(values[0], torch.Tensor):
            # If the values are tensors, stack them into a single tensor
            collated_batch[key] = torch.stack(values, dim=0)
        else:
            collated_batch[key] = values

    return collated_batch


def cam2cad_transformation_from(
    cam_s_recons, cam_R_recons, cam_t_recons, recons_s_blender, recons_t_blender, blender_R_cad, blender_s_cad
):
    """Helper function for calculating cam_R_cad, cam_t_cad, cam_s_cad"""

    cam_s_cad = cam_s_recons * recons_s_blender * blender_s_cad

    # recons_R_blender = torch.eye(3) so we skip
    cam_R_cad = cam_R_recons @ blender_R_cad

    # blender_t_cad = 0 so we skip its term
    cam_t_cad = cam_s_recons * cam_R_recons @ recons_t_blender.reshape(3, 1) + cam_t_recons.reshape(3, 1)

    return cam_s_cad, cam_R_cad, cam_t_cad


def normalize_mesh(mesh: trimesh.Trimesh, obj_meta: dict):
    """Helper function to normalize the trimesh mesh"""
    # center and normalize according to the logged data
    blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender, cad_s_nonmetric_cad = (
        np.array(obj_meta["blender_s_cad"]),
        np.array(obj_meta["blender_R_cad"]),
        np.array(obj_meta["recons_t_blender"]),
        obj_meta["recons_s_blender"],
        obj_meta["cad_s_nonmetric_cad"],
    )
    # cad_s_nonmetric_cad: scaling factor between the CAD model (non-metric if loading NOCS camera objects)
    # and the metric CAD model.
    # The CAD frame for the other transformation parameter assume metric scale, i.e., CAD == metric CAD.
    mesh_verts = mesh.vertices
    blender_coords = blender_s_cad.reshape((1, 3)) * cad_s_nonmetric_cad * mesh_verts @ blender_R_cad.T
    recons_coords = recons_s_blender * blender_coords + recons_t_blender.reshape((1, 3))
    mesh_new = mesh.copy()
    mesh_new.vertices = recons_coords
    return mesh_new


def unified_objects_obj_sdf_cache_fname(obj_name, voxel_res):
    """Helper function for determining the name of the object cache"""
    return f"{obj_name}_voxres={voxel_res}.pkl"


def mesh_loading_worker(x):
    obj_name, obj_path, obj_meta = x
    mesh = trimesh.load(obj_path, skip_materials=True, process=False)
    return obj_name, mesh, obj_meta


def cad2recons(coords, blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender, cad_s_nonmetric_cad):
    blender_coords = blender_s_cad.reshape((1, 3)) * cad_s_nonmetric_cad * coords @ blender_R_cad.T
    recons_coords = recons_s_blender * blender_coords + recons_t_blender.reshape((1, 3))
    return recons_coords


def process_individual_object(
    dataset_name,
    obj_name,
    mesh,
    pc_size,
    sample_bounds,
    global_nonmnfld_points_voxel_res,
    obj_meta,
    debug_vis,
    compute_sdf=True,
):
    try:
        coords, normals = crisp.utils.geometry.sample_pts_and_normals(mesh=mesh, count=pc_size, interp=False)
    except:
        print("Error encountered")
        return None

    # center and normalize according to the logged data
    blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender = (
        np.array(obj_meta["blender_s_cad"]),
        np.array(obj_meta["blender_R_cad"]),
        np.array(obj_meta["recons_t_blender"]),
        obj_meta["recons_s_blender"],
    )

    # for NOCS camera objects, the CAD model is not metric
    if "cad_s_nonmetric_cad" not in obj_meta.keys():
        obj_meta["cad_s_nonmetric_cad"] = 1.0
    cad_s_nonmetric_cad = obj_meta["cad_s_nonmetric_cad"]

    normalized_mesh = normalize_mesh(mesh, obj_meta)

    if not compute_sdf:
        # for the case where we are computing everything online
        result = {
            "dataset_name": dataset_name,
            "obj_name": obj_name,
            "mesh": mesh,
            "normalized_mesh": normalized_mesh,
            "processed_obj_meta": obj_meta,
        }
        return result

    # these should bring models from CAD frame to blender frame (should be consistent with NOCS)
    recons_coords = cad2recons(
        coords, blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender, cad_s_nonmetric_cad
    )
    normals = normals @ blender_R_cad.T
    normals /= np.linalg.norm(normals, axis=1).reshape((-1, 1))

    if debug_vis:
        print("Visualizing sampled coordinates and normals...")
        pl = pv.Plotter(shape=(1, 1))
        rand_idx = np.random.choice(recons_coords.shape[0], size=5000)
        pl.add_points(recons_coords[rand_idx, :])
        pl.add_arrows(recons_coords[rand_idx, :], normals[rand_idx, :], mag=0.1)
        pl.show()

    # calculate sigmas for near-surface nonmanifold point generation (sigmas are in normalized frame)
    ptree = cKDTree(recons_coords)
    sigma_set = []
    for p in np.array_split(recons_coords, 100, axis=0):
        d = ptree.query(p, 50 + 1)
        sigma_set.append(d[0][:, -1])
    sigmas = np.concatenate(sigma_set)

    cube_scale = sample_bounds[1] - sample_bounds[0]
    assert cube_scale > 0

    # randomly sampled off surface points
    # 1. global samples (the points should be within [-1.0, 1.0] if self.normalized_recons is True)
    off_surface_coords_global, voxel_size, voxel_origin = crisp.utils.geometry.voxelize_cube(
        N=global_nonmnfld_points_voxel_res, cube_center=np.array([0, 0, 0]), cube_scale=cube_scale
    )

    # 2. local samples (near mesh points but perturbed)
    off_surface_coords_local = recons_coords + (np.random.randn(*recons_coords.shape) * sigmas[:, None])

    result = {
        "dataset_name": dataset_name,
        "obj_name": obj_name,
        "normalized_mesh": normalized_mesh,
        # on surface points
        "surface_points": torch.tensor(recons_coords).float(),
        "surface_points_k50_sigmas": sigmas,
        "normals": torch.tensor(normals).float(),
        # locally perturbed points
        "nonmnfld_coords_local": torch.tensor(off_surface_coords_local).float(),
        # global off surface points
        "nonmnfld_coords_global": torch.tensor(off_surface_coords_global).float(),
    }

    # compute SDFs
    # this uses fast winding numbers for inside/outside estimation which should be pretty robust
    local_nz_sdf_values = crisp.utils.geometry.query_sdf_from_mesh(off_surface_coords_local, normalized_mesh)
    global_nz_sdf_values = crisp.utils.geometry.query_sdf_from_mesh(off_surface_coords_global, normalized_mesh)
    result["nonmnfld_coords_local_sdf"] = torch.tensor(local_nz_sdf_values).float()
    result["nonmnfld_coords_global_sdf"] = torch.tensor(global_nz_sdf_values).float()

    return result


def process_individual_object_and_cache(x):
    """Picklable function for processing individual object.
    To enable multiprocessing with Python to speed up dataloading.
    """
    (
        dataset_name,
        obj_name,
        mesh,
        pc_size,
        sample_bounds,
        global_nonmnfld_points_voxel_res,
        obj_meta,
        cache_folder_path,
        compute_sdf,
        generate_cache,
        debug_vis,
    ) = x
    result = process_individual_object(
        dataset_name=dataset_name,
        obj_name=obj_name,
        mesh=mesh,
        pc_size=pc_size,
        sample_bounds=sample_bounds,
        global_nonmnfld_points_voxel_res=global_nonmnfld_points_voxel_res,
        obj_meta=obj_meta,
        debug_vis=debug_vis,
        compute_sdf=compute_sdf,
    )

    # save object cache
    if generate_cache:
        safely_make_folders([cache_folder_path])
        fname = unified_objects_obj_sdf_cache_fname(obj_name, global_nonmnfld_points_voxel_res)
        cache_path = os.path.join(cache_folder_path, fname)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    return result


class UnifiedObjects(Dataset):
    """Dataloader for training the unified object model.
    Currently supports:
    1. YCBV objects
    2. ReplicaCAD objects
    3. ShapeNet objects
    """

    def __init__(
        self,
        folder_path,
        shapenet_dataset_path=None,
        bop_dataset_path=None,
        replicacad_dataset_path=None,
        spe3r_dataset_path=None,
        nocs_objs_dataset_path=None,
        uhumans_objs_dataset_path=None,
        preload_to_mem=False,
        preload_sdf_to_mem=True,
        generate_obj_cache=True,
        sample_surface_points_count=1000,
        sample_local_nonmnfld_points_count=1000,
        sample_global_nonmnfld_points_count=5000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=None,
        pc_size=10000,
        rgb_key="colors",
        normals_key="normals",
        depth_key="depth",
        distance_key="distance",
        segmap_key="instance_segmaps",
        metadata_key="metadata",
        nocs_key="nocs",
        cam_states_key="cam_states",
        cam_intrinsics_key="cam_K",
        cam_pose_key="cam2world",
        split="train",
        data_to_output=None,
        num_workers=0,
        normalized_recons=True,
        force_recompute_sdf=False,
        debug_vis=True,
    ):
        assert os.path.isdir(folder_path)
        (
            self.shapenet_dataset_path,
            self.bop_dataset_path,
            self.replicacad_dataset_path,
            self.spe3r_dataset_path,
            self.nocs_objs_dataset_path,
            self.uhumans_objs_dataset_path,
        ) = (
            shapenet_dataset_path,
            bop_dataset_path,
            replicacad_dataset_path,
            spe3r_dataset_path,
            nocs_objs_dataset_path,
            uhumans_objs_dataset_path,
        )
        self._check_dataset_paths()

        self.folder_path = folder_path
        self.preload_to_mem = preload_to_mem
        self.num_workers = num_workers
        self.force_recompute_sdf = force_recompute_sdf
        self.normalized_recons = normalized_recons
        self.pc_size = pc_size
        self.preload_sdf_to_mem = preload_sdf_to_mem
        self.generate_obj_cache = generate_obj_cache

        # size of the voxel grid for generating nonmnfld SDF values
        self.global_nonmnfld_points_voxel_res = global_nonmnfld_points_voxel_res
        self.global_nonmnfld_points_count = global_nonmnfld_points_voxel_res**3

        # number of points to sample in each batch
        self.sample_surface_points_count = sample_surface_points_count
        self.sample_local_nonmnfld_points_count = sample_local_nonmnfld_points_count
        self.sample_global_nonmnfld_points_count = sample_global_nonmnfld_points_count
        assert self.sample_surface_points_count < self.pc_size
        assert self.sample_local_nonmnfld_points_count < self.pc_size

        if sample_bounds is None:
            sample_bounds = (-1.0, 1.0)
        assert sample_bounds[0] < sample_bounds[1]
        self.sample_bounds = sample_bounds

        # start of the global nonmnfld points (for loss functions)
        self.global_nonmnfld_points_start = self.sample_surface_points_count * 2
        if data_to_output is None:
            self.data_to_output = ["rgb", "nocs", "coords", "normals", "sdf", "cam_intrinsics", "cam_pose"]
        else:
            self.data_to_output = data_to_output
        print(f"Unified objects dataset will output: {self.data_to_output}")

        (
            self.rgb_key,
            self.normals_key,
            self.depth_key,
            self.distance_key,
            self.segmap_key,
            self.metadata_key,
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
            metadata_key,
            nocs_key,
            cam_states_key,
            cam_intrinsics_key,
            cam_pose_key,
        )

        # RGB pre-processing
        # note: transforms.ToTensor() will convert (H x W x C) [0-255] to (C x H x W) [0, 1]
        self.rgb_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
        )

        self.hdf5files = []
        for file in sorted(os.listdir(self.folder_path)):
            if file.endswith(".hdf5"):
                self.hdf5files.append(file)

        self.data = None
        self._used_obj_names = set()
        self._get_obj_names_in_dataset()

        if preload_to_mem:
            self.data = []
            for file in self.hdf5files:
                self.data.append(self._make_data_entry(file))

        # load model metadata and mesh
        self.distinct_objects = {"ycbv": {}, "replicacad": {}, "shapenet": {}, "spe3r": {}, "nocs": {}}

        self._nocs_post_process, self._pose_post_process = {}, {}
        with open(os.path.join(self.folder_path, "objects_info.yaml"), "r") as f:
            print(f"Loading {os.path.join(self.folder_path, 'objects_info.yaml')}")
            self.objects_info = yaml.load(f, Loader=yaml.CLoader)

            for dataset in self.objects_info.keys():
                self._prepare_output_preprocess(dataset)
                if dataset == "ycbv":
                    self._load_ycbv_objects(objects_data=self.objects_info["ycbv"], debug_vis=debug_vis)
                elif dataset == "replicacad":
                    self._load_replicacad_objects(objects_data=self.objects_info["replicacad"], debug_vis=debug_vis)
                elif dataset == "spe3r":
                    self._load_spe3r_objects(objects_data=self.objects_info["spe3r"], debug_vis=debug_vis)
                elif dataset == "nocs_real_test":
                    self._load_nocs_real_objects(
                        objects_data=self.objects_info["nocs_real_test"], split="test", debug_vis=debug_vis
                    )
                elif dataset == "nocs_real_train":
                    self._load_nocs_real_objects(
                        objects_data=self.objects_info["nocs_real_train"], split="train", debug_vis=debug_vis
                    )
                elif dataset == "nocs_camera_train":
                    self._load_nocs_camera_objects(
                        objects_data=self.objects_info["nocs_camera_train"], split="train", debug_vis=debug_vis
                    )
                elif dataset == "nocs_camera_val":
                    self._load_nocs_camera_objects(
                        objects_data=self.objects_info["nocs_camera_val"], split="val", debug_vis=debug_vis
                    )
                elif dataset == "nocs":
                    raise NotImplementedError(
                        "Unsupported dataset: nocs. Need to use nocs_camera_SPLIT or nocs_real_SPLIT."
                    )
                elif dataset == "shapenet":
                    self._load_shapenet_objects(objects_data=self.objects_info["shapenet"], debug_vis=debug_vis)
                elif dataset == "uhumans":
                    self._load_uhumans_objects(objects_data=self.objects_info["uhumans"], debug_vis=debug_vis)
                else:
                    raise ValueError(f"Unrecognized dataset {dataset} in objects_info.yaml.")

                # TODO: Sample bounds for NOCS camera objects are not set correctly.
                print(f"Setting sample bounds for {dataset} to: {self.sample_bounds}")

    def _prepare_output_preprocess(self, dataset):
        if (
            dataset == "ycbv"
            or dataset == "spe3r"
            or dataset == "nocs_real_train"
            or dataset == "nocs_real_test"
            or dataset == "nocs_camera_train"
            or dataset == "nocs_camera_val"
            or dataset == "shapenet"
            or dataset == "uhumans"
        ):
            # handle normalization issues for reconstruction frame
            if self.normalized_recons:
                self._nocs_post_process[dataset] = lambda x: torch.tensor(x["nocs"])
                self._pose_post_process[dataset] = lambda x: torch.tensor(x["pose"])
            else:
                # set sample bounds
                # NOTE: This is not exact if recons_t_blender is not zero
                # in practice it's pretty small for SPE3R; zero for other datasets
                max_bnd = 0
                for obj_name, obj_meta in self.objects_info[dataset].items():
                    max_bnd = max(max_bnd, 1 / obj_meta["recons_s_blender"])
                self.sample_bounds = (-max_bnd, max_bnd)

                # NOCS are normalized, so we need to unnormalize them
                # assume p_recons are already centered
                # p^recons = 2.0 *( NOCS - 0.5)
                # p^recons_un = 1/(recons_s_blender) * p^recons
                def nocs_fn(x):
                    return (2.0 * (x["nocs"] - 0.5)) / x["recons_s_blender"]

                self._nocs_post_process[dataset] = nocs_fn

                # pose is recons_un_T_cam
                def pose_fn(x):
                    recons_un_T_recons = torch.zeros((4, 4))
                    recons_un_T_recons[3, 3] = 1
                    recons_un_T_recons[:3, :3] = torch.diag(
                        torch.tensor([1 / x["recons_s_blender"], 1 / x["recons_s_blender"], 1 / x["recons_s_blender"]])
                    )
                    # x["pose"] is recons_T_cam
                    processed_pose = recons_un_T_recons @ torch.tensor(x["pose"]).float()
                    # remove the applied scaling on translation
                    processed_pose[:3, 3] *= x["recons_s_blender"]
                    return processed_pose

                self._pose_post_process[dataset] = pose_fn
        else:
            raise NotImplementedError

    def _load_nocs_camera_objects(self, objects_data, split="train", debug_vis=False):
        """Load NOCS Real objects"""
        inputs = []
        if self.num_workers == 0:
            max_workers = int(multiprocessing.cpu_count() / 2) + 1
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False
        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading NOCS Camera objects"):
            scenename = obj_name.split("_")[0]
            oname = obj_name.split("_")[1]
            obj_path = os.path.join(
                self.nocs_objs_dataset_path, f"camera_{split}", f"{scenename}", f"{oname}", "model.obj"
            )

            mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)

            if isinstance(mesh, list) and len(mesh) == 0:
                print(f"Skipping {obj_name} because we can't load it.")
                continue

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # if we turn off normalization, the reconstruction frame is the same scale as the metric CAD frame
                # we set recons_s_blender to 1 so the normalized coords are the original scale
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    f"nocs_camera_{split}",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"nocs_camera_{split}"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_nocs_real_objects(self, objects_data, split="train", debug_vis=False):
        """Load NOCS Real objects"""
        inputs = []
        if self.num_workers == 0:
            max_workers = int(multiprocessing.cpu_count() / 2) + 1
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading NOCS Real objects"):
            obj_path = os.path.join(self.nocs_objs_dataset_path, f"real_{split}", f"{obj_name}.obj")
            mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # if we turn off normalization, the reconstruction frame is the same scale as the metric CAD frame
                # we set recons_s_blender to 1 so the normalized coords are the original scale
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    f"nocs_real_{split}",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"nocs_real_{split}"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_uhumans_objects(self, objects_data, debug_vis=False):
        """Load uHumans objects"""
        inputs = []
        if self.num_workers == 0:
            max_workers = max(int(multiprocessing.cpu_count() / 1.5), 1)
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading uHumans objects"):
            if ("uhumans", obj_name) not in self._used_obj_names:
                continue
            split, bare_obj_name = obj_name.split("_")[0], "_".join(obj_name.split("_")[1:])
            obj_path = os.path.join(self.uhumans_objs_dataset_path, "meshes", split, f"{bare_obj_name}.obj")
            mesh = trimesh.load(obj_path, force="mesh", process=False, skip_materials=True)

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # for ycbv, the depths are in meters, the objects are in mm
                # so we still apply blender_s_cad (0.001) to the mesh
                # but we set recons_s_blender to 1 so the normalized coords are the original scale (in m)
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    "uhumans",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"uhumans"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_shapenet_objects(self, objects_data, debug_vis=False):
        """Load shapenet objects"""
        inputs = []
        if self.num_workers == 0:
            max_workers = max(int(multiprocessing.cpu_count() / 1.5), 1)
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading ShapeNet objects"):
            if ("shapenet", obj_name) not in self._used_obj_names:
                continue
            synset_id, source_id = obj_name.split("_")
            obj_path = os.path.join(self.shapenet_dataset_path, synset_id, source_id, "models", "model_normalized.obj")
            mesh = trimesh.load(obj_path, force="mesh", process=False, skip_materials=True)

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # for ycbv, the depths are in meters, the objects are in mm
                # so we still apply blender_s_cad (0.001) to the mesh
                # but we set recons_s_blender to 1 so the normalized coords are the original scale (in m)
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    "shapenet",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"shapenet"),
                    False,  # preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_spe3r_objects(self, objects_data, debug_vis=False):
        """Load SPE3R objects (satellites)"""
        inputs = []
        if self.num_workers == 0:
            max_workers = int(multiprocessing.cpu_count() / 2) + 1
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading SPE3R objects"):
            obj_path = os.path.join(self.spe3r_dataset_path, obj_name, "models", "model_normalized.obj")
            mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # for ycbv, the depths are in meters, the objects are in mm
                # so we still apply blender_s_cad (0.001) to the mesh
                # but we set recons_s_blender to 1 so the normalized coords are the original scale (in m)
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    "spe3r",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"spe3r"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_replicacad_objects(self, objects_data, debug_vis=False):
        """Load ReplicaCAD objects"""
        inputs = []
        if self.num_workers == 0:
            max_workers = int(multiprocessing.cpu_count() / 2) + 1
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading ReplicaCAD objects"):
            config_path = os.path.join(
                self.replicacad_dataset_path, "configs", "objects", f"{obj_name}.object_config.json"
            )
            with open(config_path, "r") as config_f:
                config_data = json.load(config_f)
            asset_path = os.path.abspath(
                os.path.join(self.replicacad_dataset_path, "objects", os.path.basename(config_data["render_asset"]))
            )
            scene = trimesh.load_mesh(asset_path, force="mesh", skip_materials=True)

            # note: the trimesh dump applies the scaling in the glb file
            # which is why we need to reset the scale to 1.0 here
            mesh = scene.dump(concatenate=True)
            obj_meta["blender_s_cad"] = (1, 1, 1)

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    "replicacad",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta,
                    self._obj_sdf_cache_folder(f"replicacad"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_ycbv_objects(self, objects_data, debug_vis=False):
        inputs = []
        if self.num_workers == 0:
            max_workers = int(multiprocessing.cpu_count() / 2) + 1
        else:
            max_workers = self.num_workers
        if max_workers > 1 and debug_vis:
            print("Setting debug_vis to False for multiprocessing.")
            debug_vis = False

        for obj_name, obj_meta in tqdm(objects_data.items(), desc="Loading YCBV objects meshes"):
            model_path = os.path.join(self.bop_dataset_path, "ycbv", "models", f"{obj_name}.ply")
            # force='mesh' makes sure trimesh loads everything into the same mesh
            # skip_materials=True because we don't need colors
            mesh = trimesh.load(model_path, force="mesh", skip_materials=True)

            obj_meta_to_use = copy.deepcopy(obj_meta)
            if not self.normalized_recons:
                # for ycbv, the depths are in meters, the objects are in mm
                # so we still apply blender_s_cad (0.001) to the mesh
                # but we set recons_s_blender to 1 so the normalized coords are the original scale (in m)
                og_recons_s_blender = obj_meta_to_use["recons_s_blender"]
                obj_meta_to_use["recons_s_blender"] = 1
                obj_meta_to_use["recons_t_blender"] = (
                    np.array(obj_meta_to_use["recons_t_blender"]) / og_recons_s_blender
                ).tolist()

            # make sure this is consistent with the order of process_individual_object
            inputs.append(
                (
                    "ycbv",
                    obj_name,
                    mesh,
                    self.pc_size,
                    self.sample_bounds,
                    self.global_nonmnfld_points_voxel_res,
                    obj_meta_to_use,
                    self._obj_sdf_cache_folder(f"ycbv"),
                    self.preload_sdf_to_mem,
                    self.generate_obj_cache,
                    debug_vis,
                )
            )

        self._multiprocess_objs_loading_helper(inputs=inputs, max_workers=max_workers)

    def _load_obj_mesh(self, obj_name, dataset):
        """ " Dataset-specific mesh loading logic"""
        if dataset == "ycbv":
            model_path = os.path.join(self.bop_dataset_path, "ycbv", "models", f"{obj_name}.ply")
            mesh = trimesh.load(model_path, force="mesh", skip_materials=True)
        elif dataset == "spe3r":
            obj_path = os.path.join(self.spe3r_dataset_path, obj_name, "models", "model_normalized.obj")
            mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)
        elif dataset == "shapenet":
            synset_id, source_id = obj_name.split("_")
            obj_path = os.path.join(self.shapenet_dataset_path, synset_id, source_id, "models", "model_normalized.obj")
            mesh = trimesh.load(obj_path, force="mesh", skip_materials=True)
        else:
            raise NotImplementedError
        return mesh

    def _obj_sdf_cache_folder(self, dataset_name):
        if self.preload_sdf_to_mem:
            if self.normalized_recons:
                return os.path.join(self.folder_path, "sdf_cache", dataset_name)
            else:
                return os.path.join(self.folder_path, "unnormalized_sdf_cache", dataset_name)
        else:
            # no SDFs are generated; reflect this in the cache folder names
            if self.normalized_recons:
                return os.path.join(self.folder_path, "obj_cache", dataset_name)
            else:
                return os.path.join(self.folder_path, "unnormalized_obj_cache", dataset_name)

    def _check_sdf_cache(self, inputs):
        """Check whether object SDF cache exists"""
        for x in inputs:
            # x[0] -> dataset_name, x[1] -> obj_name
            folder = self._obj_sdf_cache_folder(x[0])
            fname = unified_objects_obj_sdf_cache_fname(x[1], self.global_nonmnfld_points_voxel_res)
            if not os.path.exists(os.path.join(folder, fname)):
                return False
        return True

    def _save_sdf_caches(self, results):
        for x in tqdm(results):
            dataset_name, obj_name = x["dataset_name"], x["obj_name"]
            folder = self._obj_sdf_cache_folder(dataset_name)
            safely_make_folders([folder])
            fname = unified_objects_obj_sdf_cache_fname(obj_name, self.global_nonmnfld_points_voxel_res)
            cache_path = os.path.join(folder, fname)
            with open(cache_path, "wb") as f:
                pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

    def _load_sdf_caches(self, inputs):
        results = []
        for x in inputs:
            # x[0] -> dataset_name, x[1] -> obj_name
            folder = self._obj_sdf_cache_folder(x[0])
            fname = unified_objects_obj_sdf_cache_fname(x[1], self.global_nonmnfld_points_voxel_res)
            cache_path = os.path.join(folder, fname)
            if not os.path.exists(os.path.join(folder, fname)):
                return None
            else:
                try:
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                except RuntimeError as e:
                    e.add_note(f"SDF cache path is: {cache_path}")
                    raise e
                results.append(data)
        return results

    def _multiprocess_objs_loading_helper(self, inputs, max_workers):
        # check cache
        results = None
        if self._check_sdf_cache(inputs) and not self.force_recompute_sdf:
            # load cache
            print("Object caches found. Loading ...")
            try:
                results = self._load_sdf_caches(inputs)
            except RuntimeError as e:
                print(f"Encountered error while loading SDF caches, shown below:")
                print(e)

        if results is None:
            logging.info("Object caches not found/failed to load. Computing SDFs...")
            start = time.time()
            logging.info(f"Using {max_workers} threads to process objects.")

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(process_individual_object_and_cache, inputs), total=len(inputs)))
            results = [r for r in results if r is not None]
            end = time.time()
            print(f"Total object processing time: {end - start} s")

            # save cache
            if self.generate_obj_cache:
                print(f"Caching the SDF results.")
                self._save_sdf_caches(results)

        #results = [process_individual_object_and_cache(x) for x in inputs]
        for x in results:
            dataset_name, obj_name = x["dataset_name"], x["obj_name"]
            if dataset_name in self.distinct_objects.keys():
                self.distinct_objects[dataset_name][obj_name] = x
            else:
                self.distinct_objects[dataset_name] = {obj_name: x}

    def __len__(self):
        return len(self.hdf5files)

    def _getitem_from_mem_objs(self, data):
        # append object info
        dataset, obj_name = (data["metadata"]["dataset_name"], data["metadata"]["obj_name"])
        data["obj_geom"] = self.distinct_objects[dataset][obj_name]

        # depending on our settings, we may change the output data
        output_data = {}

        # get rand indices
        surface_rand_idxs = np.random.choice(
            data["obj_geom"]["surface_points"].shape[0], size=self.sample_surface_points_count
        )
        local_nonmnfld_rand_idxs = np.random.choice(
            data["obj_geom"]["nonmnfld_coords_local"].shape[0], size=self.sample_local_nonmnfld_points_count
        )
        global_nonmnfld_rand_idxs = np.random.choice(
            data["obj_geom"]["nonmnfld_coords_global"].shape[0], size=self.sample_global_nonmnfld_points_count
        )

        if "rgb" in self.data_to_output:
            output_data["rgb"] = data["rgb"]

        if "nocs" in self.data_to_output:
            output_data["nocs"] = data["nocs"]
            output_data["nocs"][:3, ...] = self._nocs_post_process[dataset](
                {
                    "nocs": data["nocs"][:3, ...],
                    "recons_t_blender": torch.tensor(self.objects_info[dataset][obj_name]["recons_t_blender"]),
                    "recons_s_blender": self.objects_info[dataset][obj_name]["recons_s_blender"],
                }
            )

        if "coords" in self.data_to_output:
            on_surface_coords = data["obj_geom"]["surface_points"][surface_rand_idxs, :]
            nonmnfld_coords_local = data["obj_geom"]["nonmnfld_coords_local"][local_nonmnfld_rand_idxs, :]
            nonmnfld_coords_global = data["obj_geom"]["nonmnfld_coords_global"][global_nonmnfld_rand_idxs, :]
            coords = torch.cat((on_surface_coords, nonmnfld_coords_local, nonmnfld_coords_global), dim=0)
            output_data["coords"] = coords

        if "object_pc" in self.data_to_output:
            output_data["object_pc"] = data["obj_geom"]["surface_points"]

        if "normalized_mesh" in self.data_to_output:
            output_data["normalized_mesh"] = [
                torch.tensor(data["obj_geom"]["normalized_mesh"].vertices).float(),
                torch.tensor(data["obj_geom"]["normalized_mesh"].faces).float(),
            ]

        if "depth" in self.data_to_output:
            # output depth map
            output_data["depth"] = data["depth"]
            output_data["depth"][data["depth"] > 5000] = 0

        if "distance" in self.data_to_output:
            output_data["distance"] = data["distance"]

        if "instance_segmap" in self.data_to_output:
            output_data["instance_segmap"] = data["instance_segmap"]

        if "normals" in self.data_to_output:
            # off surface normals are all -1
            output_data["normals"] = data["obj_geom"]["normals"]

        if "sdf" in self.data_to_output:
            sdf = torch.zeros(
                self.sample_surface_points_count
                + self.sample_local_nonmnfld_points_count
                + self.sample_global_nonmnfld_points_count
            )
            # retrieve sdf from stored values
            nonmnfld_sdf_local = data["obj_geom"]["nonmnfld_coords_local_sdf"][local_nonmnfld_rand_idxs]
            sdf[
                self.sample_surface_points_count : self.sample_surface_points_count
                + self.sample_local_nonmnfld_points_count
            ] = nonmnfld_sdf_local

            nonmnfld_sdf_global = data["obj_geom"]["nonmnfld_coords_global_sdf"][global_nonmnfld_rand_idxs]
            sdf[self.sample_surface_points_count + self.sample_local_nonmnfld_points_count :] = nonmnfld_sdf_global

            output_data["sdf"] = sdf

        if "sdf_grid" in self.data_to_output:
            output_data["sdf_grid"] = data["obj_geom"]["nonmnfld_coords_global_sdf"]

        if "metadata" in self.data_to_output:
            output_data["metadata"] = data["metadata"]

        if "cam_intrinsics" in self.data_to_output:
            output_data["cam_intrinsics"] = torch.tensor(np.array(data["intrinsics"])).float()

        if "cam_pose" in self.data_to_output:
            processed_pose = self._pose_post_process[dataset](
                {
                    "pose": np.array(data["cam_pose"]),
                    "recons_t_blender": torch.tensor(self.objects_info[dataset][obj_name]["recons_t_blender"]),
                    "recons_s_blender": self.objects_info[dataset][obj_name]["recons_s_blender"],
                }
            )
            output_data["cam_pose"] = processed_pose.float()

        return output_data

    def _getitem_online(self, data):
        """Compute relevant data for a single object online"""
        # append object info
        dataset, obj_name = (data["metadata"]["dataset_name"], data["metadata"]["obj_name"])
        data["obj_geom"] = self.distinct_objects[dataset][obj_name]
        processed_obj_meta = data["obj_geom"]["processed_obj_meta"]
        mesh = data["obj_geom"]["mesh"]
        normalized_mesh = data["obj_geom"]["normalized_mesh"]
        blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender, cad_s_nonmetric_cad = (
            np.array(processed_obj_meta["blender_s_cad"]),
            np.array(processed_obj_meta["blender_R_cad"]),
            np.array(processed_obj_meta["recons_t_blender"]),
            processed_obj_meta["recons_s_blender"],
            processed_obj_meta["cad_s_nonmetric_cad"],
        )

        output_data = {}
        if "rgb" in self.data_to_output:
            output_data["rgb"] = data["rgb"]

        if "nocs" in self.data_to_output:
            output_data["nocs"] = data["nocs"]
            output_data["nocs"][:3, ...] = self._nocs_post_process[dataset](
                {
                    "nocs": data["nocs"][:3, ...],
                    "recons_t_blender": torch.tensor(self.objects_info[dataset][obj_name]["recons_t_blender"]),
                    "recons_s_blender": self.objects_info[dataset][obj_name]["recons_s_blender"],
                }
            )

        coords_numpy, on_surface_coords, nonmnfld_coords_local, nonmnfld_coords_global = None, None, None, None
        if "coords" in self.data_to_output:
            # sample recons_coords/surface points
            surface_points = crisp.utils.geometry.sample_pts(mesh=mesh, count=self.sample_surface_points_count)
            on_surface_coords = cad2recons(
                surface_points,
                blender_s_cad,
                blender_R_cad,
                recons_t_blender,
                recons_s_blender,
                cad_s_nonmetric_cad,
            )

            # perturbed local points
            nonmnfld_coords_local = on_surface_coords + (np.random.randn(*on_surface_coords.shape) * 0.01)

            # sampled global points
            nonmnfld_coords_global = np.random.uniform(
                low=self.sample_bounds[0],
                high=self.sample_bounds[1],
                size=(self.sample_global_nonmnfld_points_count, 3),
            )

            coords_numpy = np.concatenate((on_surface_coords, nonmnfld_coords_local, nonmnfld_coords_global), axis=0)
            coords = torch.tensor(coords_numpy).float()
            output_data["coords"] = coords

        if "object_pc" in self.data_to_output:
            surface_points = crisp.utils.geometry.sample_pts(mesh=mesh, count=self.sample_surface_points_count)
            on_surface_coords = cad2recons(
                surface_points,
                blender_s_cad,
                blender_R_cad,
                recons_t_blender,
                recons_s_blender,
                cad_s_nonmetric_cad,
            )
            output_data["object_pc"] = torch.tensor(on_surface_coords).float()

        if "normalized_mesh" in self.data_to_output:
            output_data["normalized_mesh"] = [
                torch.tensor(normalized_mesh.vertices).float(),
                torch.tensor(normalized_mesh.faces).float(),
            ]

        if "depth" in self.data_to_output:
            # output depth map
            output_data["depth"] = data["depth"]
            output_data["depth"][data["depth"] > 5000] = 0

        if "distance" in self.data_to_output:
            output_data["distance"] = data["distance"]

        if "instance_segmap" in self.data_to_output:
            output_data["instance_segmap"] = data["instance_segmap"]

        if "normals" in self.data_to_output:
            # off surface normals are all -1
            output_data["normals"] = data["obj_geom"]["normals"]

        if "sdf" in self.data_to_output:
            assert coords_numpy is not None, "Needs to output coords as well if you want to output sdf."
            sdf = torch.tensor(crisp.utils.geometry.query_sdf_from_mesh(coords_numpy, normalized_mesh)).float()
            output_data["sdf"] = sdf

        if "sdf_grid" in self.data_to_output:
            cube_scale = self.sample_bounds[1] - self.sample_bounds[0]
            assert cube_scale > 0

            # randomly sampled off surface points
            # 1. global samples (the points should be within [-1.0, 1.0] if self.normalized_recons is True)
            off_surface_coords_global, _, _ = crisp.utils.geometry.voxelize_cube(
                N=self.global_nonmnfld_points_voxel_res, cube_center=np.array([0, 0, 0]), cube_scale=cube_scale
            )
            global_nz_sdf_values = crisp.utils.geometry.query_sdf_from_mesh(off_surface_coords_global, normalized_mesh)
            output_data["sdf_grid"] = torch.tensor(global_nz_sdf_values).float()

        if "metadata" in self.data_to_output:
            output_data["metadata"] = data["metadata"]

        if "cam_intrinsics" in self.data_to_output:
            output_data["cam_intrinsics"] = torch.tensor(np.array(data["intrinsics"])).float()

        if "cam_pose" in self.data_to_output:
            processed_pose = self._pose_post_process[dataset](
                {
                    "pose": np.array(data["cam_pose"]),
                    "recons_t_blender": torch.tensor(self.objects_info[dataset][obj_name]["recons_t_blender"]),
                    "recons_s_blender": self.objects_info[dataset][obj_name]["recons_s_blender"],
                }
            )
            output_data["cam_pose"] = processed_pose.float()

        return output_data

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
        dataset, obj_name = (data["metadata"]["dataset_name"], data["metadata"]["obj_name"])
        data["obj_geom"] = self.distinct_objects[dataset][obj_name]

        if "surface_points" in data["obj_geom"].keys():
            return self._getitem_from_mem_objs(data)
        else:
            return self._getitem_online(data)

    def _get_obj_names_in_dataset(self):
        """Get all unique object names in the dataset"""
        for file in tqdm(self.hdf5files, desc="Loading unique object names"):
            with h5py.File(os.path.join(self.folder_path, file), "r") as f:
                metadata = json.loads(f[self.metadata_key][()])
                self._used_obj_names.add((metadata["dataset_name"], metadata["obj_name"]))
        print(f"Total unique object names: {len(self._used_obj_names)}")

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
                "metadata": json.loads(f[self.metadata_key][()]),
                # load cam states and pose
                # cam_states -> cam2world, cam_K
                "intrinsics": json.loads(f[self.cam_states_key][()])[self.cam_intrinsics_key],
                "cam_pose": json.loads(f[self.cam_states_key][()])[self.cam_pose_key],
            }

            if "nocs" in self.data_to_output:
                data_dict.update({"nocs": torch.tensor(np.transpose(np.array(f[self.nocs_key]), (2, 0, 1))).float()})

            return data_dict

    def _check_dataset_paths(self):
        if self.shapenet_dataset_path is not None:
            assert os.path.isdir(self.shapenet_dataset_path)
        if self.bop_dataset_path is not None:
            assert os.path.isdir(self.bop_dataset_path)
        if self.replicacad_dataset_path is not None:
            assert os.path.isdir(self.replicacad_dataset_path)
        if self.spe3r_dataset_path is not None:
            assert os.path.isdir(self.spe3r_dataset_path)


class DistributedObjClassBatchSampler(DistributedSampler):
    def __init__(
        self,
        per_obj_class_batch_size,
        num_classes_per_batch,
        unified_ds,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        self.per_obj_class_batch_size = per_obj_class_batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_size = self.num_classes_per_batch * self.per_obj_class_batch_size

        if isinstance(unified_ds, torch.utils.data.dataset.Subset):
            self.unified_ds = unified_ds.dataset
            self.indices = unified_ds.indices
        else:
            self.unified_ds = unified_ds
            self.indices = range(len(self.unified_ds.hdf5files))

        # note: we always drop last
        super().__init__(
            dataset=unified_ds, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed, drop_last=True
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
        for i in range(len(self.indices)):
            f_index = self.indices[i]
            hdf5file = self.unified_ds.hdf5files[f_index]
            with h5py.File(os.path.join(self.unified_ds.folder_path, hdf5file), "r") as f:
                metadata = json.loads(f[self.unified_ds.metadata_key][()])
            dataset, obj_name = (metadata["dataset_name"], metadata["obj_name"])
            self.obj_cls_to_index[(dataset, obj_name)].append(i)

        # sorted obj classes
        self.all_obj_classes = sorted(list(self.obj_cls_to_index.keys()))

        # calculate how many batches
        # ideally, we want to at least cover each frame 1 time
        # This can be formulated as a coupon collector's problem
        # https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
        # T is the number of samples.
        # E(T) = n * H(n) where n is the number of frame
        num_frames = len(self.unified_ds.hdf5files)
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


class ObjClassBatchSampler:
    def __init__(
        self,
        per_obj_class_batch_size,
        num_classes_per_batch,
        unified_ds,
        num_devices=1,
    ):
        """
        Sampling based batch sampler.
        For each batch, we output images belong to a fixed number of object classes.
        For each class, it has per_obj_class_batch_size images.
        Not guaranteed to be covering all images due to random sampling.
        Within the batch, images belong to the same class are sequential:
        [class1_img, class1_img, class2_img, class2_img, ...]
        """
        self.drop_last = True

        self.per_obj_class_batch_size = per_obj_class_batch_size
        self.num_classes_per_batch = num_classes_per_batch
        self.batch_size = self.num_classes_per_batch * self.per_obj_class_batch_size
        print(f"Resetting batch size to {self.batch_size}")

        if isinstance(unified_ds, torch.utils.data.dataset.Subset):
            self.unified_ds = unified_ds.dataset
            self.indices = unified_ds.indices
        else:
            self.unified_ds = unified_ds
            self.indices = range(len(self.unified_ds.hdf5files))
        self.num_devices = num_devices
        self._make_batches()

    def _make_batches(self):
        # read the hdf5 files
        # make obj label to files mapping
        self.obj_cls_to_index = defaultdict(list)
        # In case we are using a Subset wrapped dataset, we need to get the actual file index from the
        # subset indices.
        # If we are using the dataset directly, self.indices is just a consecutive sequence of integers
        for i in range(len(self.indices)):
            f_index = self.indices[i]
            hdf5file = self.unified_ds.hdf5files[f_index]
            with h5py.File(os.path.join(self.unified_ds.folder_path, hdf5file), "r") as f:
                metadata = json.loads(f[self.unified_ds.metadata_key][()])
            dataset, obj_name = (metadata["dataset_name"], metadata["obj_name"])
            self.obj_cls_to_index[(dataset, obj_name)].append(i)

        # sorted obj classes
        self.all_obj_classes = sorted(list(self.obj_cls_to_index.keys()))

        # calculate how many batches
        # ideally, we want to at least cover each frame 1 time
        # This can be formulated as a coupon collector's problem
        # https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
        # T is the number of samples.
        # E(T) = n * H(n) where n is the number of frame
        num_frames = len(self.unified_ds.hdf5files)
        T = num_frames * crisp.utils.math.H(num_frames)
        self.num_batches_per_epoch = int(np.floor(T / self.batch_size))
        # divide by the number of devices we are running training on
        self.num_batches_per_epoch = int(self.num_batches_per_epoch / self.num_devices)

    def __iter__(self):
        all_cls = self.all_obj_classes

        # sample images
        for i in range(self.num_batches_per_epoch):
            batch = []
            # sample classes
            shuffled_cls_idx = np.random.choice(len(all_cls), size=self.num_classes_per_batch, replace=False)
            for j in shuffled_cls_idx:
                cls = all_cls[j]
                # sample images
                cls_img_count = len(self.obj_cls_to_index[cls])
                sampled_image_idx = np.random.choice(cls_img_count, size=self.per_obj_class_batch_size, replace=True)
                batch.extend([self.obj_cls_to_index[cls][k] for k in list(sampled_image_idx)])

            yield batch

    def __len__(self):
        return self.num_batches_per_epoch
