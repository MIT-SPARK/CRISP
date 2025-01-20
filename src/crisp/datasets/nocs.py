import logging
import pickle
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import cv2
import glob
import torch
import yaml
from pathlib import Path
from torchvision.transforms import functional as tvf
from skimage.util import img_as_float
from joblib import Parallel, delayed

from crisp.datasets.nocs_config import Config
from crisp.datasets.nocs_utils import (
    obj_name_from_path,
    load_mesh,
    rotate_and_crop_images,
    resize_image,
    resize_mask,
    extract_bboxes,
    extract_inst_bboxes,
    minimize_mask,
)
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.math import depth_to_point_cloud_map_batched
from crisp.datasets.augmentations import invert_T, to_torch_uint8
from crisp.utils.visualization_utils import visualize_meshes_pyvista
from crisp.utils.parallel import tqdm_joblib


class Dataset:
    """Generates the NOCS dataset."""

    def __init__(self, synset_names=None, subset=None, make_yaml=False, admissible_objects=None, config=Config()):
        """

        Parameters
        ----------
        synset_names
        subset
        make_yaml
        admissible_objects: a function that takes subset and obj_label and returns True if the object is admissible
        config
        """
        if synset_names is None:
            synset_names = ["BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]
        self.synset_names = synset_names

        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        # which dataset: train/val/test
        self.subset = subset
        assert subset in ["train", "val", "test"]

        self.config = config

        self.source_image_ids = {}
        self.make_yaml = make_yaml
        if self.make_yaml:
            self.yaml_dump = {}
            self.yaml_dump["nocs"] = {}
        # Add classes
        for i, obj_name in enumerate(synset_names):
            if i == 0:  ## class 0 is bg class
                continue
            self.add_class("BG", i, obj_name)  ## class id starts with 1

        # object/instance index
        # a table containing instance IDs and their corresponding image IDs
        self.obj_image_index = []

        # admissible objects: objects that we can use; used for filtering out objects that have invalid model files
        self.admissible_objects = admissible_objects

        # gt transforms
        self.gt_transforms = {}

    @property
    def image_ids(self):
        return self._image_ids

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.d"""

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        # self.num_classes = len(self.class_info)
        self.num_classes = 0

        # self.class_ids = np.arange(self.num_classes)
        self.class_ids = []

        # self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.class_names = []

        # self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
        #                              for info, id in zip(self.class_info, self.class_ids)}
        self.class_from_source_map = {}

        for cls_info in self.class_info:
            source = cls_info["source"]
            if source == "coco":
                map_key = "{}.{}".format(cls_info["source"], cls_info["id"])
                self.class_from_source_map[map_key] = self.class_names.index(class_map[cls_info["name"]])
            else:
                self.class_ids.append(self.num_classes)
                self.num_classes += 1
                self.class_names.append(cls_info["name"])

                map_key = "{}.{}".format(cls_info["source"], cls_info["id"])
                self.class_from_source_map[map_key] = self.class_ids[-1]

        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))

        print(self.class_names)
        print(self.class_from_source_map)
        print(self.sources)

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_gt_transforms(self, gt_transforms_dir):
        """Load ground truth transformations from pickle files in specified directories."""
        folder_map = {"camera": ["train", "val"], "real": ["train", "test"]}
        self.gt_transforms = {}

        for main_folder, subfolders in folder_map.items():
            for subfolder in subfolders:
                folder_path = os.path.join(gt_transforms_dir, main_folder, subfolder)
                if os.path.exists(folder_path):
                    pkl_file = os.path.join(folder_path, "gt_results.pkl")
                    if os.path.isfile(pkl_file):
                        logging.info(f"Loading gt_results.pkl in {folder_path}.")
                        with open(pkl_file, "rb") as f:
                            gt_transforms = pickle.load(f)
                            if "gt_results_per_image_inst" in gt_transforms.keys():
                                dset_str = f"nocs_{main_folder}_{subfolder}"
                                self.gt_transforms[dset_str] = gt_transforms["gt_results_per_image_inst"]
                            else:
                                logging.info(f"gt_results_per_image_inst not found in {pkl_file}. Skipping.")
                    else:
                        logging.info(f"gt_results.pkl not found in {folder_path}. Skipping.")
                else:
                    logging.info(f"Folder {folder_path} does not exist. Skipping.")

    def load_camera_scenes(self, dataset_dir, obj_dir, if_calculate_mean=False):
        """Load a subset of the CAMERA dataset.
        dataset_dir: The root directory of the CAMERA dataset.
        subset: What to load (train, val)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """
        self.subset = "val" if self.subset == "test" else self.subset
        image_dir = os.path.join(dataset_dir, self.subset)
        source = "CAMERA"
        num_images_before_load = len(self.image_info)

        folder_list = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]

        num_total_folders = len(folder_list)

        image_ids = range(10 * num_total_folders)
        color_mean = np.zeros((0, 3), dtype=np.float32)
        # Add images
        for i in tqdm(image_ids, total=len(image_ids), desc="Loading CAMERA scenes"):
            image_id = int(i) % 10
            folder_id = int(i) // 10

            image_path = os.path.join(image_dir, "{:05d}".format(folder_id), "{:04d}".format(image_id))
            color_path = image_path + "_color.png"
            if not os.path.exists(color_path):
                continue

            meta_path = os.path.join(image_dir, "{:05d}".format(folder_id), "{:04d}_meta.txt".format(image_id))
            inst_dict = {}
            inst2mesh_dict = {}
            with open(meta_path, "r") as f:
                for line in f:
                    line_info = line.split(" ")
                    inst_id = int(line_info[0])  ##one-indexed
                    cls_id = int(line_info[1])  ##zero-indexed
                    # skip background objs
                    mesh_folder_id = line_info[2]
                    mesh_id = line_info[3][:-1]
                    if self.make_yaml:
                        yaml_key = f"{mesh_folder_id}_{mesh_id}"
                        curr_yamldict = {
                            "blender_R_cad": np.eye(3).tolist(),
                            "blender_s_cad": [1, 1, 1],
                            "recons_s_blender": 1.0,
                            "recons_t_blender": [1, 1, 1],
                        }
                        self.yaml_dump["nocs"][yaml_key] = curr_yamldict
                    mesh_path = f"{obj_dir}/camera_{self.subset}/{mesh_folder_id}/{mesh_id}/model.obj"
                    assert os.path.isfile(mesh_path)
                    inst2mesh_dict[inst_id] = mesh_path
                    inst_dict[inst_id] = cls_id

            width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
            height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

            self.add_image(
                source=source,
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                inst_dict=inst_dict,
                inst2mesh_dict=inst2mesh_dict,
            )

            if if_calculate_mean:
                image_file = image_path + "_color.png"
                image = cv2.imread(image_file).astype(np.float32)
                print(i)
                color_mean_image = np.mean(image, axis=(0, 1))[:3]
                color_mean_image = np.expand_dims(color_mean_image, axis=0)
                color_mean = np.append(color_mean, color_mean_image, axis=0)

        if if_calculate_mean:
            dataset_color_mean = np.mean(color_mean[::-1], axis=0)
            print("The mean color of this dataset is ", dataset_color_mean)

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print(
            "{} images are loaded into the dataset from {}.".format(
                num_images_after_load - num_images_before_load, source
            )
        )

    def load_real_scenes(self, dataset_dir, obj_dir):
        """Load a subset of the Real dataset.
        dataset_dir: The root directory of the Real dataset.
        subset: What to load (train, val, test)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """

        source = "Real"
        num_images_before_load = len(self.image_info)

        folder_name = "train" if self.subset == "train" else "test"
        image_dir = os.path.join(dataset_dir, folder_name)
        folder_list = [name for name in glob.glob(image_dir + "/*") if os.path.isdir(name)]
        folder_list = sorted(folder_list)
        image_id = 0
        for folder in tqdm(folder_list, total=len(folder_list), desc="Loading REAL scenes"):
            image_list = glob.glob(os.path.join(folder, "*_color.png"))
            image_list = sorted(image_list)
            for image_full_path in image_list:
                image_name = os.path.basename(image_full_path)
                image_ind = image_name.split("_")[0]
                image_path = os.path.join(folder, image_ind)

                meta_path = image_path + "_meta.txt"
                inst_dict = {}
                inst2mesh_dict = {}
                with open(meta_path, "r") as f:
                    for line in f:
                        line_info = line.split(" ")
                        inst_id = int(line_info[0])  ##one-indexed
                        cls_id = int(line_info[1])  ##zero-indexed
                        mesh_id = line_info[2][:-1]
                        if self.make_yaml:
                            yaml_key = f"real_{folder_name}_{image_ind}_{mesh_id}"
                            curr_yamldict = {
                                "blender_R_cad": np.eye(3).tolist(),
                                "blender_s_cad": [1, 1, 1],
                                "recons_s_blender": 1.0,
                                "recons_t_blender": [1, 1, 1],
                            }
                            self.yaml_dump["nocs"][yaml_key] = curr_yamldict
                        mesh_path = f"{obj_dir}/real_{folder_name}/{mesh_id}.obj"
                        assert os.path.isfile(mesh_path)
                        inst2mesh_dict[inst_id] = mesh_path
                        inst_dict[inst_id] = cls_id

                width = self.config.IMAGE_MAX_DIM  # meta_data['viewport_size_x'].flatten()[0]
                height = self.config.IMAGE_MIN_DIM  # meta_data['viewport_size_y'].flatten()[0]

                self.add_image(
                    source=source,
                    image_id=image_id,
                    path=image_path,
                    width=width,
                    height=height,
                    inst_dict=inst_dict,
                    inst2mesh_dict=inst2mesh_dict,
                )
                image_id += 1

        num_images_after_load = len(self.image_info)
        self.source_image_ids[source] = np.arange(num_images_before_load, num_images_after_load)
        print(
            "{} images are loaded into the dataset from {}.".format(
                num_images_after_load - num_images_before_load, source
            )
        )
        if self.make_yaml:
            print("Saving Yaml file...")
            yaml.dump(self.yaml_dump, open("/home/hengyu/harry/objects_info.yaml", "w"))

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            image_path = info["path"] + "_color.png"
            assert os.path.exists(image_path), "{} is missing".format(image_path)

            # depth_path = info["path"] + '_depth.png'
        else:
            assert False, "[ Error ]: Unknown image source: {}".format(info["source"])

        # print(image_path)
        image = cv2.imread(image_path)[:, :, :3]
        image = image[:, :, ::-1]

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    def load_depth(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file.
        """
        info = self.image_info[image_id]
        if info["source"] in ["CAMERA", "Real"]:
            depth_path = info["path"] + "_depth.png"
            depth = cv2.imread(depth_path, -1)

            if len(depth.shape) == 3:
                # This is encoded depth image, let's convert
                depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(
                    depth[:, :, 2]
                )  # NOTE: RGB is actually BGR in opencv
                depth16 = depth16.astype(np.uint16)
            elif len(depth.shape) == 2 and depth.dtype == "uint16":
                depth16 = depth
            else:
                assert False, "[ Error ]: Unsupported depth type."
        else:
            depth16 = None

        return depth16

    def image_reference(self, image_id):
        """Return the object data of the image."""
        info = self.image_info[image_id]
        if info["source"] in ["ShapeNetTOI", "Real"]:
            return info["inst_dict"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def process_data(self, mask_im, coord_map, inst_dict, inst2mesh_dict, meta_path, load_RT=False):
        # parsing mask
        cdata = mask_im
        cdata = np.array(cdata, dtype=np.int32)

        # instance ids
        instance_ids = list(np.unique(cdata))
        instance_ids = sorted(instance_ids)
        # remove background
        assert instance_ids[-1] == 255
        del instance_ids[-1]

        cdata[cdata == 255] = -1
        assert np.unique(cdata).shape[0] < 20

        num_instance = len(instance_ids)
        h, w = cdata.shape

        # flip z axis of coord map
        coord_map = np.array(coord_map, dtype=np.float32) / 255
        coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

        masks = np.zeros([h, w, num_instance], dtype=np.uint8)
        coords = np.zeros((h, w, num_instance, 3), dtype=np.float32)
        class_ids = np.zeros([num_instance], dtype=np.int_)
        scales = np.zeros([num_instance, 3], dtype=np.float32)

        with open(meta_path, "r") as f:
            lines = f.readlines()

        scale_factor = np.zeros((len(lines), 3), dtype=np.float32)
        for i, line in enumerate(lines):
            words = line[:-1].split(" ")

            if len(words) == 3:
                ## real scanned objs
                if words[2][-3:] == "npz":
                    npz_path = os.path.join(self.config.OBJ_MODEL_DIR, "real_val", words[2])
                    with np.load(npz_path) as npz_file:
                        scale_factor[i, :] = npz_file["scale"]
                else:
                    bbox_file = os.path.join(self.config.OBJ_MODEL_DIR, "real_" + self.subset, words[2] + ".txt")
                    scale_factor[i, :] = np.loadtxt(bbox_file)

                scale_factor[i, :] /= np.linalg.norm(scale_factor[i, :])

            else:
                bbox_file = os.path.join(
                    self.config.OBJ_MODEL_DIR, f"camera_{self.subset}", words[2], words[3], "bbox.txt"
                )
                bbox = np.loadtxt(bbox_file)
                scale_factor[i, :] = bbox[0, :] - bbox[1, :]

        i = 0

        # delete ids of background objects and non-existing objects
        inst_id_to_be_deleted = []
        for inst_id in inst_dict.keys():
            if inst_dict[inst_id] == 0 or (not inst_id in instance_ids):
                inst_id_to_be_deleted.append(inst_id)
        for delete_id in inst_id_to_be_deleted:
            del inst_dict[delete_id]
            del inst2mesh_dict[delete_id]

        inst_id_to_index_map = {}
        for inst_id in instance_ids:  # instance mask is one-indexed
            if not inst_id in inst_dict:
                continue
            inst_mask = np.equal(cdata, inst_id)
            assert np.sum(inst_mask) > 0
            assert inst_dict[inst_id]

            masks[:, :, i] = inst_mask
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))

            # class ids is also one-indexed
            class_ids[i] = inst_dict[inst_id]
            scales[i, :] = scale_factor[inst_id - 1, :]
            inst_id_to_index_map[inst_id] = i
            i += 1

        # print('before: ', inst_dict)

        masks = masks[:, :, :i]
        coords = coords[:, :, :i, :]
        coords = np.clip(coords, 0, 1)

        class_ids = class_ids[:i]
        scales = scales[:i]

        return cdata, masks, coords, class_ids, scales, inst_id_to_index_map

    def load_mask(self, image_id):
        """Generate instance masks for the objects in the image with the given ID.
        Note: this function will change the image_info field of the dataset.
        """
        info = self.image_info[image_id]
        # masks, coords, class_ids, scales, domain_label = None, None, None, None, None

        if info["source"] in ["CAMERA", "Real"]:
            domain_label = 0  ## has coordinate map loss

            mask_path = info["path"] + "_mask.png"
            coord_path = info["path"] + "_coord.png"

            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)

            inst_dict = info["inst_dict"]
            inst2mesh_dict = info["inst2mesh_dict"]
            meta_path = info["path"] + "_meta.txt"

            mask_im = cv2.imread(mask_path)[:, :, 2]
            coord_map = cv2.imread(coord_path)[:, :, :3]
            coord_map = coord_map[:, :, (2, 1, 0)]

            mask, masks, coords, class_ids, scales, inst_id_to_index_map = self.process_data(
                mask_im, coord_map, inst_dict, inst2mesh_dict, meta_path
            )

        else:
            assert False

        return mask, masks, coords, class_ids, scales, domain_label, inst_id_to_index_map

    def make_obj_image_index(self):
        """Helper function to generate the object/instance index"""

        def single_image_helper(image_id, info, admissible_objects):
            mask_path = info["path"] + "_mask.png"
            coord_path = info["path"] + "_coord.png"

            assert os.path.exists(mask_path), "{} is missing".format(mask_path)
            assert os.path.exists(coord_path), "{} is missing".format(coord_path)

            inst_dict = info["inst_dict"]
            inst2mesh_dict = info["inst2mesh_dict"]

            mask_im = cv2.imread(mask_path)[:, :, 2]
            cdata = mask_im
            cdata = np.array(cdata, dtype=np.int32)

            # instance ids
            instance_ids = list(np.unique(cdata))
            instance_ids = sorted(instance_ids)

            # remove background
            assert instance_ids[-1] == 255
            del instance_ids[-1]

            cdata[cdata == 255] = -1
            assert np.unique(cdata).shape[0] < 20

            # delete ids of background objects and non-existing objects
            inst_id_to_be_deleted = []
            for inst_id in inst_dict.keys():
                obj_label, subset = obj_name_from_path(Path(inst2mesh_dict[inst_id]))
                if (
                    inst_dict[inst_id] == 0
                    or (not inst_id in instance_ids)
                    or (not admissible_objects(subset, obj_label))
                ):
                    inst_id_to_be_deleted.append(inst_id)
            for delete_id in inst_id_to_be_deleted:
                del inst_dict[delete_id]
                del inst2mesh_dict[delete_id]

            partial_obj_image_index = []
            for key in inst_dict.keys():
                obj_label, subset = obj_name_from_path(Path(inst2mesh_dict[key]))
                partial_obj_image_index.append(
                    {
                        "image_id": image_id,
                        "inst_id": key,
                        "obj_label": obj_label,
                        "dataset_name": f"nocs_{subset}",
                    }
                )
            return partial_obj_image_index

        def accumulator(generator):
            result = []
            for value in tqdm(generator, total=len(self.image_info), desc="Making obj-image index."):
                result.extend(value)
            return result

        # use all but 1 CPU
        res = Parallel(n_jobs=-2, prefer="threads", return_as="generator")(
            delayed(single_image_helper)(image_id, info, self.admissible_objects)
            for image_id, info in enumerate(self.image_info)
        )

        self.obj_image_index = accumulator(res)


class NOCSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Dataset,
        config=Config(),
        augment=False,
        bop_ds_dir=None,
        folder_path=None,
        nocs_objects_dataset_path=None,
        preload_to_mem=False,
        split="train",
        sample_surface_points_count=1000,
        sample_local_nonmnfld_points_count=1000,
        sample_global_nonmnfld_points_count=5000,
        global_nonmnfld_points_voxel_res=128,
        sample_bounds=(-0.5, 0.5),
        pc_size=10000,
        model_img_H=420,
        model_img_W=420,
        normalized_recons=True,
        force_recompute_sdf=False,
        data_to_output=None,
        debug_vis=False,
    ):
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment
        unified_data_to_output = ["rgb", "nocs", "coords", "normals", "sdf", "cam_intrinsics", "cam_pose"]
        self.model_img_H, self.model_img_W = model_img_H, model_img_W

        self.unified_objects = UnifiedObjects(
            folder_path=folder_path,
            shapenet_dataset_path=None,
            bop_dataset_path=bop_ds_dir,
            replicacad_dataset_path=None,
            nocs_objs_dataset_path=nocs_objects_dataset_path,
            preload_to_mem=preload_to_mem,
            pc_size=pc_size,
            sample_surface_points_count=sample_surface_points_count,
            sample_local_nonmnfld_points_count=sample_local_nonmnfld_points_count,
            sample_global_nonmnfld_points_count=sample_global_nonmnfld_points_count,
            global_nonmnfld_points_voxel_res=global_nonmnfld_points_voxel_res,
            sample_bounds=sample_bounds,
            debug_vis=debug_vis,
            split=split,
            normalized_recons=normalized_recons,
            force_recompute_sdf=force_recompute_sdf,
            data_to_output=unified_data_to_output,
        )
        self.split = split
        self.data_to_output = data_to_output

        # camera intrinsics data
        self.camera_intrinsics = {
            "camera": np.array([[577.5, 0, 319.5], [0.0, 577.5, 239.5], [0.0, 0.0, 1.0]]),
            "real": np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]),
        }

        # for depth map computation
        self._depth_map_x_indices = torch.arange(640)
        self._depth_map_y_indices = torch.arange(480)
        self._depth_map_grid_x, self._depth_map_grid_y = torch.meshgrid(
            self._depth_map_x_indices, self._depth_map_y_indices, indexing="xy"
        )

        # classid-name mapping
        self.class_id_to_name = {}
        for entry in self.dataset.class_info:
            self.class_id_to_name[entry["id"]] = entry["name"]

    def compose_image_meta(self, image_id, image_shape, window, active_class_ids):
        meta = np.array(
            [image_id]
            + list(image_shape)  # size=1
            + list(window)  # size=3
            + list(active_class_ids)  # size=4 (y1, x1, y2, x2) in image cooredinates  # size=num_classes
        )
        return meta

    def load_image_gt(self, image_id, augment=False, use_mini_mask=False, load_scale=False):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        depth = self.dataset.load_depth(image_id)
        _, mask, coord, class_ids, scales, domain_label, _ = self.dataset.load_mask(image_id)

        shape = image.shape
        image, window, scale, padding = resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            padding=self.config.IMAGE_PADDING,
        )
        depth, _, _, _ = resize_image(
            depth,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            padding=self.config.IMAGE_PADDING,
        )
        mask = resize_mask(mask, scale, padding)
        coord = resize_mask(coord, scale, padding)
        bbox = extract_bboxes(mask)
        bbox = np.hstack((bbox, class_ids[:, np.newaxis]))

        active_class_ids = np.ones([self.dataset.num_classes], dtype=np.int32)

        # Resize masks to smaller size to reduce memory usage
        if use_mini_mask:
            mask = minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)
            coord = minimize_mask(bbox, coord, self.config.MINI_MASK_SHAPE)

        # Image meta data
        image_meta = self.compose_image_meta(image_id, shape, window, active_class_ids)

        return image, depth, image_meta, bbox, mask, coord, domain_label

    def load_image_gt_no_resize(self, image_id, augment=False, use_mini_mask=False, load_scale=False):
        # Load image and mask
        image = self.dataset.load_image(image_id)
        depth = self.dataset.load_depth(image_id)
        info = self.dataset.image_info[image_id]
        inst_dict = info["inst_dict"]

        # int to float
        # image, depth = img_as_float(image), img_as_float(depth)
        image, depth = img_as_float(image), depth.astype(np.float32)

        # gt_inst_mask preserves the instance ID as non-zero mask value
        gt_inst_mask, mask, coord, class_ids, scales, domain_label, inst_id_to_index_map = self.dataset.load_mask(
            image_id
        )

        shape = image.shape
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        bbox = extract_inst_bboxes(inst_dict, gt_inst_mask)

        active_class_ids = np.ones([self.dataset.num_classes], dtype=np.int32)

        # Image meta data
        image_meta = self.compose_image_meta(image_id, shape, window, active_class_ids)

        return image, depth, image_meta, bbox, mask, coord, domain_label, inst_id_to_index_map

    def mold_image(self, images: np.ndarray, config: Config):
        """Takes RGB images with 0-255 values and subtraces
        the mean pixel and converts it to float. Expects image
        colors in RGB order.
        """
        return images.astype(np.float32) - config.MEAN_PIXEL

    def __getitem__(self, inst_index):
        inst_obj_info = self.dataset.obj_image_index[inst_index]
        # inst_id: The ID of the instance in the image mask; different from the index of the particular object
        # in the list of labels/coords/masks
        image_id, inst_id, obj_label, subset_name = (
            inst_obj_info["image_id"],
            inst_obj_info["inst_id"],
            inst_obj_info["obj_label"],
            inst_obj_info["dataset_name"],
        )
        image_info = self.dataset.image_info[image_id]

        (
            image,
            depth,
            image_metas,
            gt_boxes,
            gt_masks,
            gt_coords,
            gt_domain_label,
            inst_id_to_index_map,
        ) = self.load_image_gt_no_resize(
            image_id,
            augment=self.augment,
            use_mini_mask=False,
        )

        # from mm to m for depth
        depth /= 1000

        # crop images
        inst_id_idx = inst_id_to_index_map[inst_id]
        gt_box = gt_boxes[inst_id]
        y1, x1, y2, x2 = gt_box
        w = x2 - x1
        h = y2 - y1
        cropped_image = tvf.crop(
            torch.tensor(image.transpose((2, 0, 1))).float(), top=y1, left=x1, width=w, height=h
        ).contiguous()
        cropped_depth = tvf.crop(torch.tensor(depth), top=y1, left=x1, width=w, height=h).contiguous()
        cropped_mask = tvf.crop(
            torch.tensor(gt_masks[:, :, inst_id_idx]), top=y1, left=x1, width=w, height=h
        ).contiguous()
        cropped_nocs_pcs = (
            tvf.crop(
                torch.tensor(gt_coords[:, :, inst_id_idx, :].transpose((2, 0, 1))), top=y1, left=x1, width=w, height=h
            )
            .float()
            .contiguous()
        )

        processed_img = nn.functional.interpolate(
            cropped_image.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="bilinear"
        )
        processed_depth = nn.functional.interpolate(
            cropped_depth.unsqueeze(0).unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        )
        processed_mask = nn.functional.interpolate(
            cropped_mask.unsqueeze(0).unsqueeze(0).float(), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        ).int()

        processed_nocs_pc = nn.functional.interpolate(
            cropped_nocs_pcs.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
        )
        nonzero_nocs_pc_mask = torch.ne(torch.sum(processed_nocs_pc[0, ...], dim=0, keepdim=True), 0)
        processed_mask = torch.logical_and(processed_mask, nonzero_nocs_pc_mask)

        obj_meta = self.unified_objects.objects_info[subset_name][obj_label]
        blender_s_cad, blender_R_cad, recons_t_blender, recons_s_blender = (
            np.array(obj_meta["blender_s_cad"]),
            np.array(obj_meta["blender_R_cad"]),
            np.array(obj_meta["recons_t_blender"]),
            obj_meta["recons_s_blender"],
        )

        if self.unified_objects.normalized_recons:
            nocs = processed_nocs_pc
        else:
            # the NOCS loaded is in [0, 1]
            # we first need to convert it to [-1, 1]
            nocs = 2 * (processed_nocs_pc - 0.5)
            # we then need to scale it to the original scale
            nocs /= recons_s_blender
        nocs *= processed_mask

        # SDF
        obj_geom = self.unified_objects.distinct_objects[subset_name][obj_label]

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

        # gt transforms
        image_inst_name = f"{image_id}_{inst_id}"
        gt_transform = None
        if (
            subset_name in self.dataset.gt_transforms.keys()
            and image_inst_name in self.dataset.gt_transforms[subset_name].keys()
        ):
            gt_transform = self.dataset.gt_transforms[subset_name][image_inst_name]

        data = {}
        if "rgb" in self.data_to_output:
            data["rgb"] = processed_img.squeeze()

        if "depth" in self.data_to_output:
            data["depth"] = processed_depth.squeeze()

        if "depth_pc" in self.data_to_output:
            K = torch.tensor(self.camera_intrinsics[image_info["source"].lower()]).reshape(1, 1, 3, 3)
            pcs = depth_to_point_cloud_map_batched(
                torch.tensor(depth).unsqueeze(0).unsqueeze(0),
                K,
                grid_x=self._depth_map_grid_x,
                grid_y=self._depth_map_grid_y,
            )
            cropped_depth_pcs = tvf.crop(pcs, top=y1, left=x1, width=w, height=h).float().contiguous()
            interp_depth_pc = nn.functional.interpolate(
                cropped_depth_pcs, (self.model_img_H, self.model_img_W), mode="nearest-exact"
            )
            data["depth_pc"] = interp_depth_pc.squeeze()

        if "depth_pc_mask" in self.data_to_output:
            nonzero_depth_mask = torch.ne(torch.sum(processed_depth[0, ...], dim=0, keepdim=True), 0)
            data["depth_pc_mask"] = nonzero_depth_mask

        if "normals" in self.data_to_output:
            data["normals"] = obj_geom["normals"]

        if "instance_segmap" in self.data_to_output:
            data["instance_segmap"] = processed_mask.squeeze()

        if "coords" in self.data_to_output:
            data["coords"] = coords.float()

        if "normalized_mesh" in self.data_to_output:
            data["normalized_mesh"] = normalized_mesh

        if "nocs" in self.data_to_output:
            data["nocs"] = nocs.squeeze(0)

        if "sdf" in self.data_to_output:
            data["sdf"] = sdf

        if "sdf_grid" in self.data_to_output:
            data["sdf_grid"] = obj_geom["nonmnfld_coords_global_sdf"]

        if "object_pc" in self.data_to_output:
            data["object_pc"] = obj_geom["surface_points"]

        if "obj_pose" in self.data_to_output:
            # camera_T_recons
            # Convert to the correct frame/transformation
            # gt_transform: transforms NOCS to camera frame (cam_T_nocs)
            # gt_transformed_pts = gt_s[jj] * gt_R[jj] @ (gt_NOCS_pts - 0.5) + gt_t[jj].reshape(3, 1)
            #
            # what we want: recons_T_camera
            # first, we can get cam_T_recons
            # cam_p = gt_s[jj] * gt_R[jj] @ (p^my_nocs * recons_s_blender/2) + gt_t[jj].reshape(3, 1)
            # so we have:
            # cam_s_recons = gt_s * recons_s_blender / 2
            # cam_R_recons = gt_R
            # cam_t_recons = gt_t
            if gt_transform is not None:
                cam_s_recons = gt_transform["gt_s"] * recons_s_blender / 2
                cam_R_recons, cam_t_recons = gt_transform["gt_R"], gt_transform["gt_t"]
                data["obj_pose"] = [cam_s_recons, cam_R_recons, cam_t_recons]
            else:
                data["obj_pose"] = [None, None, None]

        if "cam_pose" in self.data_to_output:
            # recons_T_camera
            data["cam_pose"] = []
            # raise NotImplementedError

        if "metadata" in self.data_to_output:
            data["metadata"] = {
                "class_id": image_info["inst_dict"][inst_id],
                "class_name": self.class_id_to_name[image_info["inst_dict"][inst_id]],
                **inst_obj_info,
            }

        return data

    def __len__(self):
        return len(self.dataset.obj_image_index)


if __name__ == "__main__":
    synset_names = ["BG", "bottle", "bowl", "camera", "can", "laptop", "mug"]  # 0  # 1  # 2  # 3  # 4  # 5  # 6

    class_map = {
        "bottle": "bottle",
        "bowl": "bowl",
        "cup": "mug",
        "laptop": "laptop",
    }
    camera_dir = "/home/hengyu/harry/camera_dataset"
    obj_dir = "/home/hengyu/harry/obj_models"
    real_dir = "/home/hengyu/harry/real_dataset"

    dataset_train = Dataset(synset_names, "test", False, Config())
    dataset_train.load_camera_scenes(camera_dir, obj_dir)
    dataset_train.load_real_scenes(real_dir, obj_dir)
    dataset_train.prepare(class_map)
    dataset_train.load_image(0)
    dataset_train.load_depth(0)
    dataset_train.load_mask(0)
    dataloader_train = NOCSDataset(
        dataset_train,
        nocs_objects_dataset_path="/home/hengyu/harry/obj_models",
        folder_path="/home/hengyu/harry",
        split="test",
    )
    dataloader_train[0]
