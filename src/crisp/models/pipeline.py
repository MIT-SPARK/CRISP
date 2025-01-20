from collections import defaultdict
import contextlib
from collections import deque
from typing import Optional

from torch.nn.modules.module import T
from torchvision.transforms import functional as tvf
from torchvision.transforms import v2 as tvt2
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from crisp.backend.slam import ObjectPGOSolver
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import JointCorrector
from crisp.models.mf_corrector import MultiFrameShapeCorrector, MultiFrameGeometricShapeCorrector
from crisp.models.certifier import FrameCertifier
from crisp.models.shape_optim import create_F_matrix, compute_condition_number
from crisp.models.registration import (
    umeyama_ransac_batched,
    umeyama_ransac_batched_inlier_thres_target,
    arun_ransac_batched,
)
from crisp.utils.constants import *
from crisp.utils.math import (
    depth_to_point_cloud_map_batched,
    sample_within_nonzero_masks,
    make_scaled_se3_inverse_batched,
    make_scaled_se3_batched,
    make_se3_batched,
    instance_depth_to_point_cloud_torch,
)
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    gen_pyvista_voxel_slices,
    visualize_meshes_pyvista,
    imgs_show,
    visualize_sdf_slices_pyvista,
)


def get_reg_inlier_thres_from_nocs(nocs, masks, registration_inlier_thres):
    cnts = torch.sum(masks, dim=1)
    source_centroid = (torch.sum(nocs * masks.unsqueeze(1).int(), dim=2, keepdim=False) / cnts.unsqueeze(1)).unsqueeze(
        2
    )
    centered_source = nocs - source_centroid
    source_diameter = 2 * torch.amax(torch.linalg.norm(centered_source, dim=1), dim=1)
    reg_inlier_thres = registration_inlier_thres * source_diameter
    return reg_inlier_thres


class Pipeline(nn.Module):
    """Main pipeline for running experiments. Batched inputs, batched outputs."""

    def __init__(
        self,
        model: JointShapePoseNetwork,
        corrector: JointCorrector,
        frame_certifier: FrameCertifier,
        pgo_solver: Optional[ObjectPGOSolver],
        multi_frame_shape_code_corrector: Optional[MultiFrameShapeCorrector],
        multi_frame_geometric_shape_corrector: Optional[MultiFrameGeometricShapeCorrector],
        device,
        registration_inlier_thres=0.01,
        nr_downsample_before_corrector=5000,
        frame_corrector_mode="nocs-only",
        sdf_input_loss_multiplier=5e3,
        ssl_batch_size=5,
        ssl_nocs_clamp_quantile=0.9,
        input_H=480,
        input_W=640,
        normalized_recons=True,
        normalize_input_image=True,
        output_intermediate_vars=False,
        output_precrt_results=True,
        output_degen_condition_number=False,
        ssl_augmentation=False,
        ssl_augmentation_type="random-brightness-contrast",
        ssl_augmentation_gaussian_perturb_std=0.01,
        ssl_train_nocs=True,
        ssl_train_shape=True,
        no_grad_model_forward=False,
        profile_runtime=False,
        shape_code_library=None,
        debug_settings=None,
    ):
        """

        Parameters
        ----------
        model
        corrector
        frame_certifier
        device
        registration_inlier_thres
        nr_downsample_before_corrector
        frame_corrector_mode
        output_intermediate_vars: set to True and the output payload will include intermediate values
                                  (precorrector & post corrector NOCS, precorrector & post corrector shape)
        debug_settings
        """
        super().__init__()
        self.model = model
        self.corrector = corrector
        self.frame_certifier = frame_certifier
        self.pgo_solver = pgo_solver
        self.mf_shape_code_corrector = multi_frame_shape_code_corrector
        self.mf_geometric_shape_corrector = multi_frame_geometric_shape_corrector
        self.device = device
        self.ssl_train = True
        self.normalized_recons = normalized_recons
        self.no_grad_model_forward = no_grad_model_forward
        self.output_degen_condition_number = output_degen_condition_number
        self.degen_condition_tracker = defaultdict(list)
        self.nocs_acc_tracker = defaultdict(list)
        self.shape_code_library = shape_code_library
        self.profile_runtime = profile_runtime
        if self.output_degen_condition_number and self.shape_code_library is None:
            raise ValueError("Cannot output degenerate condition number without shape code library.")
        if self.shape_code_library is not None:
            ordered_shape_keys = sorted(list(self.shape_code_library.keys()))
            self._shape_code_mat = np.zeros(
                (self.shape_code_library[ordered_shape_keys[0]].shape[0], len(ordered_shape_keys))
            )
            for i, obj_name in enumerate(ordered_shape_keys):
                self._shape_code_mat[:, i] = self.shape_code_library[obj_name]

        self.normalize_input_image = normalize_input_image
        if normalize_input_image:
            self.img_transform = transforms.Compose([transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.img_transform = nn.Identity()
        self.input_img_H, self.input_img_W = input_H, input_W
        try:
            self.model_img_H, self.model_img_W = model.backbone_input_res
        except:
            self.model_img_H, self.model_img_W = 420, 420
        print(f"Pipeline backbone model resolution: {self.model_img_H}, {self.model_img_W}.")

        self._depth_map_x_indices = torch.arange(self.input_img_W).to(device)
        self._depth_map_y_indices = torch.arange(self.input_img_H).to(device)
        self._depth_map_grid_x, self._depth_map_grid_y = torch.meshgrid(
            self._depth_map_x_indices, self._depth_map_y_indices, indexing="xy"
        )

        self.frame_corrector_mode = frame_corrector_mode
        self.nr_downsample_before_corrector = nr_downsample_before_corrector
        self.sdf_input_loss_multiplier = sdf_input_loss_multiplier
        self.registration_inlier_thres = registration_inlier_thres
        if self.nr_downsample_before_corrector is not None and self.nr_downsample_before_corrector < 10:
            raise ValueError(f"Downsample settings using too few points: {self.nr_downsample_before_corrector}")
        if debug_settings is None:
            self.debug_settings = self._default_debug_settings()
        else:
            self.debug_settings = debug_settings
        self.output_intermediate_vars = output_intermediate_vars
        self.output_precrt_results = output_precrt_results

        # ssl-related
        self.ssl_batch_size = ssl_batch_size
        self.ssl_nocs_clamp_quantile = ssl_nocs_clamp_quantile

        # ssl augmentation settings
        # whether to use data augmentation during SSL for shape code
        self.ssl_augmentation = ssl_augmentation
        # augmentations to use
        self.ssl_aug_type = ssl_augmentation_type
        assert ssl_augmentation_type in ["random-brightness-contrast", "gaussian-perturb"]
        self.ssl_aug_gaussian_std = ssl_augmentation_gaussian_perturb_std
        self.ssl_aug_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
            ]
        )

        self.ssl_train_nocs = ssl_train_nocs
        self.ssl_train_shape = ssl_train_shape
        self.ssl_samples = {
            # inputs
            "input_imgs": deque(),
            "input_unnormalized_imgs": deque(),
            "input_masks": deque(),
            # outputs
            "cert_nocs": deque(),
            "cert_shape": deque(),
            # other
            "nocs_sample_indices": deque(),
            "metadata": deque(),
        }

    def _default_debug_settings(self):
        return {
            # visualization settings
            "vis_input_batched_pcs": False,
            "vis_input_cropped_pcs": False,
            "vis_input_sampled_pcs": False,
            "vis_nocs": False,
            "vis_precorrector_nocs_with_sdf": False,
            "vis_postcorrector_nocs_with_sdf": False,
            "vis_precorrector_transformed_nocs": False,
            "vis_postcorrector_transformed_nocs": False,
            "vis_sdf_slices": False,
            "vis_mf_geometric_corrector": False,
            # parameters
            "vis_sdf_grid_scale": 1,
        }

    def _get_batch_size(self, data):
        cnt = 0
        for frame_objs in data:
            cnt += len(frame_objs)
        return cnt

    def _corrector_helper(self, precrt_nocs, precrt_shape, pcs, masks, reg_inlier_thres):
        """Helper function to handle different corrector type to test

        Parameters
        ----------
        precrt_nocs: B, 3, N
        precrt_shape: B, k
        pcs: B, 3, N
        masks: B, N
        reg_inlier_thres: B
        """
        if self.corrector is not None:
            with torch.cuda.amp.autocast(enabled=False):
                if self.frame_corrector_mode == "inv-depths-sdf-nocs-input-scale-free":
                    # NOCS + Shape corrector with SDF-NOCS-INPUT loss
                    results = self.corrector.solve_inv_depths_sdf_input_nocs_scale_free(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs-no-scale",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape + results["shape_correction"]
                elif self.frame_corrector_mode == "nocs-only-inv-depths-sdf-input-nocs-scale-free":
                    results = self.corrector.solve_nocs_only_inv_depths_sdf_input_nocs_scale_free(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs-no-scale",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "sdf-input-nocs-no-scale-lsq-onehot-shape":
                    results = self.corrector.solve_sdf_input_nocs_no_scale_lsq_onehot_shape(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = results["postcrt_shape_code"]
                elif self.frame_corrector_mode == "sdf-input-nocs-lsq-onehot-initial-code-basis-shape":
                    results = self.corrector.solve_sdf_input_nocs_no_scale_lsq_onehot_initial_code_basis_shape(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = results["postcrt_shape_code"]
                elif self.frame_corrector_mode == "inv-depths-sdf-input-nocs-active-shape-projection":
                    results = self.corrector.solve_inv_depths_sdf_input_nocs_scale_free_active_shape_projection(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = results["postcrt_shape_code"]
                elif self.frame_corrector_mode == "nocs-only-sdf-nocs":
                    # NOCS-only corrector with SDF-NOCS loss
                    results = self.corrector.solve_nocs_only_sdf_nocs(
                        nocs=precrt_nocs.detach(), shape_code=precrt_shape.detach(), masks=masks
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-proj-sdf-cad":
                    results = self.corrector.solve_nocs_only_proj_sdf_cad(
                        nocs=precrt_nocs.detach(), shape_code=precrt_shape.detach(), masks=masks
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-sdf-input":
                    # NOCS-only corrector with SDF-INPUT loss
                    results = self.corrector.solve_nocs_only_sdf_input(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-fix-sdf-nocs-corr":
                    # Fixing the NOCS-SDF surface correspondence
                    results = self.corrector.solve_nocs_only_fix_sdf_nocs_corr(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-inv-depths-sdf-input-nocs":
                    results = self.corrector.solve_nocs_only_inv_depths_sdf_input_nocs(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-inv-depths-sdf-input-nocs-upper-scale":
                    results = self.corrector.solve_nocs_only_inv_depths_sdf_input_nocs(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs-upper-scale",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-inv-depths-sdf-input-nocs-fixed-scale":
                    results = self.corrector.solve_nocs_only_inv_depths_sdf_input_nocs(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs-fixed-scale",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-sdf-input-nocs-scale-free":
                    results = self.corrector.solve_nocs_only_sdf_input_nocs_scale_free(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                        loss_multiplier=self.sdf_input_loss_multiplier,
                        loss_type="sdf-input-nocs-no-scale",
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                elif self.frame_corrector_mode == "nocs-only-inv-depths":
                    # NOCS-only corrector that solves transformation with NOCS and use inverse transformed depths as
                    # corrected nocs
                    results = self.corrector.solve_nocs_only_inv_depths(
                        nocs=precrt_nocs.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                    )
                    postcrt_nocs = precrt_nocs + results["nocs_correction"]
                    postcrt_shape_code = precrt_shape
                else:
                    # NOCS-shape corrector
                    print("Warning: deprecated")
                    nocs_correction, shape_correction = self.corrector.solve_nocs_and_shape(
                        nocs=precrt_nocs.detach(),
                        shape_code=precrt_shape.detach(),
                        pcs=pcs.detach(),
                        masks=masks,
                        registration_inlier_thres=reg_inlier_thres.detach(),
                    )
                    postcrt_nocs = precrt_nocs + nocs_correction
                    postcrt_shape_code = precrt_shape + shape_correction
        else:
            postcrt_nocs = precrt_nocs
            postcrt_shape_code = precrt_shape

        return postcrt_nocs, postcrt_shape_code

    def _preprocess(self, rgbs, masks, depths, intrinsics, objs, frames_info, additional_data=None):
        """
        additional_data: an optional list of dictionary containing additional image-based data to be cropped and
        processed. Assume the sizes of the image data tensors are the same as the original images.
        """
        if additional_data is None:
            additional_data = [{} for _ in range(len(frames_info))]

        # batch depth map to point clouds
        pcs = depth_to_point_cloud_map_batched(
            depths, intrinsics, grid_x=self._depth_map_grid_x, grid_y=self._depth_map_grid_y
        )

        N_frames = len(objs)
        # preallocate memory for model forward pass
        # img dimension: B, 3, H, W
        # mask dimension: B, H, W
        # sampled_indices: (B, self.nr_downsample_before_corrector)
        # sampled_indices_masks: (B, self.nr_downsample_before_corrector)
        # processed_imgs: (B, 3, self.model_img_H, self.model_img_W)
        # processed_original_imgs: (B, 3, self.model_img_H, self.model_img_W)
        # processed_pcs: (B, 3, self.nr_downsample_before_corrector)
        # processed_masks: (B, 1, self.model_img_H, self.model_img_W)
        sampled_indices = []
        sampled_indices_masks = []
        processed_pcs = []
        processed_masks = []
        processed_imgs = []
        processed_unnormalized_imgs = []
        processed_obj_meta = []
        processed_additional_data = []

        # preprocess for entire batch
        bid = 0
        frame_index = []
        object_index = []
        for fid in range(N_frames):
            obj_data = objs[fid]
            f_mask = torch.logical_and(depths[fid, ...] < 500, depths[fid, ...] > 0) * masks[fid, ...]
            processed_frame_obj_meta = []
            normalized_image = self.img_transform(rgbs[fid, ...])

            for obj_idx, obj in enumerate(obj_data):
                bbox = obj["bbox"]
                c_obj_additional_data = {}
                processed_c_obj_additional_data = {}

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    h, w = y2 - y1, x2 - x1
                    cropped_image = tvf.crop(normalized_image, top=y1, left=x1, width=w, height=h)
                    cropped_unnormalized_image = tvf.crop(rgbs[fid, ...], top=y1, left=x1, width=w, height=h)
                    cropped_pcs = tvf.crop(pcs[fid, ...], top=y1, left=x1, width=w, height=h).float().contiguous()
                    cropped_mask = tvf.crop(f_mask, top=y1, left=x1, width=w, height=h) == obj["id_in_segm"]

                    for k, v in additional_data[fid].items():
                        c_obj_additional_data[k] = tvf.crop(v, top=y1, left=x1, width=w, height=h)

                else:
                    # Skip cropping if bbox not present
                    cropped_image, cropped_unnormalized_image = normalized_image, rgbs[fid, ...]
                    cropped_pcs = pcs[fid, ...].float().contiguous()
                    cropped_mask = f_mask == obj["id_in_segm"]

                    for k, v in additional_data[fid].items():
                        c_obj_additional_data[k] = v

                valid_mask_pixels = torch.count_nonzero(cropped_mask)
                if valid_mask_pixels < 5:
                    print(
                        f"Valid mask pixel = {valid_mask_pixels} < 5. "
                        f"Skipping this object detection for {obj['label']} at {frames_info[fid]}."
                    )
                    continue

                if self.debug_settings["vis_input_cropped_pcs"]:
                    print("Visualizing input cropped point clouds.")
                    c_depth_pts = torch.flatten(cropped_pcs, start_dim=-2, end_dim=-1)
                    c_mask = torch.flatten(cropped_mask, start_dim=-2, end_dim=-1)
                    c_depth_pts = c_depth_pts[:, torch.nonzero(c_mask.flatten()).squeeze()]
                    visualize_pcs_pyvista(
                        [c_depth_pts], colors=["crimson"], pt_sizes=[10.0], bg_color="white", show_axes=False
                    )

                # interpolate and save
                bbox_shape = list(cropped_image.shape[1:])
                processed_img = nn.functional.interpolate(
                    cropped_image.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="bilinear"
                )
                processed_unnormalized_img = nn.functional.interpolate(
                    cropped_unnormalized_image.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="bilinear"
                )
                processed_mask = nn.functional.interpolate(
                    cropped_mask.unsqueeze(0).float(), (self.model_img_H, self.model_img_W), mode="nearest-exact"
                ).int()

                # process addtional data
                for k, v in c_obj_additional_data.items():
                    processed_c_obj_additional_data[k] = nn.functional.interpolate(
                        v.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
                    )

                # now create sampled point clouds and save the sampled indices
                # note that the sampled indices are for the interpolated images
                # we also logical AND the interpolated masks with the
                interp_pc = nn.functional.interpolate(
                    cropped_pcs.unsqueeze(0), (self.model_img_H, self.model_img_W), mode="nearest-exact"
                )
                nonzero_pc_mask = torch.ne(torch.sum(interp_pc[0, ...], dim=0, keepdim=True), 0)
                processed_mask = torch.logical_and(processed_mask, nonzero_pc_mask)

                # sample within mask of the cropped pcs
                # in the case with point cloud having fewer number of points than the requested sample amount,
                # the pc_sample_valid_masks will indicate which indices are the valid points (0/1 boolean mask)
                pc_sampled_indices, pc_sample_valid_masks, pc_sample_valid_counts = sample_within_nonzero_masks(
                    self.nr_downsample_before_corrector, processed_mask.view(1, -1).int(), padding=True
                )
                assert pc_sample_valid_masks.shape[0] == 1

                if pc_sampled_indices is None or pc_sample_valid_counts[0] < 5:
                    # 5 is the sample size used in RANSAC
                    print(
                        f"Valid depth pixels cnt = {pc_sample_valid_counts[0]} < 5. "
                        f"Skipping this object detection for {obj['label']} at {frames_info[fid]}"
                    )
                    continue

                # gather only from the valid PC points; the invalid ones are left at zero
                processed_pc = torch.zeros(1, 3, self.nr_downsample_before_corrector).to(rgbs.device)
                sampled_processed_pc = torch.gather(
                    interp_pc.view(1, 3, -1),
                    2,
                    pc_sampled_indices[:, pc_sample_valid_masks[0, ...]].unsqueeze(1).expand((1, 3, -1)),
                )
                processed_pc[0, :, : pc_sample_valid_counts[0].cpu().int().item()] = sampled_processed_pc

                # add variables to list
                sampled_indices.append(pc_sampled_indices)
                sampled_indices_masks.append(pc_sample_valid_masks.flatten())
                processed_pcs.append(processed_pc)
                processed_masks.append(processed_mask)
                processed_imgs.append(processed_img)
                processed_unnormalized_imgs.append(processed_unnormalized_img.squeeze(0))
                processed_additional_data.append(processed_c_obj_additional_data)

                if self.debug_settings["vis_input_sampled_pcs"]:
                    print(f"Visualizing input sampled point clouds.")
                    visualize_pcs_pyvista([processed_pcs[bid].squeeze(0).cpu()], colors=["crimson"], pt_sizes=[2.0])

                bid += 1
                frame_index.append(fid)
                object_index.append(obj_idx)
                processed_frame_obj_meta.append(obj)

            processed_obj_meta.append(processed_frame_obj_meta)

        # create values
        assert len(processed_imgs) == len(processed_pcs) == len(sampled_indices) == len(processed_masks)
        processed_imgs = torch.stack(processed_imgs).squeeze(1).to(rgbs.device)
        processed_pcs = torch.stack(processed_pcs).squeeze(1).to(rgbs.device)
        sampled_indices = torch.stack(sampled_indices).squeeze(1).to(rgbs.device).to(torch.int64)
        sampled_indices_masks = torch.stack(sampled_indices_masks).to(rgbs.device)
        processed_masks = torch.stack(processed_masks).squeeze(1).to(rgbs.device)
        return (
            processed_imgs,
            processed_masks,
            processed_pcs,
            sampled_indices,
            sampled_indices_masks,
            frame_index,
            object_index,
            processed_obj_meta,
            processed_unnormalized_imgs,
            processed_additional_data,
        )

    def forward(self, rgbs: torch.Tensor, masks: torch.Tensor, depths: torch.Tensor, intrinsics, objs, frames_info):
        """Forward on batched images

        Parameters
        ----------
        rgbs: (B, 3, H, W)
        masks: (B, 1, H, W)
        depths: (B, 1, H, W)
        intrinsics: (B, 1, 3, 3)
        objs: list of length B containing objects info dictionaries for each frame. Required keys for each dictionary are:
         - "label" (unique IDs for the object after data association)
        frames_info: list of length B containing meta info of each frame
        """
        assert objs[0][0].get("label") is not None

        (
            processed_imgs,
            processed_masks,
            processed_pcs,
            sampled_indices,
            sampled_indices_masks,
            frame_index,
            object_index,
            processed_objs,
            processed_unnormalized_imgs,
            _,
        ) = self._preprocess(rgbs, masks, depths, intrinsics, objs, frames_info)
        unique_ids = [x["label"] for xx in processed_objs for x in xx]

        # B is the total number of objects in the frames provided
        B = len(processed_masks)

        # pass to model
        model_time = None
        with torch.no_grad() if self.no_grad_model_forward else contextlib.nullcontext():
            if self.profile_runtime:
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                model_start.record()
            nocs_map, shape_code = self.model.forward_nocs_and_shape_code(img=processed_imgs, mask=processed_masks)
            if self.profile_runtime:
                model_end.record()
                torch.cuda.synchronize()
                model_time = model_start.elapsed_time(model_end)
            precrt_shape_code = shape_code.detach().cpu()

        # transform from [0, 1] to [-1, 1]
        if self.normalized_recons:
            nocs_map = (nocs_map - 0.5) * 2
        else:
            nocs_map = nocs_map

        # add data and retrieve shape code from multi-frame shape corrector
        shape_indices_corrected = []
        postcrt_shape_code = shape_code
        if self.mf_shape_code_corrector is not None:
            # for the case where we are using a MultiFrameShapeCorrector
            self.mf_shape_code_corrector.add_shp_codes(unique_ids, shape_code)

            # run mf shape
            # check certification w/ corrected shape code
            postcrt_shape_code = self.mf_shape_code_corrector.forward(unique_ids)

        mf_corrector_time = None
        if self.mf_geometric_shape_corrector is not None:
            # for the case where we are using a MultiFrameGeometricShapeCorrector
            if self.profile_runtime:
                mf_corrector_start = torch.cuda.Event(enable_timing=True)
                mf_corrector_end = torch.cuda.Event(enable_timing=True)
                mf_corrector_start.record()
            postcrt_shape_code, shape_indices_corrected = self.mf_geometric_shape_corrector.forward(
                unique_ids, shape_code, sdf_model=self.model.recons_net
            )
            if self.profile_runtime:
                mf_corrector_end.record()
                torch.cuda.synchronize()
                mf_corrector_time = mf_corrector_start.elapsed_time(mf_corrector_end)

            if len(shape_indices_corrected) > 0:
                if self.debug_settings["vis_mf_geometric_corrector"]:
                    print("Visualizing geometric MF corrector corrected shape codes.")
                    for i in shape_indices_corrected:
                        print("Visualize uncorrected shape code SDF slices & mesh.")
                        visualize_sdf_slices_pyvista(
                            torch.tensor(shape_code, device="cuda")[i, ...].unsqueeze(0),
                            self.model.recons_net,
                            cube_scale=self.debug_settings["vis_sdf_grid_scale"],
                        )

                        print("Visualize corrected shape code SDF slices & mesh.")
                        visualize_sdf_slices_pyvista(
                            torch.tensor(postcrt_shape_code, device="cuda")[i, ...].unsqueeze(0),
                            self.model.recons_net,
                            cube_scale=self.debug_settings["vis_sdf_grid_scale"],
                        )

        if self.debug_settings["vis_input_batched_pcs"]:
            print("Visualizing batched point clouds.")
            for i in range(B):
                c_depth_pts = processed_pcs[i, ...]
                visualize_pcs_pyvista([c_depth_pts], colors=["crimson"], pt_sizes=[2.0])

        # sampled indices are zero for the invalid points; which means the 0th nocs will be repeated in the NOCS values
        precrt_nocs = torch.gather(nocs_map.view(B, 3, -1), 2, sampled_indices.unsqueeze(1).expand((B, 3, -1)))

        pcs = processed_pcs
        masks = sampled_indices_masks  # (B, N)

        # calculate inlier threshold
        reg_inlier_thres = get_reg_inlier_thres_from_nocs(
            precrt_nocs, sampled_indices_masks, registration_inlier_thres=self.registration_inlier_thres
        )

        if self.debug_settings["vis_nocs"]:
            print(f"Visualize NOCS (before corrector)")
            for i in range(B):
                visualize_pcs_pyvista(
                    [precrt_nocs[i, :, masks[i, ...]].detach().float()], colors=["crimson"], pt_sizes=[5.0]
                )

        if self.debug_settings["vis_precorrector_transformed_nocs"]:
            print("Visualizing NOCS (before corrector) transformed into depth frame.")
            if self.normalized_recons:
                cam_s_nocs, cam_R_nocs, cam_t_nocs, _, status = umeyama_ransac_batched(
                    precrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=100
                )
            else:
                cam_s_nocs = torch.ones((B, 1), device=precrt_nocs.device)
                cam_R_nocs, cam_t_nocs, _, status = arun_ransac_batched(
                    precrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=100
                )

            for i in range(B):
                # np.save(uniquify("./batch_corrector_logs/batch_depth_pts.npy"), pcs[i, ...].detach().cpu().numpy())
                # np.save(
                #    uniquify("./batch_corrector_logs/batch_precorr_nocs.npy"),
                #    precrt_nocs[i, ...].detach().cpu().numpy(),
                # )

                print(f"Obj index = {i}, depth: blue, transformed NOCS: red.")
                # transformed nocs
                cam_p = cam_s_nocs[i, ...] * cam_R_nocs[i, :3, :3] @ precrt_nocs[i, ...] + cam_t_nocs[i, ...].reshape(
                    3, 1
                )

                # depth points
                c_depth_pts = processed_pcs[i, ...]
                visualize_pcs_pyvista(
                    [cam_p.detach(), c_depth_pts.detach()], colors=["crimson", "blue"], pt_sizes=[5.0, 5.0]
                )

        if self.debug_settings["vis_precorrector_nocs_with_sdf"]:
            print("Visualize NOCS w/ SDF (before corrector")
            # TODO: Implement this

        # handle framewise corrector logic
        if self.profile_runtime:
            sf_corrector_start = torch.cuda.Event(enable_timing=True)
            sf_corrector_end = torch.cuda.Event(enable_timing=True)
            sf_corrector_start.record()

        postcrt_nocs, postcrt_shape_code = self._corrector_helper(
            precrt_nocs=precrt_nocs.float(),
            precrt_shape=postcrt_shape_code.float(),
            pcs=pcs.float(),
            masks=masks.float(),
            reg_inlier_thres=reg_inlier_thres.float(),
        )

        sf_corrector_time = None
        if self.profile_runtime:
            sf_corrector_end.record()
            torch.cuda.synchronize()
            sf_corrector_time = sf_corrector_start.elapsed_time(sf_corrector_end)

        if self.normalized_recons:
            cam_s_nocs, cam_R_nocs, cam_t_nocs, _, status = umeyama_ransac_batched(
                postcrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=50
            )
        else:
            cam_s_nocs = torch.ones((B, 1), device=precrt_nocs.device)
            cam_R_nocs, cam_t_nocs, _, status = arun_ransac_batched(
                postcrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=50
            )

        # framewise certification
        certi = None  # (B,)
        clamped_scores = None
        if self.frame_certifier is not None:
            postcrt_nocs_T_depth = make_scaled_se3_inverse_batched(cam_s_nocs, cam_R_nocs, cam_t_nocs)
            certi, clamped_scores = self.frame_certifier.certify(
                nocs_T_depth=postcrt_nocs_T_depth, nocs=postcrt_nocs, shape_code=postcrt_shape_code, pcs=pcs
            )

        if self.mf_geometric_shape_corrector is not None:
            # accumulate nocs for multi-frame shape corrector
            self.mf_geometric_shape_corrector.add_nocs(unique_ids=unique_ids, nocs=postcrt_nocs, cert_mask=certi)

        if self.debug_settings["vis_sdf_slices"]:
            print("Visualizing SDF slices")
            for i in range(B):
                print("Visualize uncorrected shape code SDF slices & mesh.")
                visualize_sdf_slices_pyvista(
                    torch.tensor(shape_code, device="cuda")[i, ...].unsqueeze(0),
                    self.model.recons_net,
                    cube_scale=self.debug_settings["vis_sdf_grid_scale"],
                )

                print("Visualize corrected shape code SDF slices & mesh.")
                visualize_sdf_slices_pyvista(
                    torch.tensor(postcrt_shape_code, device="cuda")[i, ...].unsqueeze(0),
                    self.model.recons_net,
                    cube_scale=self.debug_settings["vis_sdf_grid_scale"],
                )

        if self.debug_settings["vis_postcorrector_transformed_nocs"]:
            print("Visualizing NOCS (after corrector) transformed into depth frame.")
            for i in range(B):
                print(f"Cert: {certi[i]}")
                # transformed nocs
                cam_p = cam_s_nocs[i, ...] * cam_R_nocs[i, :3, :3] @ postcrt_nocs[i, ...] + cam_t_nocs[i, ...].reshape(
                    3, 1
                )

                print(f"Corrected (red) and precorrected (blue) NOCS")
                visualize_pcs_pyvista(
                    [postcrt_nocs[i, :, masks[i, ...]].detach(), precrt_nocs[i, :, masks[i, ...]].detach()],
                    colors=["crimson", "blue"],
                    pt_sizes=[10.0, 10.0],
                )

                # depth points
                print(f"Corrected NOCS (red) transformed into depths (blue)")
                c_depth_pts = processed_pcs[i, ...]
                visualize_pcs_pyvista(
                    [cam_p.detach(), c_depth_pts.detach()], colors=["crimson", "blue"], pt_sizes=[5.0, 5.0]
                )

        precrt_certi = precrt_scores = None
        if self.output_precrt_results:
            if self.normalized_recons:
                precrt_cam_s_nocs, precrt_cam_R_nocs, precrt_cam_t_nocs, _, status = umeyama_ransac_batched(
                    precrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=100
                )
            else:
                precrt_cam_s_nocs = torch.ones((B, 1), device=precrt_nocs.device)
                precrt_cam_R_nocs, precrt_cam_t_nocs, _, status = arun_ransac_batched(
                    precrt_nocs, pcs, masks=masks, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=100
                )

            if self.frame_certifier is not None:
                precrt_nocs_T_depth = make_scaled_se3_inverse_batched(
                    precrt_cam_s_nocs, precrt_cam_R_nocs, precrt_cam_t_nocs
                )
                precrt_certi, precrt_scores = self.frame_certifier.certify(
                    nocs_T_depth=precrt_nocs_T_depth, nocs=precrt_nocs, shape_code=shape_code, pcs=pcs
                )

        mf_degen_conds, sf_degen_conds = [], []
        if self.output_degen_condition_number:
            # single-frame condition number
            shape_code_mat = (
                torch.tensor(self._shape_code_mat, device=self.device).float().unsqueeze(0).expand(B, -1, -1)
            )
            sf_F = create_F_matrix(
                self.model.recons_net,
                shape_code_mat=shape_code_mat,
                query_points=torch.transpose(postcrt_nocs, 1, 2),
                normalize_by_extent=True,
            )
            sf_F = np.nan_to_num(sf_F)
            sf_degen_conds = compute_condition_number(F_all=sf_F)

            # multi-frame condition number
            for ii in range(B):
                self.degen_condition_tracker[unique_ids[ii]].append(sf_F[ii, ...].T @ sf_F[ii, ...])
                self.nocs_acc_tracker[unique_ids[ii]].append(postcrt_nocs[ii, ...].cpu().detach().numpy())

        # add to ssl samples
        if self.ssl_train:
            self.update_ssl_samples(
                certi,
                input_imgs=processed_imgs,
                input_unnormalized_imgs=processed_unnormalized_imgs,
                input_masks=processed_masks,
                pred_nocs=postcrt_nocs,
                pred_shape=postcrt_shape_code,
                nocs_sample_indices=sampled_indices,
                shape_indices_corrected=shape_indices_corrected,
                metadata=[{**objs[i][j], **frames_info[i]} for i, j in zip(frame_index, object_index)],
            )

        # output:
        # for each frame:
        # objects info:
        # s, R, t, shape_code
        result_payload = {
            "cam_s_nocs": cam_s_nocs,
            "cam_R_nocs": cam_R_nocs,
            "cam_t_nocs": cam_t_nocs,
            "shape_code": postcrt_shape_code,
            "cert_mask": certi,
            "frame_index": frame_index,
            "obj_index": object_index,
            "clamped_scores": clamped_scores,
            "precrt_clamped_scores": precrt_scores,
            "sf_corrector_time": sf_corrector_time,
            "mf_corrector_time": mf_corrector_time,
            "model_inference_time": model_time,
        }

        if self.output_degen_condition_number:
            result_payload.update({"sf_degen_conds": sf_degen_conds})
            # oc quantile
            oc_score = torch.quantile(clamped_scores, self.frame_certifier.depths_quantile, dim=1, keepdim=False)
            result_payload.update({"oc_score": list(oc_score.cpu().detach().numpy())})

        if self.output_precrt_results:
            result_payload.update({"precrt_cert_mask": precrt_certi})
        if self.output_intermediate_vars:
            result_payload.update({"precrt_shp_code": precrt_shape_code})
            result_payload.update({"precrt_nocs": precrt_nocs, "postcrt_nocs": postcrt_nocs})
            result_payload.update({"pcs": pcs, "reg_inlier_thres": reg_inlier_thres, "masks": masks, "B": B})
            result_payload.update({"sampled_indices": sampled_indices})
            result_payload.update(
                {"nocs_map": nocs_map, "processed_imgs": processed_imgs, "processed_mask": processed_masks}
            )
        return result_payload

    def get_mf_acc_nocs(self):
        return self.nocs_acc_tracker

    def get_mf_degen_condition_number(self):
        """Return the degenerate condition number of the shape code for tracks"""
        mf_degen_conds = {}
        for k, v in self.degen_condition_tracker.items():
            FTF = np.cumsum([F.T @ F for F in v], axis=0)
            FTF_min_eig = [np.linalg.eigvalsh(ftf)[0] for ftf in FTF]
            FTF_cond = [np.linalg.cond(ftf) for ftf in FTF]
            FTF_rank = [np.linalg.matrix_rank(ftf) for ftf in FTF]
            mf_degen_conds[k] = [
                {"FTF_min_eig": x, "FTF_cond": y, "FTF_rank": z} for x, y, z in zip(FTF_min_eig, FTF_cond, FTF_rank)
            ]
        return mf_degen_conds

    def update_ssl_samples(
        self,
        certi,
        input_imgs,
        input_unnormalized_imgs,
        input_masks,
        pred_nocs,
        pred_shape,
        nocs_sample_indices,
        shape_indices_corrected,
        metadata,
    ):
        """Updated buffered SSL samples"""
        # save to a list with frame info / metadata about the object
        # and gt nocs
        B = certi.shape[0]
        print(f"Certified: {torch.sum(certi)}/{B}")
        for i in range(B):
            if certi[i]:
                # certified outputs
                self.ssl_samples["cert_nocs"].append(pred_nocs[i, ...].detach().cpu().numpy())
                self.ssl_samples["cert_shape"].append(pred_shape[i, ...].detach().cpu().numpy())

                # save the inputs
                self.ssl_samples["input_imgs"].append(input_imgs[i, ...].detach().cpu().numpy())
                self.ssl_samples["input_masks"].append(input_masks[i, ...].detach().cpu().numpy())
                # unnormalized images are converted to uint8 for data augmentation
                self.ssl_samples["input_unnormalized_imgs"].append(
                    tvt2.functional.convert_image_dtype(
                        input_unnormalized_imgs[i].detach().cpu(), dtype=torch.uint8
                    ).numpy()
                )

                self.ssl_samples["nocs_sample_indices"].append(nocs_sample_indices[i, ...].detach().cpu().numpy())

                # metadata
                # indicate whether the shape code was corrected for this instance
                mtd = metadata[i]
                mtd["nocs_corrected"] = True
                if "nocs-only" in self.frame_corrector_mode:
                    mtd["shape_code_corrected"] = i in shape_indices_corrected
                else:
                    # if frame-wise corrector also corrects shape,
                    # then we say all shape codes are corrected
                    mtd["shape_code_corrected"] = True
                self.ssl_samples["metadata"].append(mtd)

    def ssl_step(
        self, ssl_nocs_optimizer: torch.optim.Optimizer, ssl_recons_optimizer: torch.optim.Optimizer, fabric=None
    ):
        """Run a single SSL step, given an SSL optimizer, if we have enough certified samples"""
        B = self.ssl_batch_size
        if len(self.ssl_samples["cert_nocs"]) < B:
            return None
        assert len(self.ssl_samples["cert_nocs"]) == len(self.ssl_samples["cert_nocs"])
        print("We have enough certified samples. Performing SSL step.")

        # batch samples
        nocs_ssl_losses, shape_ssl_losses = [], []
        nocs_ssl_steps, shape_ssl_steps = 0, 0
        while len(self.ssl_samples["cert_nocs"]) >= B:
            ssl_batch = self._make_ssl_batch()
            cert_nocs, cert_shape, imgs, masks, sampled_indices, nocs_correction_mask, shape_correction_mask = (
                ssl_batch["cert_nocs"],
                ssl_batch["cert_shape"],
                ssl_batch["input_imgs"],
                ssl_batch["input_masks"],
                ssl_batch["nocs_sample_indices"],
                ssl_batch["nocs_corrected"],
                ssl_batch["shape_code_corrected"],
            )

            ssl_nocs_optimizer.zero_grad()
            ssl_recons_optimizer.zero_grad()

            # forward pass
            with fabric.autocast():
                nocs_map, shape_code = self.model.forward_nocs_and_shape_code(img=imgs, mask=masks)
                if self.normalized_recons:
                    nocs_map = (nocs_map - 0.5) * 2  # (B, 3, H, W)
                else:
                    nocs_map = nocs_map
                sampled_nocs_output = torch.gather(
                    nocs_map.view(B, 3, -1), 2, sampled_indices.unsqueeze(1).expand((B, 3, -1))
                )

                # loss: L2 on NOCS, L2 on shape
                if self.ssl_train_nocs and torch.sum(nocs_correction_mask) > 0:
                    nocs_diff_sq = torch.sum((sampled_nocs_output - cert_nocs) ** 2, dim=1)
                    nocs_loss_th = torch.quantile(nocs_diff_sq, self.ssl_nocs_clamp_quantile, dim=1, keepdim=True)
                    # 1e2 is due to the use of weight previously; don't want to tune LR again
                    nocs_loss = 1e2 * (torch.clamp(nocs_diff_sq, max=nocs_loss_th) * nocs_correction_mask).mean()

                    if fabric is not None:
                        fabric.backward(nocs_loss)
                    else:
                        nocs_loss.backward()
                    ssl_nocs_optimizer.step()
                    nocs_ssl_losses.append(nocs_loss.item())
                    nocs_ssl_steps += 1

                if self.ssl_train_shape and torch.sum(shape_correction_mask) > 0:
                    if self.ssl_augmentation:
                        aug_shape_code = self.model.forward_shape_code(
                            img=ssl_batch["input_augmented_imgs"],
                        )
                        shape_loss = (
                            torch.sum((cert_shape - aug_shape_code) ** 2, dim=1, keepdim=True) * shape_correction_mask
                        ).mean()
                    else:
                        shape_loss = (
                            torch.sum((cert_shape - shape_code) ** 2, dim=1, keepdim=True) * shape_correction_mask
                        ).mean()

                    if fabric is not None:
                        fabric.backward(shape_loss)
                    else:
                        shape_loss.backward()
                    ssl_recons_optimizer.step()
                    shape_ssl_losses.append(shape_loss.item())
                    shape_ssl_steps += 1

        print(f"SSL steps performed: nocs - {nocs_ssl_steps} shape - {shape_ssl_steps}.")
        all_ssl_losses = {"nocs_losses": nocs_ssl_losses, "shape_losses": shape_ssl_losses}
        return all_ssl_losses

    def _make_ssl_batch(self):
        """Prepare a batch of data for SSL training"""
        B = self.ssl_batch_size
        popped_samples = {}
        for k, v in self.ssl_samples.items():
            popped_samples[k] = [v.popleft() for _ in range(B)]

        if self.ssl_augmentation:
            output_keys = [
                "cert_nocs",
                "cert_shape",
                "input_imgs",
                "input_unnormalized_imgs",
                "input_masks",
                "nocs_sample_indices",
            ]
            output = {}
            for k in output_keys:
                if k == "input_unnormalized_imgs":
                    output[k] = popped_samples[k]
                else:
                    output[k] = torch.tensor(np.array(popped_samples[k])).to(self.device)

            # create aug images and masks
            output["input_augmented_imgs"] = []
            for bid in range(B):
                if self.ssl_aug_type == "random-brightness-contrast":
                    transformed = self.ssl_aug_transform(
                        image=np.transpose(output["input_unnormalized_imgs"][bid], axes=(1, 2, 0)),
                    )

                    output["input_augmented_imgs"].append(
                        self.img_transform(
                            tvf.convert_image_dtype(
                                torch.tensor(np.transpose(transformed["image"], axes=(2, 0, 1))),
                                dtype=torch.float16,
                            )
                        )
                    )
                elif self.ssl_aug_type == "gaussian-perturb":
                    noise = torch.randn_like(output["input_imgs"][bid], device=self.device) * self.ssl_aug_gaussian_std
                    new_img = output["input_imgs"][bid] + noise
                    output["input_augmented_imgs"].append(new_img)

            # stack them
            output["input_augmented_imgs"] = (
                torch.stack(output["input_augmented_imgs"], dim=0).float().to(self.device).contiguous()
            )
        else:
            output_keys = ["cert_nocs", "cert_shape", "input_imgs", "input_masks", "nocs_sample_indices"]
            output = {k: torch.tensor(0.0) for k in output_keys}
            for k in output_keys:
                output[k] = torch.tensor(np.array(popped_samples[k])).to(self.device)
        # corrector masks
        output["shape_code_corrected"] = (
            torch.tensor([popped_samples["metadata"][i]["shape_code_corrected"] for i in range(B)])
            .to(self.device)
            .unsqueeze(1)
        )
        output["nocs_corrected"] = (
            torch.tensor([popped_samples["metadata"][i]["nocs_corrected"] for i in range(B)])
            .to(self.device)
            .unsqueeze(1)
        )

        return output

    def update_pgo_odometry(self, odometry):
        """Add odometry and object detections to the PGO.

        Parameters
        ----------
        odometry: a list of tuples (frame_i, frame_j, T_ij)
        """
        for frame_i, frame_j, T_ij in odometry:
            self.pgo_solver.add_odom(frame_i, frame_j, T_ij)

    def update_pgo_objs(self, object_labels, cam_R_obj, cam_t_obj, index2frame_fn=lambda x: x):
        """Add object detections to the PGO.

        Parameters
        ----------
        object_labels: Globally unique object labels (after data associations), serving as landmark identifier
        cam_R_obj
        cam_t_obj
        """
        assert cam_R_obj.shape[0] == cam_t_obj.shape[0]
        T = make_se3_batched(cam_R_obj, cam_t_obj)
        for i in range(len(object_labels)):
            fid = index2frame_fn(i)
            self.pgo_solver.add_obj_pose(fid, object_labels[i], T[i, ...].numpy(force=True))

    def optimize_pgo(self):
        """Solve PGO problem"""
        result = self.pgo_solver.solve()
        return result

    def correct_multiframe_shape(self):
        """Correct for shape across frames

        Returns
        -------
        A dictionary with keys = object labels (unique IDs after data association) and values = shape code
        """
        return self.mf_shape_code_corrector.solve_all()

    def freeze_sdf_decoder_weights(self):
        """Freeze the weights for the trained SDF decoder (conditioned network)"""
        for param in self.model.recons_net.parameters():
            param.requires_grad = False
        self.model.recons_net.eval()

    def freeze_backbone_weights(self):
        """Freeze the backbeone weights"""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.model.backbone.eval()

    def eval(self: T) -> T:
        self.ssl_train = False
        return self.train(False)

    def make_AoS_output(self, payload):
        """Make Array-of-structure output, given Structure-of-Array"""
        N_frames = torch.tensor(payload["frame_index"]).max().item()
        result_payload = [[] for _ in range(N_frames)]
        for bid in range(len(payload["frame_index"])):
            fid = payload["frame_index"][bid]
            entry = {
                "cam_s_nocs": payload["cam_s_nocs"][bid, ...],
                "cam_R_nocs": payload["cam_R_nocs"][bid, ...],
                "cam_t_nocs": payload["cam_t_nocs"][bid, ...],
                "shape_code": payload["shape_code"][bid, ...],
                "cert_mask": None,
                "frame_index": fid,
                "obj_index": payload["obj_index"][bid],
            }
            if payload["cert_mask"] is not None:
                entry["cert_mask"] = payload["cert_mask"][bid, ...]
            result_payload[fid].append(entry)
        return result_payload
