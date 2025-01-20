import os
import numpy as np
import torch.nn as nn
import torch
from functools import partial
from pytorch3d import ops

from crisp.utils.visualization_utils import visualize_pcs_pyvista
from crisp.utils.math import (
    instance_depth_to_point_cloud_torch,
    make_se3_batched,
    make_scaled_se3_inverse_batched,
    make_scaled_se3_batched,
    project_simplex,
)
from crisp.utils.geometry import voxelize_cube
from crisp.utils.file_utils import safely_make_folders, uniquify
from crisp.models.registration import (
    umeyama_ransac,
    umeyama_ransac_batched,
    umeyama_ransac_batched_inlier_thres_target,
    arun_ransac_batched,
)
from crisp.models.shape_optim import shape_recovery_from_pc_cvxpy
from crisp.models.loss_functions import (
    snc_robust_loss,
    nocs_depths_loss,
    sdf_nocs_loss,
    sdf_input_loss,
    sdf_input_trimmed_loss,
)


class JointCorrector:
    def __init__(
        self,
        model,
        solver_algo="torch-gd-accel",
        nocs_registration_algo="ransac",
        sdf_input_loss_weight=1.0,
        sdf_nocs_loss_weight=1.0,
        trim_quantile=0.9,
        corrector_lr=0.1,
        corrector_scale_lr=1e-3,
        nocs_correction_lr=None,
        shape_correction_lr=None,
        solver_nesterov=False,
        solver_momentum=0.9,
        registration_inlier_thres=0.2,
        nonmnfld_pts_count=500,
        max_iters=50,
        max_ransac_iters=100,
        log_loss_traj=False,
        shape_code_library=None,
        log_dump_dir="./corrector_logs",
        device="cuda",
    ):
        """Joint corrector that works for batched-inputs"""
        super().__init__()
        self.model = model
        self.log_loss_traj = log_loss_traj
        self.loss_traj = []
        self.device = device

        self.solver_algo = solver_algo

        # learning rates
        self.corrector_lr = corrector_lr
        self.corrector_scale_lr = corrector_scale_lr
        self.nocs_correction_lr = corrector_lr if nocs_correction_lr is None else nocs_correction_lr
        self.shape_correction_lr = corrector_lr if shape_correction_lr is None else shape_correction_lr

        self.sdf_input_loss_weight = sdf_input_loss_weight
        self.sdf_nocs_loss_weight = sdf_nocs_loss_weight
        self.trim_quantile = trim_quantile
        self.solver_nesterov = solver_nesterov
        self.solver_momentum = solver_momentum

        print(f"Using lr={corrector_lr}, nesterov={solver_nesterov}, momentum={solver_momentum} for corrector.")

        self.registration_inlier_thres = registration_inlier_thres
        self.max_iters = max_iters
        self.max_ransac_iters = max_ransac_iters
        self.optimizer = None
        if self.solver_algo == "torch-gd-accel":
            self.optimizer = partial(
                torch.optim.SGD,
                lr=self.corrector_lr,
                momentum=self.solver_momentum,
                nesterov=self.solver_nesterov,
            )
        elif self.solver_algo == "torch-adam":
            self.optimizer = partial(
                torch.optim.Adam,
                lr=self.corrector_lr,
            )
        else:
            raise NotImplementedError

        # for registration in the objective function
        self.nocs_registration_helper = umeyama_ransac_batched

        # nonmnfld points for regularization
        self.nonmnfld_coords = (
            (torch.rand((nonmnfld_pts_count, 3), device=device, requires_grad=True) - 0.5) * 2
        ).unsqueeze(0)

        # prepare and process shape code matrix
        self.shape_code_library = shape_code_library
        if self.shape_code_library is not None:
            self._ordered_shape_keys = sorted(list(self.shape_code_library.keys()))
            self._shape_code_mat = np.zeros(
                (self.shape_code_library[self._ordered_shape_keys[0]].shape[0], len(self._ordered_shape_keys))
            )
            for i, obj_name in enumerate(self._ordered_shape_keys):
                self._shape_code_mat[:, i] = self.shape_code_library[obj_name]
            self._shape_code_mat = torch.tensor(self._shape_code_mat).float().to(device)

        self.log_dump_dir = log_dump_dir
        safely_make_folders([log_dump_dir])
        return

    def objective_nocs_only_sdf_nocs(self, corrected_nocs, shape_code, masks):
        """Object for NOCS only corrector with the SDF-NOCS loss"""

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=shape_code, coords=x)

        l = sdf_nocs_loss(f_sdf_conditioned=f_sdf_conditioned, nocs=corrected_nocs, weights=masks)
        return l

    def objective_nocs_only_sdf_input(self, corrected_nocs, shape_code, pcs, masks, reg_inlier_thres):
        """Object for NOCS only corrector with the SDF-INPUT loss"""
        # cam_s_nocs, cam_R_nocs, cam_t_nocs, best_inliers, _ = self.nocs_registration_helper(
        #    corrected_nocs,
        #    pcs,
        #    masks=masks,
        #    inlier_thres=reg_inlier_thres,
        #    confidence=0.99,
        #    max_iters=self.max_ransac_iters,
        # )
        # nocs_T_cam = make_scaled_se3_inverse_batched(cam_s_nocs, cam_R_nocs, cam_t_nocs)

        nocs_s_cam, nocs_R_cam, nocs_t_cam, best_inliers, _ = umeyama_ransac_batched_inlier_thres_target(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_scaled_se3_batched(nocs_s_cam, nocs_R_cam, nocs_t_cam)

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=shape_code, coords=x)

        l = sdf_input_loss(
            f_sdf_conditioned=f_sdf_conditioned, depth_pc=pcs, nocs_T_depth=nocs_T_cam, weights=masks, threshold=50
        )

        # l2 = 0.1 * sdf_nocs_loss(f_sdf_conditioned=f_sdf_conditioned, nocs=corrected_nocs, weights=masks)
        # return l + l2
        # penalty = nocs_penalty_max_l2_loss(corrected_nocs, weights=masks)
        # return l + penalty
        return l

    def objective_nocs_only_sdf_input_nocs(
        self, corrected_nocs, shape_code, pcs, masks, reg_inlier_thres, sdf_input_weight, sdf_nocs_weight
    ):
        """Object for NOCS only corrector with the SDF-INPUT loss"""
        nocs_s_cam, nocs_R_cam, nocs_t_cam, best_inliers, _ = umeyama_ransac_batched_inlier_thres_target(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_scaled_se3_batched(nocs_s_cam, nocs_R_cam, nocs_t_cam)

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=shape_code, coords=x)

        l = sdf_input_weight * sdf_input_loss(
            f_sdf_conditioned=f_sdf_conditioned, depth_pc=pcs, nocs_T_depth=nocs_T_cam, weights=masks, threshold=50
        )

        l2 = sdf_nocs_weight * sdf_nocs_loss(f_sdf_conditioned=f_sdf_conditioned, nocs=corrected_nocs, weights=masks)
        return l + l2

    def objective_nocs_only_sdf_input_no_scale(self, corrected_nocs, shape_code, pcs, masks, reg_inlier_thres):
        """Object for NOCS only corrector with the SDF-INPUT loss"""
        nocs_R_cam, nocs_t_cam, best_inliers, _ = arun_ransac_batched(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_se3_batched(nocs_R_cam, nocs_t_cam)

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=shape_code, coords=x)

        l = sdf_input_loss(
            f_sdf_conditioned=f_sdf_conditioned, depth_pc=pcs, nocs_T_depth=nocs_T_cam, weights=masks, threshold=50
        )

        return l

    def objective_sdf_input_nocs_no_scale(
        self,
        corrected_nocs,
        corrected_shape,
        pcs,
        masks,
        reg_inlier_thres,
        sdf_input_weight,
        sdf_nocs_weight,
        trim_quantile,
    ):
        """Object for NOCS only corrector with the SDF-INPUT loss"""
        nocs_R_cam, nocs_t_cam, best_inliers, _ = arun_ransac_batched(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_se3_batched(nocs_R_cam, nocs_t_cam)

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=corrected_shape, coords=x)

        l = sdf_input_weight * sdf_input_trimmed_loss(
            f_sdf_conditioned=f_sdf_conditioned,
            depth_pc=pcs,
            nocs_T_depth=nocs_T_cam,
            weights=masks,
            trim_quantile=trim_quantile,
        )
        l2 = sdf_nocs_weight * sdf_nocs_loss(f_sdf_conditioned=f_sdf_conditioned, nocs=corrected_nocs, weights=masks)

        # regularization on nocs correction
        # l3 = torch.sum(nocs_correction**2, dim=1).mean()
        # l3 = 0
        if torch.isnan(l):
            print("NAN loss encountered (SDF-INPUT)!")
        if torch.isnan(l2):
            print("NAN loss encountered (SDF-NOCS)!")
            # HACK: reset sdf-nocs loss to zero
            l2 = 0
        return l + l2, l, l2

    def objective_nocs_only_sdf_input_nocs_no_scale(
        self,
        corrected_nocs,
        shape_code,
        pcs,
        masks,
        reg_inlier_thres,
        sdf_input_weight,
        sdf_nocs_weight,
        trim_quantile,
    ):
        """Object for NOCS only corrector with the SDF-INPUT loss"""

        nocs_R_cam, nocs_t_cam, best_inliers, _ = arun_ransac_batched(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_se3_batched(nocs_R_cam, nocs_t_cam)

        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=shape_code, coords=x)

        l = sdf_input_weight * sdf_input_trimmed_loss(
            f_sdf_conditioned=f_sdf_conditioned,
            depth_pc=pcs,
            nocs_T_depth=nocs_T_cam,
            weights=masks,
            trim_quantile=trim_quantile,
        )
        l2 = sdf_nocs_weight * sdf_nocs_loss(f_sdf_conditioned=f_sdf_conditioned, nocs=corrected_nocs, weights=masks)

        if torch.isnan(l):
            print("NAN loss encountered!")
        if torch.isnan(l2):
            print("NAN loss encountered!")
        return l + l2

    def objective_nocs_only_sdf_input_nocs_fixed_corr(
        self,
        corrected_nocs,
        sdf_pcs,
        pcs,
        masks,
        reg_inlier_thres,
    ):
        """Object for NOCS only corrector with the SDF-INPUT-NOCS loss with fixed correspondences"""
        B = corrected_nocs.shape[0]
        nocs_s_cam, nocs_R_cam, nocs_t_cam, best_inliers, _ = umeyama_ransac_batched_inlier_thres_target(
            source_points=pcs,
            target_points=corrected_nocs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )

        # L2 Loss transform the given corr on SDF surface (projected NOCS points)
        depth_pc_nocs = nocs_s_cam.reshape(B, 1, 1) * torch.bmm(nocs_R_cam, pcs) + nocs_t_cam.reshape(B, 3, 1)

        # for i in range(B):
        #    visualize_pcs_pyvista(
        #        [depth_pc_nocs[i, ...].detach(), corrected_nocs[i, ...].detach()],
        #        colors=["cyan", "red"],
        #        pt_sizes=[10.0, 10.0],
        #    )

        l = torch.clamp(
            torch.sum(torch.square(sdf_pcs - depth_pc_nocs), dim=1),
            max=reg_inlier_thres.unsqueeze(1).expand(B, depth_pc_nocs.shape[-1]),
        ).mean()

        if torch.isnan(l):
            print("NAN loss encountered!")
        return l

    def objective(self, nocs_correction, shape_correction, nocs, shape_code, pcs, masks, reg_inlier_thres):
        """Object function for the corrector. Assume non-batched input.

        Parameters
        ----------
        nocs_correction: (3, H, W)
        shape_correction
        nocs: (3, H, W)
        shape_code
        depths
        mask
        intrinsic
        """
        ## enforce fixed shape
        # shape_correction = torch.zeros_like(shape_correction)

        corrected_nocs, corrected_shape = nocs + nocs_correction, shape_code + shape_correction

        # solve for similarity transformation
        # depths = s * R * nocs + t
        cam_s_nocs, cam_R_nocs, cam_t_nocs, _, _ = self.nocs_registration_helper(
            corrected_nocs,
            pcs,
            masks=masks,
            inlier_thres=reg_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        nocs_T_cam = make_scaled_se3_inverse_batched(cam_s_nocs, cam_R_nocs, cam_t_nocs)

        l_nocs_depths = nocs_depths_loss(nocs=corrected_nocs, depth_pc=pcs, nocs_T_depth=nocs_T_cam, inlier_thres=0.1)

        # conditioned sdf function (handles batched input)
        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=corrected_shape, coords=x)

        l_snc = snc_robust_loss(
            f_sdf_conditioned=f_sdf_conditioned,
            nocs=corrected_nocs,
            depth_pc=pcs,
            nocs_T_depth=nocs_T_cam,
            weights=masks,
            lambda_nocs=10,
            lambda_depths=10,
            threshold=1,
        )

        # l_total = l_snc + nocs_correction.norm(dim=0).mean() + shape_correction.norm(dim=0).mean()
        # l_total = l_snc + l_nocs_depths
        l_total = l_snc

        # TODO: Visualize the NOCS transformed to depth after correction
        return l_total

    def solve_nocs_only_inv_depths(
        self,
        nocs,
        pcs,
        masks,
        registration_inlier_thres,
    ):
        """Use inverse transformed depths as the corrected NOCS"""
        # solve for registration
        B = nocs.shape[0]
        nocs_s_cam, nocs_R_cam, nocs_t_cam, best_inliers, _ = umeyama_ransac_batched_inlier_thres_target(
            source_points=pcs,
            target_points=nocs,
            masks=masks,
            inlier_thres=registration_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )
        # corrected_nocs = inverse transformed depths
        corrected_nocs = nocs_s_cam.reshape(B, 1, 1) * torch.bmm(nocs_R_cam, pcs) + nocs_t_cam.reshape(B, 3, 1)
        results = {
            "nocs_correction": corrected_nocs - nocs,
            "nocs_s_cam": nocs_s_cam,
            "nocs_R_cam": nocs_R_cam,
            "nocs_t_cam": nocs_t_cam,
        }
        return results

    def solve_nocs_only_inv_depths_scale_free(
        self,
        nocs,
        pcs,
        masks,
        registration_inlier_thres,
    ):
        """Use inverse transformed depths as the corrected NOCS"""
        # solve for registration
        B = nocs.shape[0]
        nocs_R_cam, nocs_t_cam, best_inliers, _ = arun_ransac_batched(
            source_points=pcs,
            target_points=nocs,
            masks=masks,
            inlier_thres=registration_inlier_thres,
            confidence=0.99,
            max_iters=self.max_ransac_iters,
        )

        # corrected_nocs = inverse transformed depths
        corrected_nocs = torch.bmm(nocs_R_cam, pcs) + nocs_t_cam.reshape(B, 3, 1)
        results = {
            "nocs_correction": corrected_nocs - nocs,
            "nocs_R_cam": nocs_R_cam,
            "nocs_t_cam": nocs_t_cam,
        }
        return results

    def solve_nocs_only_fix_sdf_nocs_corr(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        nocs_initialization="inv_depth",
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        # optimize nocs to be on the surface of SDF
        r1 = self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-nocs",
            sdf_input_weight=0,
            sdf_nocs_weight=sdf_nocs_weight,
        )
        sdf_surf_nocs = nocs + r1["nocs_correction"]

        init_nocs_correction = None
        if nocs_initialization == "inv_depth":
            inv_depth_r = self.solve_nocs_only_inv_depths(
                nocs,
                pcs,
                masks,
                registration_inlier_thres,
            )
            init_nocs_correction = inv_depth_r["nocs_correction"].detach()
        elif nocs_initialization == "sdf_surf":
            init_nocs_correction = r1["nocs_correction"].detach()

        # for i in range(nocs.shape[0]):
        #    visualize_pcs_pyvista([sdf_surf_nocs[i, ...].detach()], colors=["cyan"], pt_sizes=[10.0])

        result = self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-nocs-input-fix-corr",
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            sdf_surf_nocs=sdf_surf_nocs.detach(),
            init_nocs_correction=init_nocs_correction,
        )
        return result

    def solve_nocs_only_inv_depths_sdf_input_nocs_scale_free(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        loss_type="sdf-input",
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths_scale_free(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type=loss_type,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            trim_quantile=trim_quantile,
            nocs_only=True,
            project_shape=False,
        )

    def solve_inv_depths_sdf_input_nocs_scale_free_active_shape_projection(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths_scale_free(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        x = self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-input-nocs-no-scale",
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            trim_quantile=trim_quantile,
            nocs_only=False,
            project_shape=True,
        )

        # shape coeffs to shape code
        corrected_shape_code = torch.transpose(
            self._shape_code_mat @ torch.transpose(x["shape_correction"], 0, 1), 0, 1
        )

        return {
            "nocs_correction": x["nocs_correction"],
            "postcrt_shape_code": corrected_shape_code,
        }

    def solve_sdf_input_nocs_no_scale_lsq_onehot_shape(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths_scale_free(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        x = self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-input-nocs-no-scale-lsq-onehot-shape",
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            trim_quantile=trim_quantile,
            nocs_only=False,
            project_shape=False,
        )

        return {
            "nocs_correction": x["nocs_correction"],
            "postcrt_shape_code": x["postcrt_shape_code"],
        }

    def solve_sdf_input_nocs_no_scale_lsq_onehot_initial_code_basis_shape(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths_scale_free(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        x = self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-input-nocs-no-scale-lsq-onehot-initial-code-basis-shape",
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            trim_quantile=trim_quantile,
            nocs_only=False,
            project_shape=False,
        )

        return {
            "nocs_correction": x["nocs_correction"],
            "postcrt_shape_code": x["postcrt_shape_code"],
        }

    def solve_nocs_only_sdf_input_nocs_scale_free(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        loss_type="sdf-input",
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type=loss_type,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            trim_quantile=trim_quantile,
        )

    def solve_inv_depths_sdf_input_nocs(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_type="sdf-input-nocs",
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type=loss_type,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            nocs_only=False,
        )

    def solve_inv_depths_sdf_input_nocs_scale_free(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_type="sdf-input-nocs",
        loss_multiplier=5e2,
        visualize=False,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths_scale_free(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type=loss_type,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            nocs_only=False,
        )

    def solve_nocs_only_inv_depths_sdf_input_nocs(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        loss_type="sdf-input",
        sdf_input_weight=None,
        sdf_nocs_weight=None,
    ):
        """Inverse transformed depths + SDF-INPUT-NOCS refinement"""
        results = self.solve_nocs_only_inv_depths(
            nocs,
            pcs,
            masks,
            registration_inlier_thres,
        )

        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type=loss_type,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=results["nocs_correction"].detach(),
            init_nocs_s_cam=results["nocs_s_cam"],
        )

    def _build_optim_problem(
        self,
        loss_type,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        init_nocs_correction=None,
        init_shape_correction=None,
        init_scale=None,
        trim_quantile=None,
        nocs_only=True,
        project_shape=False,
        **kwargs,
    ):
        B = nocs.shape[0]
        # initialize NOCS correction var
        if init_nocs_correction is None:
            nocs_correction = torch.zeros_like(nocs, requires_grad=True)
        else:
            nocs_correction = init_nocs_correction
            nocs_correction.requires_grad_(True)

        # initialize shape correction var
        if project_shape:
            # in this case, shape_correction is actually shape coefficients
            temp_shp_code = shape_code if init_shape_correction is None else shape_code + init_shape_correction

            # use least squares to solve for shape coefficients
            temp_coeffs = np.linalg.lstsq(
                self._shape_code_mat.cpu().numpy(),
                torch.transpose(temp_shp_code, 0, 1).cpu().numpy(),
            )[0]
            shape_correction = (
                torch.transpose(torch.tensor(temp_coeffs), 0, 1).detach().clone().to(shape_code.device).float()
            )
            shape_correction.requires_grad_(True)
        else:
            if init_shape_correction is None:
                shape_correction = torch.zeros_like(shape_code, requires_grad=True)
            else:
                shape_correction = init_shape_correction
                shape_correction.requires_grad_(True)

        # initial scale
        if init_scale is None:
            scale = torch.ones((B, 1, 1), requires_grad=True)
        else:
            scale = init_scale.reshape((B, 1, 1))

        if nocs_only:
            params_list = [{"params": nocs_correction, "lr": self.nocs_correction_lr}]
            optim_vars = {
                "nocs_correction": nocs_correction,
                "shape_correction": torch.tensor(0, device=shape_code.device, requires_grad=False),
            }
        else:
            params_list = [
                {"params": nocs_correction, "lr": self.nocs_correction_lr},
                {"params": shape_correction, "lr": self.shape_correction_lr},
            ]
            optim_vars = {"nocs_correction": nocs_correction, "shape_correction": shape_correction}

        # loss function builder
        if loss_type == "sdf-input":
            l_fn = lambda x: self.objective_nocs_only_sdf_input(
                x["nocs_correction"] + nocs, shape_code, pcs, masks, registration_inlier_thres
            )
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-nocs":
            l_fn = lambda x: self.objective_nocs_only_sdf_nocs(x["nocs_correction"] + nocs, shape_code, masks)
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-no-scale":
            l_fn = lambda x: self.objective_nocs_only_sdf_input_no_scale(
                x["nocs_correction"] + nocs, shape_code, pcs, masks, registration_inlier_thres
            )
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-nocs":
            l_fn = lambda x: self.objective_nocs_only_sdf_input_nocs(
                x["nocs_correction"] + nocs,
                shape_code,
                pcs,
                masks,
                registration_inlier_thres,
                sdf_input_weight=sdf_input_weight,
                sdf_nocs_weight=sdf_nocs_weight,
            )
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-nocs-upper-scale":
            # Put scale optimization in the upper level
            optimizer = self.optimizer(
                params=params_list,
            )
            scale.requires_grad_(True)
            optimizer.add_param_group({"params": scale, "lr": self.corrector_scale_lr})
            l_fn = lambda x: (
                self.objective_nocs_only_sdf_input_nocs_no_scale(
                    x["nocs_correction"] + nocs,
                    shape_code,
                    pcs * x["scale"],
                    masks,
                    registration_inlier_thres,
                    sdf_input_weight=sdf_input_weight,
                    sdf_nocs_weight=sdf_nocs_weight,
                    trim_quantile=trim_quantile,
                )
                # regularization on nocs correction
                + torch.sum(nocs_correction**2, dim=1).mean()
            )
            optim_vars.update({"scale": scale})
        elif loss_type == "sdf-input-nocs-fixed-scale":
            # use the given scale initial values as the scale
            l_fn = (
                lambda x: self.objective_nocs_only_sdf_input_nocs_no_scale(
                    x["nocs_correction"] + nocs,
                    shape_code,
                    pcs * scale,
                    masks,
                    registration_inlier_thres,
                    sdf_input_weight=sdf_input_weight,
                    sdf_nocs_weight=sdf_nocs_weight,
                    trim_quantile=trim_quantile,
                )
                # regularization on nocs correction
                + torch.sum(nocs_correction**2, dim=1).mean()
            )
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-nocs-no-scale":
            # Does not estimate scale in the loss function
            if project_shape:
                # project coefficients to shape code
                def l_fn(x):
                    corrected_shape_code = torch.transpose(
                        self._shape_code_mat @ torch.transpose(x["shape_correction"], 0, 1), 0, 1
                    )

                    return self.objective_sdf_input_nocs_no_scale(
                        x["nocs_correction"] + nocs,
                        corrected_shape_code,
                        pcs,
                        masks,
                        registration_inlier_thres,
                        sdf_input_weight=sdf_input_weight,
                        sdf_nocs_weight=sdf_nocs_weight,
                        trim_quantile=trim_quantile,
                    )

            else:
                l_fn = lambda x: self.objective_sdf_input_nocs_no_scale(
                    x["nocs_correction"] + nocs,
                    x["shape_correction"] + shape_code,
                    pcs,
                    masks,
                    registration_inlier_thres,
                    sdf_input_weight=sdf_input_weight,
                    sdf_nocs_weight=sdf_nocs_weight,
                    trim_quantile=trim_quantile,
                )
            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-nocs-no-scale-lsq-onehot-shape":
            # NOCS-only optimization w/ CVXPY solving onehot shape
            def l_fn(x):
                shape_results = shape_recovery_from_pc_cvxpy(
                    sdf_model=self.model.recons_net,
                    initial_shape_code=shape_code,
                    nocs=torch.transpose(x["nocs_correction"] + nocs, 1, 2),
                    masks=masks,
                    shape_code_library=self._shape_code_mat.detach().cpu().numpy(),
                    use_L1_reg=True,
                    use_onehot=True,
                    L1_weight=5,
                )
                x["postcrt_shape_code"] = shape_results["postcrt_shape_code"]
                corrected_shape_code = shape_results["postcrt_shape_code"]
                return self.objective_sdf_input_nocs_no_scale(
                    x["nocs_correction"] + nocs,
                    corrected_shape_code,
                    pcs,
                    masks,
                    registration_inlier_thres,
                    sdf_input_weight=sdf_input_weight,
                    sdf_nocs_weight=sdf_nocs_weight,
                    trim_quantile=trim_quantile,
                )

            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-input-nocs-no-scale-lsq-onehot-initial-code-basis-shape":
            # NOCS-only optimization w/ CVXPY solving onehot shape
            def l_fn(x):
                shape_results = shape_recovery_from_pc_cvxpy(
                    sdf_model=self.model.recons_net,
                    initial_shape_code=shape_code,
                    nocs=torch.transpose(x["nocs_correction"] + nocs, 1, 2),
                    masks=masks,
                    shape_code_library=self._shape_code_mat.detach().cpu().numpy(),
                    use_L1_reg=True,
                    use_initial_shape_code_basis=True,
                    use_onehot=True,
                    L1_weight=5,
                )
                x["postcrt_shape_code"] = shape_results["postcrt_shape_code"]
                corrected_shape_code = shape_results["postcrt_shape_code"]
                return self.objective_sdf_input_nocs_no_scale(
                    x["nocs_correction"] + nocs,
                    corrected_shape_code,
                    pcs,
                    masks,
                    registration_inlier_thres,
                    sdf_input_weight=sdf_input_weight,
                    sdf_nocs_weight=sdf_nocs_weight,
                    trim_quantile=trim_quantile,
                )

            optimizer = self.optimizer(
                params=params_list,
            )
        elif loss_type == "sdf-nocs-input-fix-corr":
            # fixed correspondences
            l_fn = (
                lambda x: self.objective_nocs_only_sdf_input_nocs_fixed_corr(
                    corrected_nocs=x["nocs_correction"] + nocs,
                    sdf_pcs=kwargs["sdf_surf_nocs"],
                    pcs=pcs,
                    masks=masks,
                    reg_inlier_thres=registration_inlier_thres,
                )
                + torch.sum(nocs_correction**2, dim=1).mean()
            )
            optimizer = self.optimizer(
                params=params_list,
            )
        else:
            raise NotImplementedError

        return l_fn, optimizer, optim_vars

    def solve(
        self,
        nocs,
        shape_code,
        pcs,
        masks,
        registration_inlier_thres,
        loss_multiplier=5e2,
        visualize=False,
        loss_type="sdf-input",
        sdf_input_weight=None,
        sdf_nocs_weight=None,
        trim_quantile=None,
        init_nocs_correction=None,
        init_nocs_s_cam=None,
        max_iters=None,
        iters_per_log=100,
        nocs_only=True,
        log_trajectory=False,
        project_shape=False,
        **kwargs,
    ):
        """Correct for NOCS only with multiple loss options

        Parameters
        ----------
        nocs: (B, 3, N)
        shape_code: (B, K)
        pcs: (B, 3, N)
        masks: (B, N)
        registration_inlier_thres: (B)
        """
        if masks is not None:
            assert nocs.shape[0] == shape_code.shape[0] == masks.shape[0]
        else:
            assert nocs.shape[0] == shape_code.shape[0]

        if pcs is not None:
            assert nocs.shape[1] == pcs.shape[1] == 3
            assert nocs.shape[0] == pcs.shape[0]
        if registration_inlier_thres is not None:
            assert registration_inlier_thres.shape[0] == nocs.shape[0]

        if max_iters is None:
            max_iters = self.max_iters

        if sdf_nocs_weight is None:
            sdf_nocs_weight = self.sdf_nocs_loss_weight
        if sdf_input_weight is None:
            sdf_input_weight = self.sdf_input_loss_weight
        if trim_quantile is None:
            trim_quantile = self.trim_quantile

        l_fn, optimizer, optim_vars = self._build_optim_problem(
            loss_type=loss_type,
            nocs=nocs,
            shape_code=shape_code,
            pcs=pcs,
            masks=masks,
            registration_inlier_thres=registration_inlier_thres,
            sdf_input_weight=sdf_input_weight,
            sdf_nocs_weight=sdf_nocs_weight,
            init_nocs_correction=init_nocs_correction,
            init_scale=init_nocs_s_cam,
            trim_quantile=trim_quantile,
            nocs_only=nocs_only,
            project_shape=project_shape,
            **kwargs,
        )

        best_l = torch.tensor(float("inf"))
        best_vars = optim_vars
        self.loss_traj.clear()
        for iter in range(max_iters):
            optimizer.zero_grad()

            l_all = loss_multiplier * l_fn(optim_vars)

            if type(l_all) == tuple:
                l, components = l_all[0], l_all[1:]
            else:
                l, components = l_all, (l_all)

            if l.item() < best_l:
                best_l = l.detach()
                best_vars = {}
                for k, v in optim_vars.items():
                    best_vars[k] = v.detach()

            l.backward()
            optimizer.step()

            if project_shape:
                # project coefficients to shape code
                temp_coeffs = optim_vars["shape_correction"].detach().cpu().numpy()
                projected_coeffs = project_simplex(temp_coeffs)
                optim_vars["shape_correction"].data = (
                    torch.tensor(projected_coeffs).float().to(optim_vars["shape_correction"].device)
                )

            if iter % iters_per_log == 0:
                print(f"iter = {iter}, corrector l: {l}, best l: {best_l}")

            if visualize:
                print("visualize NOCS")
                bs = nocs.shape[0]
                for j in range(bs):
                    visualize_pcs_pyvista(
                        [(nocs + optim_vars["nocs_correction"])[j, ...].detach()], colors=["cyan"], pt_sizes=[5.0]
                    )

                print("visualize PC + NOCS")
                for j in range(bs):
                    visualize_pcs_pyvista(
                        [(nocs + optim_vars["nocs_correction"])[j, ...].detach()], colors=["cyan"], pt_sizes=[5.0]
                    )
            if log_trajectory:
                self.loss_traj.append([x.detach().cpu().item() for x in components])

        print(f"iter = {max_iters - 1}  best l: {best_l}")

        if torch.isnan(l) or torch.isinf(best_l) or torch.isnan(best_l):
            print("NAN loss encountered!")

        if log_trajectory:
            np.save("corrector_traj.npy", self.loss_traj)

        return best_vars

    def solve_nocs_only_sdf_input(
        self, nocs, shape_code, pcs, masks, registration_inlier_thres, loss_multiplier=5e4, visualize=False
    ):
        """Correct for only nocs with the SDF-INPUT loss. DEPRECATED.

        Parameters
        ----------
        nocs: (B, 3, N)
        shape_code: (B, K)
        pcs: (B, 3, N)
        masks: (B, N)
        registration_inlier_thres: (B)
        """
        return self.solve(
            nocs,
            shape_code,
            pcs,
            masks,
            registration_inlier_thres,
            loss_multiplier=loss_multiplier,
            visualize=visualize,
            loss_type="sdf-input",
        )

    def solve_nocs_only_sdf_nocs(self, nocs, shape_code, masks, loss_multiplier=5e2):
        """Correct for only nocs with the SDF-NOCS loss"""
        return self.solve(
            nocs,
            shape_code,
            None,
            masks,
            None,
            loss_multiplier=loss_multiplier,
            visualize=False,
            loss_type="sdf-nocs",
        )

    def solve_nocs_only_proj_sdf_cad(self, nocs, shape_code, masks, loss_multiplier=5e2):
        """Correct for only nocs
        1. sample in grid
        2. project to sdf surface
        3. use the projected surface points as CAD model
        4. find the closest points among projected surface points to NOCS
        """
        B = nocs.shape[0]

        # proj to surface
        with torch.no_grad():
            grid_samples, voxel_size, voxel_origin = voxelize_cube(
                N=16, cube_center=np.array([0, 0, 0]), cube_scale=2.0
            )
            grid_samples = (
                torch.transpose(torch.tensor(grid_samples).float(), 0, 1).unsqueeze(0).expand(B, -1, -1).to(nocs.device)
            )
            samples = 2.1 * (torch.rand(B, 3, 2000, device=nocs.device) - 0.5)
            samples = torch.cat([grid_samples, samples], dim=2)

        sdf_proj_results = self.solve(
            samples,
            shape_code,
            None,
            None,
            None,
            loss_multiplier=loss_multiplier,
            visualize=False,
            loss_type="sdf-nocs",
            max_iters=150,
            iters_per_log=50,
        )
        projected_cad = samples + sdf_proj_results["nocs_correction"]

        with torch.no_grad():
            # closest point to NOCS: for each NOCS point, get the closest point in projected_cad
            knn_res = ops.knn_points(
                torch.transpose(nocs, -1, -2), torch.transpose(projected_cad, -1, -2), K=1, return_nn=True
            )
            # nocs_projected = torch.flatten(knn_res.knn, start_dim=1, end_dim=2)
            nocs_projected = torch.transpose(knn_res.knn.squeeze(2), -1, -2)

        # nocs_project_sdf_results = self.solve_nocs_only(
        #    nocs,
        #    shape_code,
        #    None,
        #    masks,
        #    None,
        #    loss_multiplier=loss_multiplier,
        #    visualize=False,
        #    loss_type="sdf-nocs",
        #    max_iters=150,
        # )
        # nocs_projected_direct = nocs + nocs_project_sdf_results["nocs_correction"]

        # for i in range(B):
        #    temp_cad = projected_cad[i, ...].detach().cpu()
        #    temp_proj_nocs = nocs_projected[i, ...].detach().cpu()
        #    temp_direct_proj_nocs = nocs_projected_direct[i, ...].detach().cpu()
        #    visualize_pcs_pyvista(
        #        [temp_cad, temp_proj_nocs, temp_direct_proj_nocs],
        #        colors=["cyan", "red", "blue"],
        #        pt_sizes=[5.0, 8.0, 8.0],
        #    )

        return {"nocs_correction": nocs_projected - nocs}

    def solve_nocs_and_shape(self, nocs, shape_code, pcs, masks, registration_inlier_thres):
        nocs_correction = torch.zeros_like(nocs, requires_grad=True)
        shape_correction = torch.zeros_like(shape_code, requires_grad=True)

        optimizer = self.optimizer(
            params=[{"params": nocs_correction}, {"params": shape_correction}],
            lr=self.corrector_lr,
            momentum=self.solver_momentum,
            nesterov=self.solver_nesterov,
        )

        self.loss_traj.clear()
        best_l = torch.tensor(float("inf"))
        best_nocs_correction = nocs_correction
        best_shape_correction = shape_correction
        for iter in range(self.max_iters):
            optimizer.zero_grad()
            l = self.objective(
                nocs_correction, shape_correction, nocs, shape_code, pcs, masks, registration_inlier_thres
            )

            if l < best_l:
                best_l = l.detach()
                best_nocs_correction = nocs_correction.detach()
                best_shape_correction = shape_correction.detach()

            if iter % 10 == 0:
                print(f"iter = {iter}, corrector l: {best_l}")

            l.backward()
            optimizer.step()

            if self.log_loss_traj:
                self.loss_traj.append(
                    {
                        "iter": iter,
                        "loss": l.item(),
                        "shape_code": (shape_code + shape_correction).detach().cpu().numpy(),
                        "nocs_code": (nocs + nocs_correction).detach().cpu().numpy(),
                    }
                )

        if self.log_loss_traj:
            np.save(uniquify(os.path.join(self.log_dump_dir, "corrector_traj.npy")), self.loss_traj)

        return best_nocs_correction, best_shape_correction


class JointInstanceCorrector(nn.Module):
    def __init__(
        self,
        model,
        corrector_lr=0.1,
        corrector_stop_tol=1e-4,
        solver_algo="torch-builtin-gd",
        nocs_registration_algo="ransac",
        nonmnfld_pts_count=500,
        log_dump_dir=None,
        device="cuda",
    ):
        """

        Parameters
        ----------
        model: Joint model for shape and NOCS
        solver_algo
        nocs_registration_algo
        """
        super().__init__()
        # corrector design
        # 1. Solve for T
        # B = sRA + t
        # T = [sR, t;
        #      0,  1]
        # R, t = self.point_set_registration_fn.forward(torch.nan_to_num(detected_keypoints + correction))
        # 2. Solve for delta_h_p
        # delta_h_p = self.delta_hp_registration.forward()
        # 3. loss
        # SDF / depth / NOCS consistency loss and other terms
        self.model = model
        self.log_loss_traj = log_dump_dir is not None
        self.log_dump_dir = log_dump_dir
        self.corrector_lr = corrector_lr
        self.corrector_stop_tol = corrector_stop_tol
        safely_make_folders([log_dump_dir])
        self.loss_traj = []

        self.solver_algo = solver_algo
        # fmt: off
        if self.solver_algo == "torch-gd-accel":
            # fixed step deepest descent
            def solver_helper(nocs, shape_codes, depths, masks, intrinsics):
                return self.batch_accel_gd(nocs, shape_codes, depths, masks, intrinsics)
        elif self.solver_algo == "torch-builtin-gd":
            def solver_helper(nocs, shape_codes, depths, masks, intrinsics):
                return self.torch_builtin_gd(nocs, shape_codes, depths, masks, intrinsics)
        else:
            raise NotImplementedError
        # fmt: on
        self.solver_helper = solver_helper

        # for registration in the objective function
        self.registration_algo = nocs_registration_algo
        if self.registration_algo == "ransac":

            def registration_helper(source, target):
                device = source.device
                try:
                    # p^CAM = T^CAM_CAD * p^CAD
                    # this gives us T^CAM_CAD
                    # source: nocs, target: cam
                    s, R, t, T, _ = umeyama_ransac(source=source, target=target, verbose=False)
                except RuntimeError as e:
                    print(f"Error: {str(e)}")
                    s = 1.0
                    R = torch.eye(3).to(device)
                    t = torch.zeros(3).to(device)
                    T = torch.eye(4).to(device)
                return s, R, t, T

        else:
            raise NotImplementedError
        self.nocs_registration_helper = registration_helper

        # nonmnfld points for regularization
        self.nonmnfld_coords = (
            (torch.rand((nonmnfld_pts_count, 3), device=device, requires_grad=True) - 0.5) * 2
        ).unsqueeze(0)

        return

    def solve(self, nocs, shape_code, depths, masks, intrinsics):
        """Solve a single instance"""
        return self.solver_helper(nocs, shape_code, depths, masks, intrinsics)

    def objective(self, nocs_correction, shape_correction, nocs, shape_code, depths, mask, intrinsic):
        """Object function for the corrector. Assume non-batched input.

        Parameters
        ----------
        nocs_correction: (3, H, W)
        shape_correction
        nocs: (3, H, W)
        shape_code
        depths
        mask
        intrinsic
        """
        # solve for similarity transformation
        # depths = s * R * nocs + t
        s, cam_R_cad, cam_t_cad, cam_T_cad = self.nocs_registration_helper(source=nocs, target=depths)
        nocs_T_cam = make_scaled_se3_inverse_batched(
            s.unsqueeze(0).unsqueeze(0), cam_R_cad.unsqueeze(0), cam_t_cad.unsqueeze(0).unsqueeze(2)
        )

        # conditioned sdf function (handles batched input)
        def f_sdf_conditioned(x):
            return self.model.recons_net.forward(shape_code=(shape_code + shape_correction).unsqueeze(0), coords=x)

        l_snc = snc_robust_loss(
            f_sdf_conditioned=f_sdf_conditioned,
            nocs=(nocs + nocs_correction).unsqueeze(0),
            depth_pc=depths.unsqueeze(0),
            nocs_T_depth=nocs_T_cam,
            weights=torch.ones((1, nocs.shape[1], 1)).to(nocs.device),
            lambda_nocs=10,
            lambda_depths=10,
            threshold=10,
        )

        # sample free-space points in (-1, 1)
        # nonmnfld_sdf = f_sdf_conditioned(nonmnfld_coords)
        # nonmnfld_gradients = diff_operators.gradient(nonmnfld_sdf, nonmnfld_coords)
        # l_sgr = sgr_robust_loss(gradients=torch.cat((nocs_gradients, depth_gradients, nonmnfld_gradients), dim=1),
        #                        nonmnfld_pts=nonmnfld_sdf, grad_weight=1, inter_weight=1)

        l_total = l_snc + nocs_correction.norm(dim=0).mean() + shape_correction.norm(dim=0).mean()

        return l_total

    def torch_builtin_gd(self, nocs, shape_code, depths, mask, intrinsic, max_iterations=50):
        """Use the PyTorch builtin standard GD solver"""
        torch.autograd.set_detect_anomaly(True)
        depth_pts, idxs = instance_depth_to_point_cloud_torch(depths, intrinsic, mask)
        nocs_pts = nocs[:3, ...].clone().detach()
        nocs_pts = (nocs_pts[:, idxs[0], idxs[1]] - 0.5) * 2  # to [-1, 1]
        nocs_pts = nocs_pts.detach()
        depth_pts = depth_pts.detach()
        nocs_correction = torch.zeros_like(nocs_pts, requires_grad=True)
        shape_correction = torch.zeros_like(shape_code, requires_grad=True)

        print(f"corrector lr = {self.corrector_lr}")
        optimizer = torch.optim.SGD(
            params=[{"params": nocs_correction}, {"params": shape_correction}], lr=self.corrector_lr
        )

        best_l = torch.tensor(float("inf"))
        prev_l = torch.tensor(float("inf"))
        best_nocs_correction = nocs_correction
        best_shape_correction = shape_correction
        self.loss_traj.clear()
        for iter in range(max_iterations):
            optimizer.zero_grad()
            l = self.objective(nocs_correction, shape_correction, nocs_pts, shape_code, depth_pts, mask, intrinsic)

            if l < best_l:
                best_l = l.detach()
                best_nocs_correction = nocs_correction.detach()
                best_shape_correction = shape_correction.detach()

            if iter % 10 == 0:
                print(f"Current NAGD iter = {iter}, best obj = {best_l.detach().cpu().item()}")

            if (l - prev_l).abs().max() < self.corrector_stop_tol:
                break

            l.backward()
            optimizer.step()

            if self.log_loss_traj:
                self.loss_traj.append(
                    {
                        "iter": iter,
                        "loss": l.item(),
                        "shape_code": (shape_code + shape_correction).detach().cpu().numpy(),
                        "nocs_code": (nocs_pts + nocs_correction).detach().cpu().numpy(),
                    }
                )

        if self.log_loss_traj:
            np.save(uniquify(os.path.join(self.log_dump_dir, "instance_corrector_traj.npy")), self.loss_traj)

        return best_nocs_correction, best_shape_correction

    def batch_accel_gd(self, nocs, shape_code, depths, mask, intrinsic, max_iterations=50, tol=1e-12, gamma=0):
        """Steepest descent with Nesterov acceleration

        See Nocedal & Wright, eq. 3.6(a) & (b)

        inputs:

        outputs:
        correction          : torch.tensor of shape (B, 3, N)

        Parameters
        ----------
        nocs: (3, H, W) note that this is within the range of [0, 1], and the corrector transforms it to [-1, 1]
        shape_code
        depths
        mask
        intrinsic
        lr
        max_iterations
        tol
        gamma
        """
        lr = self.corrector_lr

        def _get_objective_jacobian(fun, nocs_correction, shape_correction):
            torch.set_grad_enabled(True)

            # TODO: Use torch.func
            dfdcorrX = torch.autograd.functional.jacobian(fun, (nocs_correction, shape_correction))

            return dfdcorrX[0], dfdcorrX[1]

        N = nocs.shape[-1]
        device = nocs.device

        depth_pts, idxs = instance_depth_to_point_cloud_torch(depths, intrinsic, mask)
        nocs_pts = nocs[:3, ...].clone().detach().requires_grad_(True)
        nocs_pts = (nocs_pts[:, idxs[0], idxs[1]] - 0.5) * 2  # to [-1, 1]
        nocs_correction = torch.zeros_like(nocs_pts)
        shape_correction = torch.zeros_like(shape_code)
        nocs_correction.requires_grad_(True)
        shape_correction.requires_grad_(True)

        # wrapper for function and gradient function
        if torch.all(torch.isnan(nocs)):
            print(f"Joint corrector has all NaN NOCS. Aborting.")
            return nocs_correction

        f = lambda nx, sx: self.objective(nx, sx, nocs_pts, shape_code, depth_pts, mask, intrinsic)

        # max_iterations = max_iterations
        # tol = tol
        # lr = lr
        # create a new trajectory
        if self.log_loss_traj:
            self.loss_traj.clear()

        # calculate initial obj value (this stores the current value)
        obj_ = f(nocs_correction, shape_correction)
        obj = torch.tensor(float("inf"), device=device)

        # prepare variables
        y_nocs, y_shape = nocs_correction.clone(), shape_correction.clone()
        y_nocs_prev, y_shape_prev = y_nocs.clone(), y_shape.clone()

        iter = 0
        while iter < max_iterations:
            iter += 1
            if iter % 10 == 0:
                print(f"Current NAGD iter = {iter}, obj = {obj_}")

            # using steepest descent, descent direction = -gradient
            # dfdcorrection size: (B, 3, num keypoints)
            dfdcorrection_nocs, dfdcorrection_shape = _get_objective_jacobian(f, nocs_correction, shape_correction)
            if torch.all(torch.isnan(dfdcorrection_nocs)) and torch.all(torch.isnan(dfdcorrection_shape)):
                print(f"Joint corrector Jacobians are all NaNs at iter={iter}.")
                break

            # gradient descent
            y_nocs = nocs_correction - lr * dfdcorrection_nocs
            y_shape = shape_correction - lr * dfdcorrection_shape

            # momentum
            nocs_correction = y_nocs + gamma * (y_nocs - y_nocs_prev)
            shape_correction = y_shape + gamma * (y_shape - y_shape_prev)

            # update y
            y_nocs_prev = y_nocs.clone()
            y_shape_prev = y_shape.clone()

            # update objective value
            obj_ = f(nocs_correction, shape_correction)

            if self.log_loss_traj:
                self.loss_traj.append(
                    {
                        "iter": iter,
                        "loss": obj_.item(),
                        "shape_code": (shape_code + shape_correction).detach().cpu().numpy(),
                        "nocs_code": (nocs_pts + nocs_correction).detach().cpu().numpy(),
                    }
                )

            if (obj - obj_).abs().max() < tol:
                break

            # save old obj value for convergence check
            obj = torch.clone(obj_)

        print(f"Solver (w/ NAGD) done. Final iter: {iter}")
        self.iters = iter

        if self.log_loss_traj:
            np.save(uniquify(os.path.join(self.log_dump_dir, "instance_corrector_traj.npy")), self.loss_traj)

        del nocs_pts, depth_pts, idxs, dfdcorrection_nocs, dfdcorrection_shape, obj, obj_
        torch.cuda.empty_cache()

        return nocs_correction, shape_correction
