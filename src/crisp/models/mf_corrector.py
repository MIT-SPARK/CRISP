from collections import defaultdict, deque
from functools import partial

import logging
import numpy as np
import torch
from torch import nn as nn

from crisp.models.robust_centroid import robust_centroid_gnc
from crisp.utils.geometry import geometric_median, voxelize_cube
from crisp.models.shape_optim import (
    create_F_matrix,
    compute_condition_number,
    shape_recovery_from_pc_gd,
    shape_recovery_from_pc_proj_gd,
    shape_recovery_from_pc_cvxpy,
)


class MultiFrameShapeCorrector(nn.Module):
    def __init__(self, mode="gnc", clamping_thres=0.1, rolling_window_size=None, min_buffer_size=5):
        """Multi-frame shape code corrector.
        The corrector takes input as unique IDs coming from data association.
        The corrector then solves a robust averaging optimization problem to obtain
        the corrected shape code.
        """
        super().__init__()

        # data structure for shape code storage: dictionary
        self.rolling_window_size = rolling_window_size
        if rolling_window_size is not None:
            assert rolling_window_size > 0
        # keys of shp_index would be track IDs
        # values will be shape codes
        self.shp_index = defaultdict(partial(deque, maxlen=rolling_window_size))
        self.mode = mode
        self.min_buffer_size = min_buffer_size
        self.corrector_fn = None
        if mode == "gnc":
            self.corrector_fn = self._robust_gnc_shim
        elif mode == "geometric-median":
            self.corrector_fn = self._geometric_median_shim
        elif mode == "mean":
            self.corrector_fn = self._mean_shim
        else:
            raise NotImplementedError(f"Unknown mode for multi-frame shape corrector: {mode}")

        self.clamping_thres = clamping_thres
        return

    def clear_buffer(self):
        """Clear the buffer of shape codes"""
        self.shp_index.clear()
        return

    def add_shp_codes(self, unique_ids, shp_codes: torch.Tensor):
        """Add shape code for one unique DA ID

        Parameters
        ----------
        unique_ids: lists of unique DA IDs with length = B
        shp_codes: (B, K) where K is the shape code dimension
        """
        assert len(unique_ids) == shp_codes.shape[0]
        for i, uid in enumerate(unique_ids):
            self.shp_index[uid].append(shp_codes[i, ...].float().numpy(force=True))

    def reached_rolling_window_size(self, id):
        """Return True if the rolling window is full for the provided id"""
        if self.rolling_window_size is None:
            print("Rolling window size is not set. Return True by default.")
            return True
        else:
            return len(self.shp_index[id]) == self.rolling_window_size

    def forward(self, unique_ids, device="cuda"):
        """Given unique IDs, return the corrected shape code batched for each unique ID"""
        # check for existing shape codes
        shp_code_cts = [len(self.shp_index[k]) for k in unique_ids]
        postcrt_shp_index = self.corrector_fn(unique_ids=unique_ids)
        # use the most recent shape code if the buffer is not past the min_buffer_size
        shp_code = torch.tensor(
            [
                postcrt_shp_index[k] if shp_code_cts[ii] >= self.min_buffer_size else self.shp_index[k][-1]
                for ii, k in enumerate(unique_ids)
            ],
            device=device,
        )
        return shp_code

    def solve_all(self):
        """Run robust averaging on the shape code for all unique DA IDs stored"""
        unique_ids = list(self.shp_index.keys())
        return self.corrector_fn(unique_ids=unique_ids)

    def _robust_gnc_shim(self, unique_ids):
        postcrt_shp_index = {}
        for k in unique_ids:
            v = self.shp_index[k]
            # (1, K, N)
            X = torch.tensor(np.array(v)).transpose(0, 1).unsqueeze(0).requires_grad_(False)
            payload = robust_centroid_gnc(X=X, cost_type="gnc-tls", clamp_thres=self.clamping_thres, max_iterations=50)
            x = payload["robust_centroid"]
            postcrt_shp_index[k] = x.numpy(force=True).squeeze(0).squeeze(1)
            assert postcrt_shp_index[k].ndim == 1
        return postcrt_shp_index

    def _geometric_median_shim(self, unique_ids):
        postcrt_shp_index = {}
        for k in unique_ids:
            v = self.shp_index[k]
            x = geometric_median(X=np.array(v))
            postcrt_shp_index[k] = x
        return postcrt_shp_index

    def _mean_shim(self, unique_ids):
        postcrt_shp_index = {}
        for k in unique_ids:
            v = self.shp_index[k]
            x = np.mean(np.array(v), axis=0)
            postcrt_shp_index[k] = x
        return postcrt_shp_index


class MultiFrameGeometricShapeCorrector(nn.Module):
    # Available solvers for shape recovery from partial point clouds
    available_solvers = [
        "GD",
        "PROJ_GD",
        "LSQ_CVXPY",
        "LSQ_CVXPY_L1",
        "LSQ_CVXPY_L1_ONEHOT",
        "LSQ_CVXPY_L1_INITIAL_CODE_BASIS",
        "LSQ_CVXPY_L1_INITIAL_CODE_BASIS_ONEHOT",
    ]

    def __init__(
        self,
        lr,
        rolling_window_size=None,
        shape_code_library=None,
        min_buffer_size=5,
        cube_scale=0.5,
        global_nonmnfld_points_voxel_res=64,
        cube_center=np.array([0, 0, 0]),
        mnfld_weight=3e3,
        inter_weight=1e2,
        nocs_sample_size=2000,
        max_iters=25,
        ignore_cert_mask=False,
        lsq_normalize_F_matrix=False,
        solver="GD",
        device="cpu",
    ):
        """Multi-frame shape code corrector.
        The corrector takes input as unique IDs coming from data association.
        The corrector then solves a robust averaging optimization problem to obtain
        the corrected shape code.
        """
        super().__init__()

        # data structure for shape code storage: dictionary
        self.rolling_window_size = rolling_window_size
        if rolling_window_size is not None:
            assert rolling_window_size > 0
        # keys of shp_index would be track IDs
        # values will be shape codes
        self.nocs_index = defaultdict(partial(deque, maxlen=rolling_window_size))
        self.min_buffer_size = min_buffer_size
        self.lr = lr
        self.inter_weight = inter_weight
        self.mnfld_weight = mnfld_weight
        self.max_iters = max_iters
        self.nocs_sample_size = nocs_sample_size
        self.ignore_cert_mask = ignore_cert_mask
        self.lsq_normalize_F_matrix = lsq_normalize_F_matrix
        self.device = device
        self.solver = solver
        assert solver in MultiFrameGeometricShapeCorrector.available_solvers

        # free space query points
        if global_nonmnfld_points_voxel_res != 0:
            self.off_surface_coords_global, self.voxel_size, self.voxel_origin = voxelize_cube(
                N=global_nonmnfld_points_voxel_res, cube_center=cube_center, cube_scale=cube_scale
            )
            if self.solver == "GD" or self.solver == "PROJ_GD":
                self.off_surface_coords_global = (
                    torch.tensor(self.off_surface_coords_global).float().to(self.device).unsqueeze(0)
                )
        else:
            self.off_surface_coords_global = None

        # prepare and process shape code matrix
        self.shape_code_library = shape_code_library
        self._ordered_shape_keys = sorted(list(self.shape_code_library.keys()))
        self._shape_code_mat = np.zeros(
            (self.shape_code_library[self._ordered_shape_keys[0]].shape[0], len(self._ordered_shape_keys))
        )
        for i, obj_name in enumerate(self._ordered_shape_keys):
            self._shape_code_mat[:, i] = self.shape_code_library[obj_name]

    def add_nocs(self, unique_ids, nocs: torch.Tensor, cert_mask: torch.Tensor):
        assert len(unique_ids) == nocs.shape[0]
        if self.ignore_cert_mask:
            # use all NOCS instead of only certified NOCS
            for i, uid in enumerate(unique_ids):
                self.nocs_index[uid].append(nocs[i, ...].float().numpy(force=True))
        else:
            for i, uid in enumerate(unique_ids):
                if cert_mask[i, ...]:
                    self.nocs_index[uid].append(nocs[i, ...].float().numpy(force=True))

    def compute_condition_number(self, sdf_model):
        """Compute condition number of accumulated NOCS"""
        conds = []
        shape_code_mat = torch.tensor(self._shape_code_mat[None, :], device=self.device).float()
        for uid in self.nocs_index.keys():
            # sample NOCS
            acc_nocs = self.nocs_index[uid]
            acc_nocs_tensor = np.concatenate(acc_nocs, axis=1)
            acc_nocs_sample_indices = np.random.choice(
                acc_nocs_tensor.shape[1], size=self.nocs_sample_size, replace=False
            )
            acc_nocs_sampled = (
                torch.tensor(np.transpose(acc_nocs_tensor[:, acc_nocs_sample_indices]), device=self.device)
                .unsqueeze(0)
                .float()
            )
            F_all = create_F_matrix(sdf_model, shape_code_mat, acc_nocs_sampled)

            # prepare payload
            ccond = compute_condition_number(F_all)[0]
            ccond["id"] = uid
            ccond["length"] = len(self.nocs_index[uid])

            conds.append(ccond)
        return conds

    def forward(self, unique_ids, shape_code, sdf_model):
        # retrieve accumulated NOCS
        # run corrector
        shape_code_to_correct, nocs_to_use, original_indices = [], [], []
        for i, uid in enumerate(unique_ids):
            acc_nocs = self.nocs_index[uid]
            if len(acc_nocs) < self.min_buffer_size:
                continue
            else:
                acc_nocs_tensor = np.concatenate(acc_nocs, axis=1)
                acc_nocs_sample_indices = np.random.choice(
                    acc_nocs_tensor.shape[1], size=self.nocs_sample_size, replace=False
                )

                acc_nocs_sampled = torch.tensor(np.transpose(acc_nocs_tensor[:, acc_nocs_sample_indices]))
                shape_code_to_correct.append(shape_code[i, ...])
                nocs_to_use.append(acc_nocs_sampled)
                original_indices.append(i)

        if len(shape_code_to_correct) == 0:
            return shape_code, original_indices

        # batch correction
        precrt_shape_code = torch.stack(shape_code_to_correct, dim=0).detach().to(shape_code.device).contiguous()
        nocs_sampled = torch.stack(nocs_to_use, dim=0).detach().to(shape_code.device).contiguous()
        if self.solver == "GD":
            result = shape_recovery_from_pc_gd(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                off_surface_coords_global=self.off_surface_coords_global,
                mnfld_weight=self.mnfld_weight,
                inter_weight=self.inter_weight,
                lr=self.lr,
                max_iters=self.max_iters,
            )
        elif self.solver == "PROJ_GD":
            result = shape_recovery_from_pc_proj_gd(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                off_surface_coords_global=self.off_surface_coords_global,
                # off_surface_coords_global=None,
                mnfld_weight=self.mnfld_weight,
                inter_weight=self.inter_weight,
                lr=self.lr,
                max_iters=self.max_iters,
            )
        elif self.solver == "LSQ_CVXPY":
            result = shape_recovery_from_pc_cvxpy(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                normalize_F_matrix=self.lsq_normalize_F_matrix,
                use_L1_reg=False,
            )
        elif self.solver == "LSQ_CVXPY_L1":
            result = shape_recovery_from_pc_cvxpy(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                normalize_F_matrix=self.lsq_normalize_F_matrix,
                use_L1_reg=True,
            )
        elif self.solver == "LSQ_CVXPY_L1_ONEHOT":
            result = shape_recovery_from_pc_cvxpy(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                normalize_F_matrix=self.lsq_normalize_F_matrix,
                use_L1_reg=True,
                use_onehot=True,
            )
        elif self.solver == "LSQ_CVXPY_L1_INITIAL_CODE_BASIS":
            result = shape_recovery_from_pc_cvxpy(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                normalize_F_matrix=self.lsq_normalize_F_matrix,
                use_L1_reg=True,
                use_onehot=False,
                use_initial_shape_code_basis=True,
            )
        elif self.solver == "LSQ_CVXPY_L1_INITIAL_CODE_BASIS_ONEHOT":
            result = shape_recovery_from_pc_cvxpy(
                sdf_model=sdf_model,
                initial_shape_code=precrt_shape_code,
                nocs=nocs_sampled,
                masks=None,
                shape_code_library=self._shape_code_mat,
                normalize_F_matrix=self.lsq_normalize_F_matrix,
                use_L1_reg=True,
                use_onehot=True,
                use_initial_shape_code_basis=True,
            )
        else:
            raise NotImplementedError

        postcrt_shape_code = shape_code.detach().clone()
        for i, oi in enumerate(original_indices):
            postcrt_shape_code[oi, ...] = result["postcrt_shape_code"][i, ...]
        postcrt_shape_code = postcrt_shape_code.contiguous()

        return postcrt_shape_code, original_indices
