import torch.nn as nn
import torch
import numpy as np

from crisp.models.shape_optim import create_F_matrix, compute_condition_number


def clamp_by_fixed_threshold(values, clamp_thres):
    """Clamping by a fixed threshold. All clamped distances will be zero, which
    won't affect certification (always passes epsilon check).
    """
    # true means the point is not clamped
    clamp_mask = torch.le(values, clamp_thres)
    return values * clamp_mask, clamp_mask


def eps_bound_by_max(sq_dists, valid_mask, sq_eps):
    """Eps bound through the max"""
    return (sq_dists * valid_mask).max(dim=1)[0] < sq_eps


def eps_bound_by_avg(sq_dists, valid_mask, sq_eps):
    """Eps bound through averaging"""
    return sq_dists.sum(dim=1) / valid_mask.sum(dim=1) < sq_eps


def eps_bound_by_quantile(scores, valid_mask, quantile, eps):
    """Eps bound through quantiles.
    We bound by asserting at least quantile fraction of points are below eps.
    In the case of 0.5 quantile, it is equivalent to saying that the median should be below eps.
    In the case of 1 quantile, it is equivalent to saying that the max should be below eps.
    """
    if valid_mask is not None:
        eps_quantile = torch.sum(torch.le(scores, eps) * valid_mask, dim=1) / valid_mask.sum(dim=1)
    else:
        eps_quantile = torch.sum(torch.le(scores, eps), dim=1) / scores.shape[1]
    return eps_quantile >= quantile


def certify_pc_by_chamfer_distances(eps_bound_fun, pc_sq_dists, pc_valid_mask, sq_epsilon):
    """Certify point clouds

    Args:
        eps_bound_fun:
        pc_sq_dists: (B, N)
        pc_valid_mask:
        sq_epsilon:
    """
    pc_flag = eps_bound_fun(pc_sq_dists, pc_valid_mask, sq_epsilon)
    return pc_flag


def certify_kp_by_distances(kp_sq_dists, sq_epsilon):
    """Certify keypoints

    Args:
        kp_sq_dists: (B, num keypoints)
        sq_epsilon:
    """
    kp_flag = kp_sq_dists.max(dim=1)[0] < sq_epsilon
    return kp_flag


class FrameCertifier:
    """Framewise certifier"""

    def __init__(
        self,
        model,
        depths_clamp_thres=10,
        depths_quantile=0.9,
        depths_eps=0.1,
        degen_min_eig_thres=1e-4,
        use_degen_cert=False,
        shape_code_library=None,
    ):
        super().__init__()
        self.model = model

        # certification parameters
        # for the transformed depth point clouds in the NOCS frame
        self.depths_clamp_thres = depths_clamp_thres
        self.depths_quantile = depths_quantile
        self.depths_eps = depths_eps

        # degeneracy certificate parameters
        self.degen_min_eig_thres = degen_min_eig_thres

        # observably correct and degeneracy certificates
        self.use_degen_cert = use_degen_cert
        self.shape_code_library = shape_code_library
        if self.shape_code_library is not None:
            ordered_shape_keys = sorted(list(self.shape_code_library.keys()))
            self._shape_code_mat = np.zeros(
                (self.shape_code_library[ordered_shape_keys[0]].shape[0], len(ordered_shape_keys))
            )
            for i, obj_name in enumerate(ordered_shape_keys):
                self._shape_code_mat[:, i] = self.shape_code_library[obj_name]
        return

    def certify(self, nocs_T_depth, nocs, shape_code, pcs):
        """Return indices of certified samples in batch"""
        depth_coords = (
            torch.bmm(nocs_T_depth[..., :3, :3], pcs) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
        ).transpose(-1, -2)
        scores = torch.abs(self.model.recons_net.forward(shape_code=shape_code, coords=depth_coords))
        clamped_scores, clamped_mask = clamp_by_fixed_threshold(scores.squeeze(2), clamp_thres=self.depths_clamp_thres)
        cert = eps_bound_by_quantile(
            clamped_scores, valid_mask=None, quantile=self.depths_quantile, eps=self.depths_eps
        )

        B = nocs.shape[0]
        if self.use_degen_cert:
            # compute degeneracy condition number
            shape_code_mat = (
                torch.tensor(self._shape_code_mat, device=nocs.device).float().unsqueeze(0).expand(B, -1, -1)
            )
            sf_F = create_F_matrix(
                self.model.recons_net, shape_code_mat=shape_code_mat,
                query_points=torch.transpose(nocs, 1, 2),
                normalize_by_extent=True
            )
            try:
                sf_degen_conds = compute_condition_number(F_all=sf_F)
                sf_ftf_min_eig = torch.tensor([x["FTF_min_eig"] for x in sf_degen_conds])
                degen_cert = torch.tensor(sf_ftf_min_eig > self.degen_min_eig_thres).bool().to(cert.device)
                cert = torch.logical_and(cert, degen_cert)
            except Exception as e:
                print(f"Error encountered while computing degeneracy condition number: {e}")

        return cert, clamped_scores
