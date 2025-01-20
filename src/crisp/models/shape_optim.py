from collections import defaultdict, deque
from functools import partial

import logging
import numpy as np
import cvxpy as cp
import torch
from torch import nn as nn

# project imports
from crisp.utils.math import project_simplex


def shape_recovery_from_pc_gd(
    sdf_model, initial_shape_code, nocs, masks, off_surface_coords_global, mnfld_weight, inter_weight, lr, max_iters
):
    shape_code = initial_shape_code.detach().clone()
    shape_code.requires_grad_(True)
    params = [{"params": shape_code}]
    optimizer = torch.optim.Adam(params=params, lr=lr)
    best_l = torch.tensor(float("inf"))
    best_shape_code = shape_code.clone()

    def f_sdf_conditioned(shp, x):
        return sdf_model.forward(shape_code=shp, coords=x)

    def loss_fn(shp):
        # mnfld loss: force nocs to be on the surface of the mesh
        mnfld_pred = f_sdf_conditioned(shp, nocs)
        mnfld_loss = torch.abs(mnfld_pred).mean() * mnfld_weight

        # inter loss: force nonmnfld to be nonnegative
        nonmnfld_pred = f_sdf_conditioned(shp, off_surface_coords_global)
        inter_cost = torch.exp(-100 * torch.abs(nonmnfld_pred)).mean() * inter_weight

        return mnfld_loss + inter_cost

    for iter in range(max_iters):
        l = loss_fn(shape_code)

        if l.item() < best_l:
            best_l = l.detach()
            best_shape_code = shape_code.detach().clone()

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % 20 == 0:
            print(f"iter = {iter}, shape_recovery_from_nocs_gd: l: {l}, best l: {best_l}")

    print(f"shape_recovery_from_nocs_gd: iter = {max_iters - 1}  best l: {best_l}")
    if torch.isnan(l) or torch.isinf(best_l) or torch.isnan(best_l):
        print("NAN loss encountered!")

    best_loss = best_l.detach().cpu().item()
    del optimizer, best_l
    torch.cuda.empty_cache()

    return {"postcrt_shape_code": best_shape_code, "best_loss": best_loss}


def shape_recovery_from_pc_proj_gd(
    sdf_model,
    initial_shape_code,
    nocs,
    masks,
    shape_code_library,
    off_surface_coords_global,
    mnfld_weight,
    inter_weight,
    lr,
    max_iters,
):
    """Use projected gradient descent on the convex shape manifold"""
    initial_shape_coeffs = np.linalg.lstsq(
        shape_code_library, torch.transpose(initial_shape_code.detach(), 0, 1).cpu().numpy()
    )[0]
    # initial_shape_coeffs: K x B where B is the batch size
    shape_coeffs = torch.tensor(initial_shape_coeffs).cuda().float()
    shape_coeffs.requires_grad = True
    params = [{"params": shape_coeffs}]
    optimizer = torch.optim.Adam(params=params, lr=lr)
    # optimizer = torch.optim.SGD(params=params, lr=lr, momentum=0.9, nesterov=False)
    best_l = torch.tensor(float("inf"))
    best_shape_coeffs = shape_coeffs.clone()
    shape_code_mat = torch.tensor(shape_code_library).float().to(nocs.device)

    def f_sdf_conditioned(shp, x):
        return sdf_model.forward(shape_code=shp, coords=x)

    def loss_fn(shape_coeffs):
        shape_code = torch.transpose(shape_code_mat @ shape_coeffs, 0, 1)

        # mnfld loss: force nocs to be on the surface of the mesh
        mnfld_pred = f_sdf_conditioned(shape_code, nocs)
        mnfld_loss = torch.abs(mnfld_pred).mean() * mnfld_weight

        # inter loss: force nonmnfld to be nonnegative
        if off_surface_coords_global is not None:
            nonmnfld_pred = f_sdf_conditioned(shape_code, off_surface_coords_global)
            inter_cost = torch.exp(-100 * torch.abs(nonmnfld_pred)).mean() * inter_weight
        else:
            inter_cost = 0

        return mnfld_loss + inter_cost

    for iter in range(max_iters):
        l = loss_fn(shape_coeffs)

        if l.item() < best_l:
            best_l = l.detach()
            best_shape_coeffs = shape_coeffs.detach().clone()

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        # projection to simplex
        temp_coeffs = shape_coeffs.detach().cpu().numpy()
        projected_coeffs = np.transpose(project_simplex(np.transpose(temp_coeffs)))
        shape_coeffs.data = torch.tensor(projected_coeffs).float().to(shape_coeffs.device)

        if iter % 20 == 0:
            print(f"iter = {iter}, shape_recovery_from_nocs_gd: l: {l}, best l: {best_l}")

    print(f"shape_recovery_from_nocs_gd: iter = {max_iters - 1}  best l: {best_l}")
    if torch.isnan(l) or torch.isinf(best_l) or torch.isnan(best_l):
        print("NAN loss encountered!")

    best_loss = best_l.detach().cpu().item()
    postcrt_shape_code = torch.transpose(shape_code_mat @ shape_coeffs, 0, 1)

    del optimizer, best_l, shape_code_mat
    torch.cuda.empty_cache()

    return {
        "postcrt_shape_code": postcrt_shape_code,
        "postcrt_shape_coeffs": best_shape_coeffs.detach().cpu(),
        "best_loss": best_loss,
    }


def create_F_matrix(sdf_model, shape_code_mat, query_points, normalize_by_extent=False):
    """Helper function to create F matrix (LSQ formulation)"""

    def f_sdf_conditioned(shape_code, x):
        return sdf_model.forward(shape_code=shape_code, coords=x)

    B = shape_code_mat.shape[0]
    K = shape_code_mat.shape[2]
    N = query_points.shape[1]

    if normalize_by_extent:
        # TODO: Potentially use quantile to get the 90% max / 10% min
        delta_xyz = (
            torch.max(query_points, dim=1, keepdim=True).values - torch.min(query_points, dim=1, keepdim=True).values
        )
        scales = torch.norm(delta_xyz, dim=2, keepdim=True)
        # set bounds on scales between 1e-4 and 1e4 to avoid numerical issues
        scales = torch.max(torch.ones_like(scales) * 1e-4, torch.min(scales, torch.ones_like(scales) * 1e4))
    else:
        scales = torch.ones((B, 1, 1), device=query_points.device)

    # create F matrix
    F_all = np.zeros((B, N, K))
    with torch.no_grad():
        for k in range(K):
            F_all[:, :, k] = (
                (f_sdf_conditioned(shape_code_mat[:, :, k], query_points) / scales).squeeze(-1).cpu().numpy()
            )
    return F_all


def compute_condition_number(F_all):
    """Compute batchwise condition numbers of F matrix"""
    if F_all.ndim == 2:
        F_all_b = F_all[None, ...]
    else:
        F_all_b = F_all

    conds = []
    for i in range(F_all_b.shape[0]):
        F = F_all_b[i, ...]
        FTF_min_eig = np.linalg.eigvalsh(F.T @ F)[0]
        FTF_cond = np.linalg.cond(F.T @ F)
        FTF_rank = np.linalg.matrix_rank(F.T @ F)
        conds.append(
            {
                "FTF_min_eig": FTF_min_eig,
                "FTF_cond": FTF_cond,
                "FTF_rank": FTF_rank,
            }
        )
    return conds


def shape_recovery_from_pc_cvxpy(
    sdf_model,
    initial_shape_code,
    nocs,
    masks,
    shape_code_library,
    use_L1_reg=False,
    use_onehot=False,
    use_initial_shape_code_basis=False,
    normalize_F_matrix=False,
    L1_weight=5,
):
    # shape coefficients to be optimized
    # note: this is not in simplex
    N = nocs.shape[1]
    og_K, latent_dim = shape_code_library.shape[1], shape_code_library.shape[0]
    if use_initial_shape_code_basis:
        K = og_K + 1
    else:
        K = og_K
    B = nocs.shape[0]

    # create shape code mat
    shape_code_mat = np.zeros((B, latent_dim, K), dtype=np.float32)
    for b in range(B):
        if use_initial_shape_code_basis:
            shape_code_mat[b, :, :og_K] = shape_code_library
            shape_code_mat[b, :, -1] = initial_shape_code[b, ...].detach().cpu().flatten().numpy()
        else:
            shape_code_mat[b, ...] = shape_code_library
    shape_code_mat = torch.tensor(shape_code_mat).to(nocs.device)

    # create F matrix
    # def f_sdf_conditioned(shape_code, x):
    #    return sdf_model.forward(shape_code=shape_code, coords=x)

    # F_all = np.zeros((B, N, K))
    # with torch.no_grad():
    #    for k in range(K):
    #        F_all[:, :, k] = (f_sdf_conditioned(shape_code_mat[:, :, k], nocs)).squeeze(-1).cpu().numpy()

    F_all = create_F_matrix(sdf_model, shape_code_mat, nocs, normalize_by_extent=normalize_F_matrix)

    with torch.no_grad():
        postcrt_shape_codes = torch.zeros((B, latent_dim), device=nocs.device)
        best_shape_coeffs = np.zeros((B, og_K))
        solver_statuses = []
        for b in range(B):
            # formulate the problem
            F = F_all[b, ...]
            c = cp.Variable(K)
            if use_L1_reg:
                cost = cp.sum_squares(F @ c) + L1_weight * cp.norm1(c)
            else:
                cost = cp.sum_squares(F @ c)
            prob = cp.Problem(cp.Minimize(cost), [cp.sum(c) == 1, c >= 0])
            prob.solve(solver="CLARABEL")
            coeffs_sol = c.value.flatten()
            c_shape_code_mat = shape_code_mat[b, ...].cpu().numpy()
            postcrt_shp = c_shape_code_mat @ coeffs_sol
            if use_initial_shape_code_basis:
                # get the coefficients if we are using the original shape code bases
                temp_coeffs = np.linalg.lstsq(shape_code_library, postcrt_shp)[0]
                # project to original simplex
                projected_coeffs = project_simplex(temp_coeffs.reshape((1, -1)))
                # compute newly projected shape code
                postcrt_shp = shape_code_library @ projected_coeffs.flatten()
                coeffs_sol = projected_coeffs.flatten()
            postcrt_shape_codes[b, ...] = torch.tensor(postcrt_shp).float()

            if use_onehot:
                # project to onehot vector and recompute post correction shape code
                max_index = np.argmax(coeffs_sol)
                best_shape_coeffs[b, max_index] = 1
                postcrt_shape_codes[b, ...] = torch.tensor(shape_code_library @ best_shape_coeffs[b, ...]).float()
            else:
                best_shape_coeffs[b, :] = coeffs_sol

            solver_statuses.append((prob.status, prob.solver_stats))

    return {
        "postcrt_shape_code": postcrt_shape_codes,
        "postcrt_shape_coeffs": torch.tensor(best_shape_coeffs),
        "solver_statuss": solver_statuses,
    }
