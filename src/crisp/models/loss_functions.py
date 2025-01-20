import torch
import torch.nn.functional as F
from crisp.utils import diff_operators
from crisp.utils.math import (
    make_se3_batched,
)
from crisp.models.registration import (
    arun_ransac_batched,
)


def nocs_penalty_relu_l1_loss(nocs: torch.Tensor):
    """Regularization loss on NOCS using ReLU to enforce NOCS lie within [-1, 1]

    Parameters
    ----------
    nocs: (B, 3, N)
    """
    penalty = torch.mean(torch.relu(torch.abs(nocs) - 1), dim=2).mean()
    return penalty


def nocs_penalty_relu_l2_loss(nocs: torch.Tensor):
    """Regularization loss on NOCS using ReLU to enforce NOCS lie within [-1, 1]

    Parameters
    ----------
    nocs: (B, 3, N)
    """
    penalty = torch.mean(torch.relu(torch.abs(nocs) - 1) ** 2, dim=2).mean()
    return penalty


def nocs_penalty_max_l2_loss(nocs: torch.Tensor, weights=None):
    """Regularization loss on NOCS using max(0, g(x)) trick.

    Parameters
    ----------
    nocs: (B, 3, N)
    """
    if weights is None:
        penalty = torch.mean(torch.maximum(torch.abs(nocs) - 1, torch.tensor(0)) ** 2, dim=2).mean()
    else:
        w_sum = weights.sum(dim=1, keepdims=True)
        penalty = (
            ((torch.maximum(torch.abs(nocs) - 1, torch.tensor(0)) ** 2) * weights.unsqueeze(1)).sum(dim=1) / w_sum
        ).mean()
    return penalty


def pseudo_gt_loss(nocs_label, shape_label, nocs_est, shape_est):
    """SSL loss using pseudo GT labels for NOCS and shape code to supervise

    Parameters
    ----------
    nocs_label
    shape_label
    nocs_est
    shape_est
    """
    return


def nocs_depths_loss(nocs, depth_pc, nocs_T_depth, inlier_thres):
    """Least-squares loss between NOCS and transformed depths"""
    depth_coords = torch.bmm(nocs_T_depth[..., :3, :3], depth_pc) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
    l = torch.clamp(((nocs - depth_coords) ** 2).sum(dim=1), max=inlier_thres**2).mean()
    return l


def sdf_nocs_loss(f_sdf_conditioned, nocs: torch.Tensor, weights: torch.Tensor, reduction="mean"):
    """SDF values of NOCS"""
    nocs_coords = nocs.transpose(-1, -2)
    nocs_sdf = f_sdf_conditioned(nocs_coords)
    a = torch.abs(nocs_sdf)
    if reduction == "mean":
        if weights is None:
            w_sum = a.shape[1]
            return (a.sum(dim=1) / w_sum).mean()
        else:
            w_sum = weights.sum(dim=1)
            return ((a.squeeze(2) * weights).sum(dim=1) / w_sum).mean()
    elif reduction == "sum":
        if weights is None:
            w_sum = a.shape[1]
            return a.sum(dim=1) / w_sum
        else:
            w_sum = weights.sum(dim=1)
            return (a.squeeze(2) * weights).sum(dim=1) / w_sum
    elif reduction == "none":
        return a


def sdf_input_loss(
    f_sdf_conditioned,
    depth_pc: torch.Tensor,
    nocs_T_depth: torch.Tensor,
    weights: torch.Tensor,
    threshold=torch.tensor(float("Inf")),
):
    """SDF values of transformed depths"""
    depth_coords = (
        torch.bmm(nocs_T_depth[..., :3, :3], depth_pc) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
    ).transpose(-1, -2)
    depth_sdf = f_sdf_conditioned(depth_coords)
    th = torch.tensor(threshold).to(depth_pc.device)
    b = torch.clamp(depth_sdf**2, max=th**2)

    if weights is None:
        w_sum = b.shape[1]
        return (b.sum(dim=1) / w_sum).mean()
    else:
        w_sum = weights.sum(dim=1)
        return ((b.squeeze(2) * weights).sum(dim=1) / w_sum).mean()


def sdf_input_trimmed_loss(
    f_sdf_conditioned,
    depth_pc: torch.Tensor,
    nocs_T_depth: torch.Tensor,
    weights: torch.Tensor,
    trim_quantile=0.9,
    reduction="mean",
):
    """SDF values of transformed depths; trimming"""
    depth_coords = (
        torch.bmm(nocs_T_depth[..., :3, :3], depth_pc) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
    ).transpose(-1, -2)
    depth_sdf = f_sdf_conditioned(depth_coords)
    depth_sdf_sq = depth_sdf**2
    th = torch.quantile(depth_sdf_sq, trim_quantile, dim=1, keepdim=True)
    b = torch.clamp(depth_sdf_sq, max=th)

    if reduction == "mean":
        if weights is None:
            w_sum = b.shape[1]
            return (b.sum(dim=1) / w_sum).mean()
        else:
            w_sum = weights.sum(dim=1)
            return ((b.squeeze(2) * weights).sum(dim=1) / w_sum).mean()
    elif reduction == "sum":
        if weights is None:
            w_sum = b.shape[1]
            return b.sum(dim=1) / w_sum
        else:
            w_sum = weights.sum(dim=1)
            return (b.squeeze(2) * weights).sum(dim=1) / w_sum
    elif reduction == "none":
        return b


def snc_robust_loss(
    f_sdf_conditioned,
    nocs: torch.Tensor,
    depth_pc: torch.Tensor,
    nocs_T_depth: torch.Tensor,
    weights: torch.Tensor,
    lambda_nocs=1.0,
    lambda_depths=1.0,
    threshold=torch.tensor(float("Inf")),
):
    """
    SDF NOCS Consistency Loss. For use in corrector.

    \lossShpNocsConsist =
    \frac{1}{\nrNocs} \sum_{i=1 \ldots \nrNocs} (\rho (\fsdf (\detNocs
                                          + \nocsCorrection \mid \detShp + \shpCorrection )))
    + \rho(\fsdf (\MT(\nocsCorrection) \depthPt_{i} \mid \detShp + \shpCorrection ))

    Assumptions:
    - NOCS and mesh reconstructed from the SDF function share the same coordinate frame
    - f_sdf_conditioned has been conditioned by some latent shape code
    - depth_pc are point clouds produced from depth measurements (1-to-1 correspondences with NOCS)
    - nocs_T_depth * depth_pc is in the NOCS frame.
    - everything is batched
    - use TLS threshold

    Parameters
    ----------
    f_sdf_conditioned
    nocs: (B, 3, N)
    depth_pc: (B, 3, N)
    nocs_T_depth: (B, 4, 4)
    weights: (B, N)
    lambda_nocs
    lambda_depths
    threshold
    """
    nocs_coords = nocs.transpose(-1, -2)
    depth_coords = (
        torch.bmm(nocs_T_depth[..., :3, :3], depth_pc) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
    ).transpose(-1, -2)
    nocs_sdf = f_sdf_conditioned(nocs_coords)
    depth_sdf = f_sdf_conditioned(depth_coords)

    th = torch.tensor(threshold).to(nocs.device)
    # a = torch.clamp(nocs_sdf**2, max=th**2)
    a = nocs_sdf**2
    b = torch.clamp(depth_sdf**2, max=th**2)
    if weights is None:
        w_sum = a.shape[1]
        return lambda_nocs * ((a).sum(dim=1) / w_sum).mean() + lambda_depths * ((b).sum(dim=1) / w_sum).mean()
    else:
        w_sum = weights.sum(dim=1)
        return (
            lambda_nocs * ((a.squeeze(2) * weights).sum(dim=1) / w_sum).mean()
            + lambda_depths * ((b.squeeze(2) * weights).sum(dim=1) / w_sum).mean()
        )


def snc_robust_loss_with_gradients(
    f_sdf_conditioned,
    nocs,
    depth_pc,
    nocs_T_depth,
    lambda_nocs=1.0,
    lambda_depths=1.0,
    threshold=torch.tensor(float("Inf")),
):
    """
    SDF NOCS Consistency Loss. For use in corrector. Returns spatial gradients.

    \lossShpNocsConsist =
    \frac{1}{\nrNocs} \sum_{i=1 \ldots \nrNocs} (\rho (\fsdf (\detNocs
                                          + \nocsCorrection \mid \detShp + \shpCorrection )))
    + \rho(\fsdf (\MT(\nocsCorrection) \depthPt_{i} \mid \detShp + \shpCorrection ))

    Assumptions:
    - NOCS and mesh reconstructed from the SDF function share the same coordinate frame
    - f_sdf_conditioned has been conditioned by some latent shape code
    - depth_pc are point clouds produced from depth measurements (1-to-1 correspondences with NOCS)
    - nocs_T_depth * depth_pc is in the NOCS frame.
    - everything is batched
    - use TLS threshold

    Parameters
    ----------
    f_sdf_conditioned
    nocs: (B, 3, N)
    depth_pc: (B, 3, N)
    nocs_T_depth: (B, 4, 4)
    lambda_nocs
    lambda_depths
    threshold
    """
    nocs_coords = nocs.transpose(-1, -2).requires_grad_(True)
    depth_coords = (
        (torch.bmm(nocs_T_depth[..., :3, :3], depth_pc) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1))
        .transpose(-1, -2)
        .requires_grad_(True)
    )

    nocs_sdf = f_sdf_conditioned(nocs_coords)
    depth_sdf = f_sdf_conditioned(depth_coords)

    nocs_gradients = diff_operators.gradient(nocs_sdf, nocs_coords)
    depth_gradients = diff_operators.gradient(depth_sdf, depth_coords)

    th = torch.tensor(threshold).to(nocs.device)
    a = torch.clamp(nocs_sdf**2, max=th)
    b = torch.clamp(depth_sdf**2, max=th)
    return lambda_nocs * a.mean() + lambda_depths * b.mean(), nocs_gradients, depth_gradients


def sgr_robust_loss(gradients, nonmnfld_pts, grad_weight=50.0, inter_weight=50.0):
    """
    SDF Geometric Regularization Loss (norms of spatial gradients equal to one and non-manifold non-zero loss).
    For use in corrector.

    \lossShpGeom = \int_{\sdfDomain \setminus \sdfManifoldDomain}
                          \exp ( - | \fsdf (\mathbf{x} \mid \detShp + \shpCorrection)|) dx
                  +\int_{\sdfDomain}\||\nabla_{\mathbf{x}} \fsdf(\mathbf{x} \mid \detShp + \shpCorrection)|-1\| dx

    Assumptions:
    """
    grad_cost = torch.abs(gradients.norm(dim=-1) - 1).mean() * grad_weight
    inter_cost = torch.exp(-100 * torch.abs(nonmnfld_pts)).mean() * inter_weight
    return grad_cost + inter_cost


def nocs_loss(pred_nocs, exp_nocs, mask, threshold=0.1):
    """Loss on NOCS
    Based on Wild6D & original NOCS paper
    """
    # 0.5(x-y)^2 / \beta, if |x-y| < \beta
    # |x-y|-0.5 * \beta,  otherwise
    diff = torch.abs(pred_nocs[:, :3, ...] - exp_nocs[:, :3, ...])
    less = diff**2 / (2.0 * threshold)
    higher = diff - threshold / 2.0

    loss = torch.where(diff > threshold, higher, less)
    loss = torch.sum(loss, dim=1, keepdim=True)
    loss = torch.mean(loss[mask])

    return loss


def nocs_loss_clamped(pred_nocs, exp_nocs, mask, threshold=0.1, nocs_min=0, nocs_max=1):
    """Loss on NOCS with hard clamping on the GT NOCS"""
    # 0.5(x-y)^2 / \beta, if |x-y| < \beta
    # |x-y|-0.5 * \beta,  otherwise
    diff = torch.abs(pred_nocs[:, :3, ...] - exp_nocs[:, :3, ...])
    less = diff**2 / (2.0 * threshold)
    higher = diff - threshold / 2.0

    # valid exp_nocs mask (nocs values should be between 0 and 1)
    exp_nocs_valid_mask = torch.logical_and(exp_nocs[:, :3, ...] > nocs_min, exp_nocs[:, :3, ...] < nocs_max)
    exp_nocs_valid_mask = torch.sum(exp_nocs_valid_mask, dim=1, keepdim=True)
    exp_nocs_valid_mask = exp_nocs_valid_mask == 3
    final_mask = torch.logical_and(mask, exp_nocs_valid_mask)

    loss = torch.where(diff > threshold, higher, less)
    loss = torch.sum(loss, dim=1, keepdim=True)
    loss = torch.mean(loss[final_mask])

    return loss


def igr_sdf_loss(
    mnfld_pred,
    mnfld_grad,
    nonmnfld_pred,
    nonmnfld_grad,
    mnfld_normals=None,
    sdf_weight=3e3,
    normals_weight=1e2,
    grad_weight=5e1,
    inter_weight=1e2,
    **kwargs,
):
    """Loss function described in Implicit Geometric Regularization for Learning Shapes, Gropp et. al.

    loss = SDF term (points on manifolds) + normal term (points on manifolds) + gradient/Eikonal term (all points)
    """
    # manifold loss
    mnfld_loss = (mnfld_pred.abs()).mean()

    # Eikonal loss
    grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean() + ((mnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

    # force non-mnfld points to be non zero
    inter_loss = torch.exp(-1e2 * torch.abs(nonmnfld_pred)).mean()

    loss = sdf_weight * mnfld_loss + grad_weight * grad_loss + inter_weight * inter_loss

    # normals loss
    if mnfld_normals is not None:
        # normals_loss = ((mnfld_grad - mnfld_normals).abs()).norm(2, dim=1).mean()
        # normals_loss = (mnfld_grad - mnfld_normals).norm(2, dim=1).mean()
        normals_loss = (1 - F.cosine_similarity(mnfld_grad, mnfld_normals, dim=-1)[..., None]).mean()
        loss = loss + normals_weight * normals_loss
    else:
        normals_loss = torch.zeros(1)

    return loss, {"sdf": mnfld_loss, "normal": normals_loss, "grad": grad_loss, "inter": inter_loss}


def siren_udf_loss(
    mnfld_pred,
    mnfld_grad,
    nonmnfld_pred,
    mnfld_normals,
    gradients,
    sdf_weight=3e3,
    inter_weight=1e2,
    normal_weight=5e1,
    grad_weight=5e1,
    **kwargs,
):
    """A loss forcing the implicit net to learn an unsigned distance function."""
    mnfld_cost = torch.abs(mnfld_pred).mean() * sdf_weight
    normal_cost = (1 - F.cosine_similarity(mnfld_grad, mnfld_normals, dim=-1)[..., None]).mean() * normal_weight
    grad_cost = torch.abs(gradients.norm(dim=-1) - 1).mean() * grad_weight

    # force nonmnfld to be positive
    inter_cost = torch.exp(-nonmnfld_pred).mean() * inter_weight

    return mnfld_cost + inter_cost + normal_cost + grad_cost, {
        "sdf": mnfld_cost,  # 1e4      # 3e3
        "inter": inter_cost,  # 1e2                   # 1e3
        "normal": normal_cost,  # 1e2
        "grad": grad_cost,
    }  # 1e1      # 5e1


def siren_sdf_fast_loss(
    mnfld_pred,
    mnfld_grad,
    nonmnfld_pred,
    mnfld_normals,
    gradients,
    sdf_weight=3e3,
    inter_weight=1e2,
    normal_weight=5e1,
    grad_weight=5e1,
    **kwargs,
):
    """SDF Loss used in SIREN paper. Modified to remove torch.where
    See paper Section 4.2.

    Parameters
    ----------
    coords : (1, N, 3)
    pred_sdf : (1, N, 1)
    gt_sdf : (1, N, 1)
        -1: not on surface, 0: on surface
    gt_normals : (1, N, 3)
    sdf_weight : hyperparam
    inter_weight : hyperparam
    normal_weight : hyperparam
    grad_weight : hyperparam
    """
    sdf_cost = torch.abs(mnfld_pred).mean() * sdf_weight
    # normal_cost = (mnfld_grad - mnfld_normals).norm(2, dim=1).mean() * normal_weight
    # normal_cost = (1 - (mnfld_grad * mnfld_normals).sum(dim=-1)).mean() * normal_weight
    if mnfld_normals is not None:
        normal_cost = (1 - F.cosine_similarity(mnfld_grad, mnfld_normals, dim=-1)[..., None]).mean() * normal_weight
    else:
        normal_cost = torch.tensor(0).to(mnfld_pred.device)
    grad_cost = torch.abs(gradients.norm(dim=-1) - 1).mean() * grad_weight

    # inter_cost = torch.exp(-1e2 * torch.abs(nonmnfld_pred)).mean() * inter_weight
    inter_cost = torch.exp(-100 * torch.abs(nonmnfld_pred)).mean() * inter_weight

    return sdf_cost + inter_cost + normal_cost + grad_cost, {
        "sdf": sdf_cost,  # 1e4      # 3e3
        "inter": inter_cost,  # 1e2                   # 1e3
        "normal": normal_cost,  # 1e2
        "grad": grad_cost,
    }  # 1e1      # 5e1


def siren_sdf_loss(
    pred_sdf,
    gradient,
    gt_sdf,
    gt_normals,
    sdf_weight=3e3,
    inter_weight=1e2,
    normal_weight=1e2,
    grad_weight=5e1,
    **kwargs,
):
    """SDF Loss used in SIREN paper. Original implementation.
    See paper Section 4.2.

    Parameters
    ----------
    coords : (1, N, 3)
    pred_sdf : (1, N, 1)
    gt_sdf : (1, N, 1)
        -1: not on surface, 0: on surface
    gt_normals : (1, N, 3)
    sdf_weight : hyperparam
    inter_weight : hyperparam
    normal_weight : hyperparam
    grad_weight : hyperparam
    """
    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient[..., :1]),
    )
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    sdf_cost = torch.abs(sdf_constraint).mean() * sdf_weight
    inter_cost = inter_constraint.mean() * inter_weight
    normal_cost = normal_constraint.mean() * normal_weight
    grad_cost = grad_constraint.mean() * grad_weight

    # Exp      # Lapl
    # -----------------
    return sdf_cost + inter_cost + normal_cost + grad_cost, {
        "sdf": sdf_cost,  # 1e4      # 3e3
        "inter": inter_cost,  # 1e2                   # 1e3
        "normal_constraint": normal_cost,  # 1e2
        "grad_constraint": grad_cost,
    }  # 1e1      # 5e1


def metric_sdf_loss(
    sdf_pred, sdf_gt, mnfld_grad, mnfld_normals, gradients, sdf_weight=3e3, normal_weight=5e1, grad_weight=5e1, **kwargs
):
    """SDF loss involving GT SDF supervision + gradient regularization
    See paper Section 4.2.

    Parameters
    ----------
    coords : (1, N, 3)
    pred_sdf : (1, N, 1)
    gt_sdf : (1, N, 1)
        -1: not on surface, 0: on surface
    gt_normals : (1, N, 3)
    sdf_weight : hyperparam
    inter_weight : hyperparam
    normal_weight : hyperparam
    grad_weight : hyperparam
    """
    sdf_cost = (torch.abs(sdf_gt - sdf_pred)).mean() * sdf_weight
    if mnfld_normals is not None:
        normal_cost = (1 - F.cosine_similarity(mnfld_grad, mnfld_normals, dim=-1)[..., None]).mean() * normal_weight
    else:
        normal_cost = torch.tensor(0, device=sdf_pred.device)
    grad_cost = torch.abs(gradients.norm(dim=-1) - 1).mean() * grad_weight

    return sdf_cost + normal_cost + grad_cost, {
        "sdf": sdf_cost,  # 1e4      # 3e3
        "normal": normal_cost,  # 1e2
        "grad": grad_cost,
    }  # 1e1      # 5e1


def metric_sdf_loss_no_gradient(sdf_pred, sdf_gt, sdf_weight=3e3, **kwargs):
    """SDF loss involving GT SDF supervision + gradient regularization
    See paper Section 4.2.

    Parameters
    ----------
    coords : (1, N, 3)
    pred_sdf : (1, N, 1)
    gt_sdf : (1, N, 1)
        -1: not on surface, 0: on surface
    sdf_weight : hyperparam
    inter_weight : hyperparam
    """
    sdf_cost = (torch.abs(sdf_gt - sdf_pred)).mean() * sdf_weight
    return sdf_cost


def implicit_loss_helper(
    pred_sdf,
    gradient,
    gt_normals,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    recons_loss_fn=None,
    opt=None,
    **kwargs,
):
    # nocs loss
    nocs_l = nocs_loss(pred_nocs=pred_nocs, exp_nocs=gt_nocs, mask=mask, threshold=opt.loss_nocs_threshold)

    # recons loss
    mnfld_grad, nonmnfld_grad = (gradient[:, :on_surface_cutoff, :], gradient[:, on_surface_cutoff:, :])
    recons_l, recons_loss_terms = recons_loss_fn(
        mnfld_pred=pred_sdf[:, :on_surface_cutoff, :],
        mnfld_grad=mnfld_grad,
        nonmnfld_pred=pred_sdf[:, on_surface_cutoff:, :],
        gradients=gradient,
        mnfld_normals=gt_normals[:, :on_surface_cutoff, :],
        sdf_weight=opt.recons_df_weight,
        inter_weight=opt.recons_inter_weight,
        normal_weight=opt.recons_normal_weight,
        grad_weight=opt.recons_grad_weight,
    )

    # TODO: Add batch consistent term for UDF
    total_l = opt.nocs_loss_weight * nocs_l + opt.nocs_loss_weight * recons_l
    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
        "recons_loss_terms": recons_loss_terms,
    }


def explicit_loss_helper(
    pred_sdf,
    gt_sdf,
    gradient,
    gt_normals,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    opt=None,
    **kwargs,
):
    # nocs loss
    nocs_l = nocs_loss_clamped(
        pred_nocs=pred_nocs,
        exp_nocs=gt_nocs,
        mask=mask,
        threshold=opt.loss_nocs_threshold,
        nocs_min=opt.nocs_min,
        nocs_max=opt.nocs_max,
    )

    # sdf loss
    mnfld_grad, nonmnfld_grad = (gradient[:, :on_surface_cutoff, :], gradient[:, on_surface_cutoff:, :])
    mnfld_normals = None if gt_normals is None else gt_normals[:, :on_surface_cutoff, :]
    recons_l, recons_loss_terms = metric_sdf_loss(
        sdf_pred=pred_sdf.squeeze(2),
        sdf_gt=gt_sdf,
        mnfld_grad=mnfld_grad,
        gradients=gradient,
        mnfld_normals=mnfld_normals,
        sdf_weight=opt.recons_df_weight,
        normal_weight=opt.recons_normal_weight,
        grad_weight=opt.recons_grad_weight,
    )
    total_l = opt.nocs_loss_weight * nocs_l + opt.recons_loss_weight * recons_l
    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
        "recons_loss_terms": recons_loss_terms,
    }


def two_cls_contrastive_explicit_loss_helper(
    pred_sdf,
    gt_sdf,
    gradient,
    gt_normals,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    shape_code,
    contrastive_same_object_weight=1.0,
    contrastive_different_object_weight=1.0,
    opt=None,
):
    lterms = explicit_loss_helper(
        pred_sdf=pred_sdf,
        gt_sdf=gt_sdf,
        gradient=gradient,
        gt_normals=gt_normals,
        on_surface_cutoff=on_surface_cutoff,
        global_nonmnfld_start=global_nonmnfld_start,
        global_nonmnfld_count=global_nonmnfld_count,
        pred_nocs=pred_nocs,
        gt_nocs=gt_nocs,
        mask=mask,
        opt=opt,
    )
    total_l = lterms["total_loss"]
    nocs_l = lterms["nocs_loss"]
    recons_l = lterms["recons_loss"]
    recons_loss_terms = lterms["recons_loss_terms"]

    split_idx = int(pred_nocs.shape[0] / opt.num_classes_per_batch)
    cls_shape_codes_1 = shape_code[:split_idx, ...]
    cls_shape_codes_2 = shape_code[split_idx:, ...]
    mean_cls_shp_code_1 = torch.mean(cls_shape_codes_1, dim=0, keepdim=True)
    mean_cls_shp_code_2 = torch.mean(cls_shape_codes_2, dim=0, keepdim=True)

    # MSE loss between mean shape code and each shape code
    if opt.recons_shape_code_normalization is None or opt.recons_shape_code_normalization == "max_norm":
        same_cls_loss_1 = torch.mean(torch.sum(torch.square(cls_shape_codes_1 - mean_cls_shp_code_1), dim=0))
        same_cls_loss_2 = torch.mean(torch.sum(torch.square(cls_shape_codes_2 - mean_cls_shp_code_2), dim=0))
    elif opt.recons_shape_code_normalization == "sphere":
        same_cls_loss_1 = torch.mean(1 - torch.nn.functional.cosine_similarity(mean_cls_shp_code_1, cls_shape_codes_1))
        same_cls_loss_2 = torch.mean(1 - torch.nn.functional.cosine_similarity(mean_cls_shp_code_2, cls_shape_codes_2))
    else:
        raise NotImplementedError

    # push different shape codes away from each other
    diff_cls_loss = torch.exp(-100 * torch.sum(torch.abs(mean_cls_shp_code_1 - mean_cls_shp_code_2)))

    total_l += (
        contrastive_same_object_weight * (same_cls_loss_1 + same_cls_loss_2)
        + contrastive_different_object_weight * diff_cls_loss
    )

    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
        "same_cls_loss": same_cls_loss_1 + same_cls_loss_2,
        "diff_cls_loss": diff_cls_loss,
        "recons_loss_terms": recons_loss_terms,
    }


def inv_transformed_depth_sdf_loss(
    nocs, f_sdf, depth_pcs, masks, reg_inlier_thres, max_ransac_iters, trim_quantile=1, reduction="mean"
):
    nocs_R_cam, nocs_t_cam, best_inliers, _ = arun_ransac_batched(
        source_points=depth_pcs,
        target_points=nocs,
        masks=masks,
        inlier_thres=reg_inlier_thres,
        confidence=0.99,
        max_iters=max_ransac_iters,
    )
    nocs_T_cam = make_se3_batched(nocs_R_cam, nocs_t_cam)

    l = sdf_input_trimmed_loss(
        f_sdf_conditioned=f_sdf,
        depth_pc=depth_pcs,
        nocs_T_depth=nocs_T_cam,
        weights=masks,
        trim_quantile=trim_quantile,
        reduction=reduction,
    )

    return l
