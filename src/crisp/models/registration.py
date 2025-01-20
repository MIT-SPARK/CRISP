import os
import time
import torch
import numpy as np

try:
    import teaserpp_python
except:
    print("Missing TEASER++. TEASER++ based registration will not work.")

from crisp.utils.math import (
    instance_depth_to_point_cloud_torch,
    project_SO3,
    batch_var,
    sample_within_nonzero_masks,
)

USE_TORCH_COMPILE = os.getenv("CORE_USE_TORCH_COMPILE", "True").lower() in ("true", "1", "t")
print("USE_TORCH_COMPILE: ", USE_TORCH_COMPILE)


def wahba(source_points, target_points, device_=None):
    """
    inputs:
    source_points: torch.tensor of shape (B, 3, N)
    target_points: torch.tensor of shape (B, 3, N)

    where
        B = batch size
        N = number of points in each point set

    output:
    R   : torch.tensor of shape (B, 3, 3)
    """
    with torch.cuda.amp.autocast(enabled=False):
        batch_size = source_points.shape[0]

        if device_ == None:
            device_ = source_points.device

        mat = target_points @ source_points.transpose(-1, -2)  # (B, 3, 3)
        U, S, Vh = torch.linalg.svd(mat)

        D = torch.eye(3).to(device=device_).to(dtype=source_points.dtype)  # (3, 3)
        D = D.unsqueeze(0)  # (1, 3, 3)
        D = D.repeat(batch_size, 1, 1)  # (B, 3, 3)

        D[:, 2, 2] = torch.linalg.det(U) * torch.linalg.det(Vh)

    return U @ D @ Vh  # (B, 3, 3)


def weighted_umeyama_batched(source_points, target_points, weights):
    """Run Umeyama algorithm to estimate s, R, t (weighted)
    target = s*R*source + t
    Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf

    Note: this is a batched version
    """
    with torch.cuda.amp.autocast(enabled=False):
        # centroids
        source_points_ave = torch.sum(source_points * weights.unsqueeze(1), dim=2) / weights.sum(dim=1, keepdim=True)
        target_points_ave = torch.sum(target_points * weights.unsqueeze(1), dim=2) / weights.sum(dim=1, keepdim=True)

        # getting the rotation
        source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
        target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

        # get rotation
        mat = (target_points_centered * weights.unsqueeze(1)) @ source_points_centered.transpose(-1, -2)  # (B, 3, 3)
        R, _, S, _, d = project_SO3(mat / weights.sum(dim=1, keepdim=True).unsqueeze(-1))
        # S.clone() is necessary for avoiding backprop error due to in-place operations
        D = S.clone()
        D[:, -1] *= d

        # scale
        varP = batch_var(source_points, weights=weights).sum(dim=1, keepdim=True).unsqueeze(-1)
        s = 1.0 / varP * D.sum(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1)

        # getting the translation
        t = target_points_ave.unsqueeze(-1) - s * R @ source_points_ave.unsqueeze(-1)

    return s, R, t


@torch.compile(mode="default", disable=not USE_TORCH_COMPILE)
def umeyama_batched(source_points, target_points):
    """Run Umeyama algorithm to estimate s, R, t (unweighted)
    target = s*R*source + t
    Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf

    Note: this is a batched version
    """
    og_dtype = source_points.dtype
    with torch.cuda.amp.autocast(enabled=False):
        assert source_points.ndim == target_points.ndim == 3

        N = source_points.shape[2]
        # centroids
        source_points_ave = (torch.sum(source_points, dim=2) / N).float()
        target_points_ave = (torch.sum(target_points, dim=2) / N).float()

        # getting the rotation
        source_points_centered = (source_points - source_points_ave.unsqueeze(-1)).float()  # (B, 3, N)
        target_points_centered = (target_points - target_points_ave.unsqueeze(-1)).float()  # (B, 3, N)

        # get rotation
        mat = (
            target_points_centered @ source_points_centered.transpose(-1, -2).contiguous() / N
        ).contiguous()  # (B, 3, 3)
        R, _, S, _, d = project_SO3(mat)
        # S.clone() is necessary for avoiding backprop error due to in-place operations
        D = S.clone()
        D[:, -1] *= d

        # scale
        varP = torch.var(source_points, dim=2, correction=0).sum(dim=1, keepdim=True).unsqueeze(-1)
        s = 1.0 / varP * D.sum(dim=1, keepdim=True).unsqueeze(1)  # (B, 1, 1)

        # getting the translation
        t = target_points_ave.unsqueeze(-1) - s * R @ source_points_ave.unsqueeze(-1)

    s, R, t = s.to(og_dtype), R.to(og_dtype), t.to(og_dtype)
    return s, R, t


@torch.compile(mode="default", disable=not USE_TORCH_COMPILE)
def umeyama_ransac_batched(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    inlier_thres: torch.Tensor,
    masks: torch.Tensor = None,
    confidence=0.99,
    max_iters=100,
    sample_size=5,
):
    """Batched RANSAC version of Umeyama.
    Terminate if all instances in the batch satisfy the convergence cost criteria.

    Parameters
    ----------
    source_points
    target_points
    masks: (B, pc_size)
    inlier_thres
    confidence
    max_iters
    sample_size
    """
    B, n_points, device = (source_points.shape[0], source_points.shape[-1], source_points.device)

    # fmt: off
    if masks is None:
        def sample_fn():
            return (torch.randint(low=0, high=n_points, size=(B, sample_size),
                                  device=device),
                    None,
                    sample_size * torch.ones((B,), device=device))

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.lt(sq_residuals, pass_thresholds.squeeze(-1))

        def inlier_ratio_fn(n_inliers):
            return n_inliers / n_points
    else:
        def sample_fn():
            return sample_within_nonzero_masks(samples=sample_size, masks=masks.int())

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.logical_and(torch.lt(sq_residuals, pass_thresholds.squeeze(-1)), masks)

        def inlier_ratio_fn(n_inliers):
            return n_inliers / torch.sum(masks, dim=1)
    # fmt: on

    with torch.no_grad():
        status = 1
        best_inlier_ratio = torch.ones((B,), device=source_points.device) * -float("inf")
        best_inliers = [[] for _ in range(B)]
        for i in range(max_iters):
            rand_idx, valid_mask, valid_cnt = sample_fn()
            if torch.any(valid_cnt < sample_size):
                print("RANSAC cannot sample enough valid samples")
                break
            rand_idx = rand_idx.unsqueeze(1).expand((B, 3, sample_size))
            s, R, t = umeyama_batched(
                torch.gather(source_points, 2, rand_idx), torch.gather(target_points, 2, rand_idx)
            )
            pass_thresholds = s * inlier_thres.view(s.shape)
            diff = (target_points - (s * torch.bmm(R, source_points) + t)) ** 2
            sq_residuals = torch.sum(diff, dim=1)

            inliers = inlier_fn(sq_residuals=sq_residuals, pass_thresholds=pass_thresholds**2)

            n_inliers = torch.sum(inliers, dim=1)
            inlier_ratio = inlier_ratio_fn(n_inliers=n_inliers)

            update_flags = inlier_ratio > best_inlier_ratio
            for bid in range(B):
                if update_flags[bid]:
                    inlier_idx = torch.nonzero(inliers[bid, ...], as_tuple=False)
                    best_inliers[bid] = inlier_idx.flatten()
                    best_inlier_ratio[bid] = inlier_ratio[bid]

            stop_flags = (1 - (1 - best_inlier_ratio**sample_size) ** i) > confidence
            if torch.all(stop_flags):
                status = 0
                break

    # inlier run
    s, R, t = (
        torch.ones((B,), device=device),
        torch.eye(3, device=device).repeat((B, 1, 1)),
        torch.zeros((B, 3, 1), device=device),
    )
    for bid in range(B):
        if len(best_inliers[bid]) != 0:
            s[bid, ...], R[bid, ...], t[bid, ...] = umeyama_batched(
                source_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
                target_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
            )
        else:
            s[bid, ...], R[bid, ...], t[bid, ...] = umeyama_batched(
                source_points[bid, :, :sample_size].unsqueeze(0), target_points[bid, :, :sample_size].unsqueeze(0)
            )

    return s, R, t, best_inliers, status


def umeyama(source, target):
    """Run Umeyama algorithm to estimate s, R, t
    target = s*R*source + t
    Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf

    Note: this is a non-batched version
    """
    with torch.cuda.amp.autocast(enabled=False):
        source_centroid = torch.mean(source[:3, :], dim=1)
        target_centroid = torch.mean(target[:3, :], dim=1)

        n_points = source.shape[1]
        centered_source = source[:3, :] - torch.tile(source_centroid.reshape(3, 1), dims=(1, n_points))
        centered_target = target[:3, :] - torch.tile(target_centroid.reshape(3, 1), dims=(1, n_points))
        cov_matrix = centered_target @ centered_source.T / n_points
        if torch.isnan(cov_matrix).any():
            print("nPoints:", n_points)
            print(source.shape)
            print(target.shape)
            raise RuntimeError("There are NANs in the input.")

        ogU, ogD, Vh = torch.linalg.svd(cov_matrix, full_matrices=True)

        # change sign if det(R) = -1
        d = torch.linalg.det(ogU) * torch.linalg.det(Vh)
        D = ogD.clone()
        D[-1] *= d

        U = ogU.clone()
        U[:, -1] *= d

        # rotation
        R = torch.matmul(U, Vh)

        # scale
        varP = torch.var(source[:3, :], dim=1, correction=0).sum()
        s = 1 / varP * torch.sum(D)

        # translation
        t = target_centroid - (s * R @ source_centroid.reshape(3, 1)).flatten()

        # transformation matrix
        T = torch.eye(4).to(source.device)
        T[:3, :3] = s * R
        T[:3, 3] = t

    return s, R, t, T


def umeyama_ransac(source, target, verbose=False):
    """RANSAC wrapped Umeyama to account for outliers.
    Note: this is a non-batched version
    """
    assert source.shape[0] == target.shape[0], "Source and Target must have same dimension."
    assert source.shape[1] == target.shape[1], "Source and Target must have same number of points."
    n_points = source.shape[1]
    source_hom = torch.vstack([source, torch.ones([1, source.shape[1]]).to(source.device)])
    target_hom = torch.vstack([target, torch.ones([1, source.shape[1]]).to(source.device)])

    # strategy from Wild6d
    # auto-threshold selection based on source heuristics
    # assume source is object model or gt nocs map, which is of high quality
    source_centroid = torch.mean(source_hom[:3, :], dim=1)
    centered_source = source_hom[:3, :] - torch.tile(source_centroid.reshape(3, 1), dims=(1, n_points))
    source_diameter = 2 * torch.amax(torch.linalg.norm(centered_source, dim=0))
    inlier_thres = source_diameter * 0.1  # 0.1 of source diameter

    max_iter = 128
    confidence = 0.99

    if verbose:
        print("Inlier threshold: ", inlier_thres)
        print("Max number of iterations: ", max_iter)

    with torch.no_grad():
        best_inlier_ratio = 0
        best_inlier_idx = torch.arange(n_points)
        for i in range(0, max_iter):
            # Pick 5 random (but corresponding) points from source and target
            rand_idx = torch.randint(low=0, high=n_points, size=(5,))
            s, _, _, T = umeyama(source_hom[:, rand_idx], target_hom[:, rand_idx])
            pass_threshold = s * inlier_thres  # propagate inlier threshold to target scale
            diff = target_hom - torch.matmul(T, source_hom)
            residual_vec = torch.linalg.norm(diff[:-1, :], axis=0)
            inlier_idx = torch.nonzero(residual_vec < pass_threshold).flatten()
            n_inliers = inlier_idx.shape[0]
            inlier_ratio = n_inliers / n_points
            # update best hypothesis
            if inlier_ratio > best_inlier_ratio:
                best_inlier_ratio = inlier_ratio
                best_inlier_idx = inlier_idx
            if verbose:
                print("Iteration: ", i)
                print("Inlier ratio: ", best_inlier_ratio)
            # early break
            if (1 - (1 - best_inlier_ratio**5) ** i) > confidence:
                break

        if best_inlier_ratio < 0.1:
            print("[ WARN ] - Something is wrong. Small BestInlierRatio: ", best_inlier_ratio)
            print("[ WARN ] - Using all points as inliers.")
            best_inlier_idx = torch.arange(n_points)

    source_inliers_hom = source_hom[:, best_inlier_idx]
    target_inliers_hom = target_hom[:, best_inlier_idx]
    s, R, t, T = umeyama(source_inliers_hom, target_inliers_hom)

    if verbose:
        print("BestInlierRatio:", best_inlier_ratio)
        print("Rotation:\n", R)
        print("Translation:\n", t)
        print("Scale:", s)

    return s, R, t, T, best_inlier_ratio


def align_nocs_to_depth(
    masks: torch.Tensor,
    nocs: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    instance_ids,
    img_path,
    normalized_nocs=True,
    verbose=False,
):
    """Align NOCS to depth map. Note that this will give the transformation aligning the point cloud
    obtained by recenter & descaling the NOCS. This assumes original object is normalized within [-1, 1].
    And it will give the transformation transforming the original object frame into the depth/camera frame.

    masks: dimension is (B, H, W) or (H, W)
    nocs: dimension is (B, 3, H, W) or (3, H, W)
    depth: dimension is (B, H, W) or (H, W)
    intrinsics: dimension is (B, 3, 3) or (3, 3)
    """
    assert depth.shape[-1] == nocs.shape[-1]
    assert depth.shape[-2] == nocs.shape[-2]
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    assert depth.dim() == nocs.dim() - 1 == masks.dim() == intrinsics.dim()
    if depth.dim() == 2:
        depth, intrinsics, masks = (depth.unsqueeze(0), intrinsics.unsqueeze(0), masks.unsqueeze(0))

    if normalized_nocs:
        nocs_processed = nocs
    else:
        # hack to make it work when the input nocs are not shifted to [0, 1]
        nocs_processed = nocs / 2 + 0.5

    device = nocs_processed.device
    num_instances = len(instance_ids)
    error_messages = ""
    elapses = []
    scales = torch.zeros(num_instances).to(device)
    rotations = torch.zeros((num_instances, 3, 3)).to(device)
    translations = torch.zeros((num_instances, 3)).to(device)
    Ts = torch.zeros((num_instances, 4, 4)).to(device)

    for i in range(num_instances):
        depth_pts, idxs = instance_depth_to_point_cloud_torch(depth[i, ...], intrinsics[i, ...], masks[i, ...])
        nocs_pts = nocs_processed[i, :3, ...]

        # center the NOCS.
        # turn nocs points into original local frame of object normalized within [-1, 1]
        # (assume original object normalized to [-1, 1])
        nocs_pts = (nocs_pts[:, idxs[0], idxs[1]] - 0.5) * 2
        try:
            start = time.time()
            # p^CAM = T^CAM_CAD * p^CAD
            # this gives us T^CAM_CAD
            s, R, t, T, _ = umeyama_ransac(source=nocs_pts, target=depth_pts, verbose=False)
            elapsed = time.time() - start
            if verbose:
                print("RANSAC Umeyama elapsed: ", elapsed)
            elapses.append(elapsed)
        except RuntimeError as e:
            message = "[ Error ] aligning instance {} in {} fails. Message: {}.".format(
                instance_ids[i], img_path, str(e)
            )
            print(message)
            error_messages += message + "\n"
            s = 1.0
            R = torch.eye(3).to(device)
            t = torch.zeros(3).to(device)
            T = torch.eye(4).to(device)

        scales[i] = s
        rotations[i, :, :] = R
        translations[i, :] = t
        Ts[i, ...] = T

    return scales, rotations, translations, Ts, error_messages, elapses


def align_pts_to_depth_no_scale(
    masks: torch.Tensor,
    pts: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    inlier_thres,
    instance_ids,
    img_path,
    verbose=False,
):
    """Align NOCS points to depth map. Note that this will give the transformation aligning the point cloud
    obtained by recenter & descaling the NOCS.
    And it will give the transformation transforming the original object frame into the depth/camera frame.

    masks: dimension is (B, H, W) or (H, W)
    nocs: dimension is (B, 3, H, W) or (3, H, W)
    depth: dimension is (B, H, W) or (H, W)
    intrinsics: dimension is (B, 3, 3) or (3, 3)
    """
    assert depth.shape[-1] == pts.shape[-1]
    assert depth.shape[-2] == pts.shape[-2]
    masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    assert depth.dim() == pts.dim() - 1 == masks.dim() == intrinsics.dim()
    if depth.dim() == 2:
        depth, intrinsics, masks = (depth.unsqueeze(0), intrinsics.unsqueeze(0), masks.unsqueeze(0))

    device = pts.device
    num_instances = len(instance_ids)
    error_messages = ""
    elapses = []
    rotations = torch.zeros((num_instances, 3, 3)).to(device)
    translations = torch.zeros((num_instances, 3)).to(device)
    Ts = torch.zeros((num_instances, 4, 4)).to(device)

    for i in range(num_instances):
        depth_pts, idxs = instance_depth_to_point_cloud_torch(depth[i, ...], intrinsics[i, ...], masks[i, ...])
        nocs_pts = pts[i, :3, ...]
        nocs_pts = nocs_pts[:, idxs[0], idxs[1]]
        T = torch.eye(4).to(device)

        try:
            start = time.time()
            # p^CAM = T^CAM_CAD * p^CAD
            # this gives us T^CAM_CAD
            R, t, _, _ = arun_ransac_batched(
                source_points=nocs_pts.unsqueeze(0), target_points=depth_pts.unsqueeze(0), inlier_thres=inlier_thres
            )
            R, t = R.squeeze(0), t.squeeze(0).flatten()
            elapsed = time.time() - start
            if verbose:
                print("RANSAC Arun elapsed: ", elapsed)
            elapses.append(elapsed)
            T[:3, :3] = R
            T[:3, -1] = t
        except RuntimeError as e:
            message = "[ Error ] aligning instance {} in {} fails. Message: {}.".format(
                instance_ids[i], img_path, str(e)
            )
            print(message)
            error_messages += message + "\n"
            R = torch.eye(3).to(device)
            t = torch.zeros(3).to(device)
        rotations[i, :, :] = R
        translations[i, :] = t
        Ts[i, ...] = T

    return rotations, translations, Ts, error_messages, elapses


@torch.compile(mode="default", disable=not USE_TORCH_COMPILE)
def umeyama_ransac_batched_inlier_thres_target(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    inlier_thres: torch.Tensor,
    masks: torch.Tensor = None,
    confidence=0.99,
    max_iters=100,
    sample_size=5,
):
    """Batched RANSAC version of Umeyama.
    Terminate if all instances in the batch satisfy the convergence cost criteria.
    Inlier threshold on target scale.

    Parameters
    ----------
    source_points
    target_points
    masks: (B, pc_size)
    inlier_thres
    confidence
    max_iters
    sample_size
    """
    B, n_points, device = (source_points.shape[0], source_points.shape[-1], source_points.device)

    # fmt: off
    if masks is None:
        def sample_fn():
            return (torch.randint(low=0, high=n_points, size=(B, sample_size),
                                  device=device),
                    None,
                    sample_size * torch.ones((B,), device=device))

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.lt(sq_residuals, pass_thresholds.squeeze(-1))

        def inlier_ratio_fn(n_inliers):
            return n_inliers / n_points
    else:
        def sample_fn():
            return sample_within_nonzero_masks(samples=sample_size, masks=masks.int())

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.logical_and(torch.lt(sq_residuals, pass_thresholds.squeeze(-1)), masks)

        def inlier_ratio_fn(n_inliers):
            return n_inliers / torch.sum(masks, dim=1)
    # fmt: on

    with torch.no_grad():
        status = 1
        best_inlier_ratio = torch.ones((B,), device=source_points.device) * -float("inf")
        best_inliers = [[] for _ in range(B)]
        for i in range(max_iters):
            rand_idx, valid_mask, valid_cnt = sample_fn()
            if torch.any(valid_cnt < sample_size):
                print("RANSAC cannot sample enough valid samples")
                break
            rand_idx = rand_idx.unsqueeze(1).expand((B, 3, sample_size))
            s, R, t = umeyama_batched(
                torch.gather(source_points, 2, rand_idx), torch.gather(target_points, 2, rand_idx)
            )
            pass_thresholds = inlier_thres.view(s.shape)
            diff = (target_points - (s * torch.bmm(R, source_points) + t)) ** 2
            sq_residuals = torch.sum(diff, dim=1)

            inliers = inlier_fn(sq_residuals=sq_residuals, pass_thresholds=pass_thresholds**2)

            n_inliers = torch.sum(inliers, dim=1)
            inlier_ratio = inlier_ratio_fn(n_inliers=n_inliers)

            update_flags = inlier_ratio > best_inlier_ratio
            for bid in range(B):
                if update_flags[bid]:
                    inlier_idx = torch.nonzero(inliers[bid, ...], as_tuple=False)
                    best_inliers[bid] = inlier_idx.flatten()
                    best_inlier_ratio[bid] = inlier_ratio[bid]

            stop_flags = (1 - (1 - best_inlier_ratio**sample_size) ** i) > confidence
            if torch.all(stop_flags):
                status = 0
                break

    # inlier run
    s, R, t = (
        torch.ones((B,), device=device),
        torch.eye(3, device=device).repeat((B, 1, 1)),
        torch.zeros((B, 3, 1), device=device),
    )
    for bid in range(B):
        if len(best_inliers[bid]) != 0:
            s[bid, ...], R[bid, ...], t[bid, ...] = umeyama_batched(
                source_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
                target_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
            )
        else:
            s[bid, ...], R[bid, ...], t[bid, ...] = umeyama_batched(
                source_points[bid, :, :sample_size].unsqueeze(0), target_points[bid, :, :sample_size].unsqueeze(0)
            )

    return s, R, t, best_inliers, status


def weighted_arun_batched(source_points, target_points, weights):
    """Run Arun's algorithm to estimate R, t (weighted)
    target = R*source + t

    Note: this is a batched version
    """
    with torch.cuda.amp.autocast(enabled=False):
        assert source_points.ndim == target_points.ndim == 3

        N = source_points.shape[2]
        # centroids
        source_points_ave = torch.sum(source_points * weights.unsqueeze(1), dim=2) / weights.sum(dim=1, keepdim=True)
        target_points_ave = torch.sum(target_points * weights.unsqueeze(1), dim=2) / weights.sum(dim=1, keepdim=True)

        # getting the rotation
        source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
        target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

        # get rotation
        mat = (target_points_centered * weights.unsqueeze(1)) @ source_points_centered.transpose(-1, -2)  # (B, 3, 3)
        R, _, S, _, d = project_SO3(mat / weights.sum(dim=1, keepdim=True).unsqueeze(-1))
        # S.clone() is necessary for avoiding backprop error due to in-place operations
        D = S.clone()
        D[:, -1] *= d

        # getting the translation
        t = target_points_ave.unsqueeze(-1) - R @ source_points_ave.unsqueeze(-1)

    return R, t


@torch.compile(mode="default", disable=not USE_TORCH_COMPILE)
def arun_batched(source_points, target_points):
    """Run Arun's algorithm to estimate R, t (unweighted)
    target = R*source + t

    Note: this is a batched version
    """
    og_dtype = source_points.dtype
    with torch.cuda.amp.autocast(enabled=False):
        assert source_points.ndim == target_points.ndim == 3

        N = source_points.shape[2]
        # centroids
        source_points_ave = torch.sum(source_points, dim=2) / N
        target_points_ave = torch.sum(target_points, dim=2) / N

        # getting the rotation
        source_points_centered = source_points - source_points_ave.unsqueeze(-1)  # (B, 3, N)
        target_points_centered = target_points - target_points_ave.unsqueeze(-1)  # (B, 3, N)

        # get rotation
        mat = target_points_centered @ source_points_centered.transpose(-1, -2) / N  # (B, 3, 3)
        R, _, S, _, d = project_SO3(mat)
        # S.clone() is necessary for avoiding backprop error due to in-place operations
        D = S.clone()
        D[:, -1] *= d

        # getting the translation
        t = target_points_ave.unsqueeze(-1) - R @ source_points_ave.unsqueeze(-1)

    return R.to(og_dtype), t.to(og_dtype)


@torch.compile(mode="default", disable=not USE_TORCH_COMPILE)
def arun_ransac_batched(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    inlier_thres: torch.Tensor,
    masks: torch.Tensor = None,
    confidence=0.99,
    max_iters=100,
    sample_size=5,
):
    """Batched RANSAC version of Umeyama.
    Terminate if all instances in the batch satisfy the convergence cost criteria.
    Inlier threshold on target scale.

    Parameters
    ----------
    source_points
    target_points
    masks: (B, pc_size)
    inlier_thres
    confidence
    max_iters
    sample_size
    """
    B, n_points, device = (source_points.shape[0], source_points.shape[-1], source_points.device)

    # fmt: off
    if masks is None:
        def sample_fn():
            return (torch.randint(low=0, high=n_points, size=(B, sample_size),
                                  device=device),
                    None,
                    sample_size * torch.ones((B,), device=device))

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.lt(sq_residuals, pass_thresholds.squeeze(-1))

        def inlier_ratio_fn(n_inliers):
            return n_inliers / n_points
    else:
        def sample_fn():
            return sample_within_nonzero_masks(samples=sample_size, masks=masks.int())

        def inlier_fn(sq_residuals, pass_thresholds):
            return torch.logical_and(torch.lt(sq_residuals, pass_thresholds.squeeze(-1)), masks)

        def inlier_ratio_fn(n_inliers):
            return n_inliers / torch.sum(masks, dim=1)
    # fmt: on

    with torch.no_grad():
        status = 1
        best_inlier_ratio = torch.ones((B,), device=source_points.device) * -float("inf")
        best_inliers = [[] for _ in range(B)]
        for i in range(max_iters):
            rand_idx, valid_mask, valid_cnt = sample_fn()
            if torch.any(valid_cnt < sample_size):
                print("RANSAC cannot sample enough valid samples")
                break
            rand_idx = rand_idx.unsqueeze(1).expand((B, 3, sample_size))
            R, t = arun_batched(torch.gather(source_points, 2, rand_idx), torch.gather(target_points, 2, rand_idx))
            pass_thresholds = inlier_thres.view((R.shape[0], 1, 1))
            diff = (target_points - (torch.bmm(R, source_points) + t)) ** 2
            sq_residuals = torch.sum(diff, dim=1)

            inliers = inlier_fn(sq_residuals=sq_residuals, pass_thresholds=pass_thresholds**2)

            # TODO: Investigate zero inliers
            n_inliers = torch.sum(inliers, dim=1)
            inlier_ratio = inlier_ratio_fn(n_inliers=n_inliers)

            update_flags = inlier_ratio > best_inlier_ratio
            for bid in range(B):
                if update_flags[bid]:
                    inlier_idx = torch.nonzero(inliers[bid, ...], as_tuple=False)
                    best_inliers[bid] = inlier_idx.flatten()
                    best_inlier_ratio[bid] = inlier_ratio[bid]

            stop_flags = (1 - (1 - best_inlier_ratio**sample_size) ** i) > confidence
            if torch.all(stop_flags):
                status = 0
                break

    # inlier run
    R, t = (
        torch.eye(3, device=device).repeat((B, 1, 1)),
        torch.zeros((B, 3, 1), device=device),
    )
    for bid in range(B):
        if len(best_inliers[bid]) != 0:
            R[bid, ...], t[bid, ...] = arun_batched(
                source_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
                target_points[bid, ...][:, best_inliers[bid]].unsqueeze(0),
            )
        else:
            R[bid, ...], t[bid, ...] = arun_batched(
                source_points[bid, :, :sample_size].unsqueeze(0), target_points[bid, :, :sample_size].unsqueeze(0)
            )

    return R, t, best_inliers, status


def teaser_with_scale_batched(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    noise_bounds: torch.Tensor,
    masks: torch.Tensor,
):
    """Solve robust registration problem with TEASER++
    Assume the following model:
    tgt = s @ R @ src + t + noise
    where noise is bounded by noise_bounds.

    After TEASER++ gives us inliers, we solve the problem again with inliers only to
    recover differentiability.

    Parameters
    ----------
    source_points: (B, 3, N)
    target_points: (B, 3, N)
    noise_bounds: (B, 1) - bounds on the noise in the target_points frame
    masks: (B, N) - masks to select points for each instance
    """
    with torch.no_grad():
        # prepare inputs
        B = source_points.shape[0]
        src_inputs = [source_points[i, :, masks[i, :]].detach().double().numpy(force=True) for i in range(B)]
        dst_inputs = [target_points[i, :, masks[i, :]].detach().double().numpy(force=True) for i in range(B)]
        nbs = noise_bounds.detach().flatten().double().numpy(force=True).tolist()

        # run teaser
        sols = teaserpp_python.batch_gnc_solve(src=src_inputs, dst=dst_inputs, noise_bound=nbs, estimate_scale=True)

        # gather solutions
        inlier_masks = torch.tensor(np.array([sols[i][1] for i in range(B)]), device=source_points.device)

    # run svd again
    s, R, t = weighted_umeyama_batched(
        source_points=source_points.float(), target_points=target_points.float(), weights=inlier_masks
    )

    return s, R, t, inlier_masks


def teaser_batched(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    noise_bounds: torch.Tensor,
    masks: torch.Tensor,
):
    """Solve robust registration problem with TEASER++
    Assume the following model:
    tgt = R @ src + t + noise
    where noise is bounded by noise_bounds.

    After TEASER++ gives us inliers, we solve the problem again with inliers only to
    recover differentiability.

    Parameters
    ----------
    source_points: (B, 3, N)
    target_points: (B, 3, N)
    noise_bounds: (B, 1) - bounds on the noise in the target_points frame
    masks: (B, N) - masks to select points for each instance
    """
    with torch.no_grad():
        # prepare inputs
        B = masks.shape[0]
        src_inputs = [source_points[i, :, masks[i, :]].double().numpy(force=True) for i in range(B)]
        dst_inputs = [target_points[i, :, masks[i, :]].double().numpy(force=True) for i in range(B)]
        nbs = noise_bounds.flatten().double().numpy(force=True).tolist()

        # run teaser
        sols = teaserpp_python.batch_gnc_solve(src=src_inputs, dst=dst_inputs, noise_bound=nbs, estimate_scale=True)

        # gather solutions
        inlier_masks = torch.tensor(np.array([sols[i][1] for i in range(B)]), device=source_points.device)

    # run svd again
    R, t = weighted_arun_batched(
        source_points=source_points.float(), target_points=target_points.float(), weights=inlier_masks
    )

    return R, t, inlier_masks
