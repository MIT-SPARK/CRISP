"""ICP distance functions/metrics (point-to-point, point-to-plane, Symmetric ICP)"""

import torch


def get_distance_function(
    dist_type, point_weight=0, plane_weight=0, symmetric_weight=0
):
    """choose distance function based on given dist_type"""
    if dist_type == "point":
        return point_to_point_dist
    if dist_type == "plane":
        return point_to_plane_dist
    if dist_type == "symmetric":
        return symmetric_dist
    if dist_type == "mixed":
        return lambda *args, **kwargs: mixed_dist(
            *args,
            **kwargs,
            point_weight=point_weight,
            plane_weight=plane_weight,
            symmetric_weight=symmetric_weight,
        )
    raise ValueError(f"dist_type {dist_type} not recognized.")


def point_to_point_dist(sources, targets, weights=None, **_):
    """
    calculate point-to-point residuals between two sets of corresponding points
    :param sources: source points, Kx3 tensor of x-y-z coordinates
    :param targets: target points, Kx3 tensor of x-y-z coordinates
    :param weights: optional correspondence weights, shape Kx1
    :return: point-to-point residuals for all correspondences, shape 3K
    """
    assert sources.shape == targets.shape
    dists = sources - targets  # Kx3
    if weights is not None:
        dists = weights * dists  # Kx3
    return dists.reshape(-1)  # 3K


def point_to_plane_dist(sources, targets, target_normals=None, weights=None, **_):
    """
    calculate point-to-plane residuals between two sets of corresponding points
    :param sources: source points, Kx3 tensor of x-y-z coordinates
    :param targets: target points, Kx3 tensor of x-y-z coordinates
    :param target_normals: normals of each target point, shape Kx3
    :param weights: optional correspondence weights, shape Kx1
    :return: point-to-plane residuals for all correspondences, shape K
    """
    assert sources.shape == targets.shape
    assert targets.shape == target_normals.shape
    dists = torch.sum((sources - targets) * target_normals, dim=-1, keepdim=True)  # Kx1
    if weights is not None:
        dists = weights * dists  # Kx1
    return dists.reshape(-1)  # K


def symmetric_dist(
    sources, targets, source_normals=None, target_normals=None, weights=None, **_
):
    """
    calculate symmetric ICP residuals between two sets of corresponding points
    :param sources: source points, Kx3 tensor of x-y-z coordinates
    :param targets: target points, Kx3 tensor of x-y-z coordinates
    :param source_normals: normals of each source point, shape Kx3
    :param target_normals: normals of each target point, shape Kx3
    :param weights: optional correspondence weights, shape Kx1
    :return: symmetric ICP residuals for all correspondences, shape K
    """
    assert sources.shape == targets.shape
    assert sources.shape == source_normals.shape
    assert targets.shape == target_normals.shape
    sym_normal = source_normals + target_normals  # Kx3
    dists = torch.sum((sources - targets) * sym_normal, dim=-1, keepdim=True)  # Kx1
    if weights is not None:
        dists = weights * dists  # Kx1
    return dists.reshape(-1)  # K


def mixed_dist(
    sources,
    targets,
    source_normals=None,
    target_normals=None,
    weights=None,
    point_weight=0,
    plane_weight=0,
    symmetric_weight=0,
    **_,
):
    """
    combination of point-to-point, point-to-plane, and symmetric ICP distances
    :param sources: source points, Kx3 tensor of x-y-z coordinates
    :param targets: target points, Kx3 tensor of x-y-z coordinates
    :param source_normals: normals of each source point, shape Kx3
    :param target_normals: normals of each target point, shape Kx3
    :param weights: optional correspondence weights, shape Kx1
    :param point_weight: weight of point-to-point constraints
    :param plane_weight: weight of point-to-plane constraints
    :param symmetric_weight: weight of symmetric ICP constraints
    :return: combined residuals, shape LK for L in {1, 2, 3, 4, 5}
    """
    assert point_weight > 0 or plane_weight > 0 or symmetric_weight > 0
    dists = torch.tensor([], device=sources.device)
    if point_weight > 0:
        point_dist = point_to_point_dist(
            sources=sources,
            targets=targets,
            weights=weights,
        )
        dists = torch.cat([dists, point_dist * point_weight], 0)
    if plane_weight > 0:
        plane_dist = point_to_plane_dist(
            sources=sources,
            targets=targets,
            target_normals=target_normals,
            weights=weights,
        )
        dists = torch.cat([dists, plane_dist * plane_weight], 0)
    if symmetric_weight > 0:
        sym_dist = symmetric_dist(
            sources=sources,
            targets=targets,
            source_normals=source_normals,
            target_normals=target_normals,
            weights=weights,
        )
        dists = torch.cat([dists, sym_dist * symmetric_weight], 0)
    return dists / max(point_weight, plane_weight, symmetric_weight)
