"""util functions for backprojecting depth maps"""

import torch

from .geometry_utils import screen_to_global_projection


def reproject_depthmap_as_pointcloud(depthmap, cam_intr, cam_pose):
    """
    project rendered depth back to global space as pointcloud
    :param depthmap: depthmap, shape HxW
    :param cam_intr: 4x4 camera intrinsics
    :param cam_pose: 4x4 camera pose
    :return: N non-zero-depth points (x,y,z) in global space, shape Nx3
    """
    y, x = torch.nonzero(depthmap, as_tuple=True)
    z = depthmap[y, x]
    points = torch.stack([x, y, z], dim=1)
    return screen_to_global_projection(points, cam_intr, cam_pose)


def reproject_depthmap_as_matrix(depthmap, cam_intr, cam_pose):
    """
    project rendered depth back to global space and keep matrix shape
    :param depthmap: depthmap, shape HxW
    :param cam_intr: 4x4 camera intrinsics
    :param cam_pose: 4x4 camera pose
    :return: depthmap containing global space coordinates (or 0), shape HxWx3
    """
    cloud = reproject_depthmap_as_pointcloud(depthmap, cam_intr, cam_pose)
    reprojected_matrix = torch.zeros(
        [*depthmap.shape, 3], device=depthmap.device, dtype=depthmap.dtype
    )
    y, x = torch.nonzero(depthmap, as_tuple=True)
    reprojected_matrix[y, x] = cloud
    return reprojected_matrix


def depthmap_matrix_to_pointcloud(matrix, normals=None):
    """
    Get all non-zero entries of a pointcloud or depthmap
    This is a utility function that converts
    """
    matrix = matrix.reshape([-1, 3])
    indices = torch.unique(torch.nonzero(matrix)[:, 0])
    pointcloud = matrix[indices]
    if normals is not None:
        normals = normals.reshape([-1, 3])[indices]
    return pointcloud, normals
