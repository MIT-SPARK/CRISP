"""util function for normal calculation and plotting"""

import torch


def get_depthmap_normals(depthmap):
    """
    Compute normals for a depthmap efficiently with torch
    :param depthmap: depthmap
    :return: depthmap normals
    """

    def _is_zero(matrix_):
        return matrix_.sum(dim=2) == 0

    def _not_zero(matrix_):
        return matrix_.sum(dim=2) != 0

    def replace_zero(matrix1, matrix2):
        zero_inds = _is_zero(matrix1)
        matrix1[zero_inds] = matrix2[zero_inds].clone()

    center = depthmap[1:-1, 1:-1].clone()
    right = depthmap[2:, 1:-1].clone()
    left = depthmap[:-2, 1:-1].clone()
    bottom = depthmap[1:-1, 2:].clone()
    top = depthmap[1:-1, :-2].clone()
    for matrix in [right, left, bottom, top]:
        replace_zero(matrix, center)
    dzdy = (right - left) / 2
    dzdx = (bottom - top) / 2
    normals = torch.zeros_like(depthmap)
    normals_center = normals[1:-1, 1:-1]
    center_non_zero = _not_zero(center)
    normals_center[center_non_zero] = torch.cross(dzdy, dzdx)[center_non_zero]
    norm = torch.sqrt(torch.sum(normals ** 2, dim=2, keepdim=True))
    normal_non_zero = _not_zero(normals)
    normals[normal_non_zero] = normals[normal_non_zero] / norm[normal_non_zero]
    return normals


def get_mesh_normals(vertices, faces):
    """
    Compute vertex normals for a mesh
    :param vertices: mesh vertices
    :param faces: mesh faces
    :return: vertex normals
    """

    triangles = vertices[faces]
    normal = torch.zeros(vertices.shape, dtype=vertices.dtype, device=vertices.device)

    def face_normal(ind):
        """calculate the face normal around a given index"""
        j, k = (ind + 1) % 3, (ind + 2) % 3  # the two other indices of the triangle
        normal_ = torch.cross(triangles[::, j] - triangles[::, ind], triangles[::, k] - triangles[::, ind])
        return torch.nn.functional.normalize(normal_, dim=1)

    for i in range(3):
        normal = normal.index_add_(0, faces[:, i], face_normal(i))
    return torch.nn.functional.normalize(normal, dim=1)
