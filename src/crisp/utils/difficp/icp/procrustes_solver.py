"""Solvers for procrustes problem (finding optimal global rotation/translation)"""

import math
import torch

from crisp.utils.difficp.utils.geometry_utils import pose_to_matrix
from .linear_solver import LinearSolverLU


class IllconditionedSVDException(Exception):
    """Raise when calculating gradients through SVD with ill-conditioned matrix"""


def get_condition_number(matrix):
    """Calculate condition number of a matrix (to check if its ill-conditioned)"""
    values = torch.linalg.eigvals(matrix)
    real_values = values.real
    max_eig_value = torch.max(torch.abs(real_values))
    min_eig_value = torch.min(torch.abs(real_values))
    cond_num = (max_eig_value / min_eig_value).item()
    return cond_num


def check_conditioning(matrix, threshold=1e9):
    """
    Check condition of a matrix and raise an error if its illconditioned
    :raises: IllconditionedSVDException
    """
    cond_num = get_condition_number(matrix)
    illconditioned = not math.isfinite(cond_num) or cond_num > threshold
    if illconditioned and matrix.requires_grad:
        raise IllconditionedSVDException(
            f"SVD with condition number {cond_num} might not be differentiable."
        )


def weighted_centroid(points, weights=None):
    if weights is not None:
        weighted_sum = torch.sum(weights * points, dim=-2)
        sum_weight = torch.sum(weights, dim=-2).squeeze()
        return weighted_sum / sum_weight
    return torch.mean(points, dim=-2)


def solve_procrustes_svd(pose, sources, targets, weights=None, **kwargs):
    """
    solve point-to-point procrustes using the Kabsch algorithm
    :param pose: initial pose (alpha, beta, gamma, tx, ty, tz), shape 6
    :param sources: source points, shape Kx3
    :param targets: target points, shape Kx3
    :param weights: correspondence weights, shape Kx1
    :param kwargs: unused keyword args
    :return: best transformation between source and target, shape 6
    """
    source_center = weighted_centroid(sources, weights)
    target_center = weighted_centroid(targets, weights)
    sources = sources - source_center
    targets = targets - target_center
    if weights is not None:
        svd_w = torch.matmul(torch.diag(weights.view(-1)), targets)
        svd_w = torch.matmul(sources.t(), svd_w)
    else:
        svd_w = torch.matmul(sources.t(), targets)  # 3x3
    check_conditioning(svd_w)
    svd_u, _, svd_v = torch.svd(svd_w)  # 3x3, 1x3, 3x3
    rotation = torch.matmul(svd_v, svd_u.t())  # 3x3
    if torch.det(rotation) < 0:  # reflection check
        reflection_fix = torch.ones_like(svd_v, dtype=svd_v.dtype, device=svd_v.device)
        reflection_fix[:, -1] = -1
        svd_v = svd_v * reflection_fix
        rotation = torch.matmul(svd_v, svd_u.t())
    translation = target_center.t() - torch.matmul(rotation, source_center.t())  # 1x3
    pose_change = torch.zeros_like(pose)  # 4x4
    pose_change[:3, :3] = rotation
    pose_change[:3, 3] = translation.view(-1)
    pose_change[3, 3] = 1
    return torch.matmul(pose_change, pose)  # 4x4


class LinearizedProcrustesSolver:
    def __init__(self, point_weight=0, plane_weight=0, symmetric_weight=0):
        assert point_weight > 0 or plane_weight > 0 or symmetric_weight > 0
        self.point_weight = point_weight
        self.plane_weight = plane_weight
        self.symmetric_weight = symmetric_weight

    def __call__(
        self,
        pose,
        sources,
        targets,
        source_normals=None,
        target_normals=None,
        weights=None,
        **_,
    ):
        """
        solve point-to-plane procrustes using linear least-squares
        assumes small angles (linear approximation method)
        https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
        :param pose: initial pose (alpha, beta, gamma, tx, ty, tz), shape 6
        :param sources: source points, shape Kx3
        :param targets: target points, shape Kx3
        :param source_normals: normals of each source point, shape Kx3
        :param target_normals: normals of each target point, shape Kx3
        :param weights: correspondence weights, shape Kx1
        :return: best transformation between source and target, shape 6
        """
        a, b = [torch.tensor([], device=sources.device) for _ in range(2)]

        if weights is None:
            weights = torch.ones_like(sources[:, :1])  # Kx1
        else:
            weights = weights.view(-1, 1)  # Kx1

        def add_point_constraints(constraint_weight):
            a_point = torch.zeros(
                [sources.shape[0], 18], dtype=sources.dtype, device=sources.device
            )  # K x 18
            a_point[:, 1] = sources[:, 2]
            a_point[:, 2] = -sources[:, 1]
            a_point[:, 3] = 1
            a_point[:, 6] = -sources[:, 2]
            a_point[:, 8] = sources[:, 0]
            a_point[:, 10] = 1
            a_point[:, 12] = sources[:, 1]
            a_point[:, 13] = -sources[:, 0]
            a_point[:, 17] = 1
            b_point = targets - sources  # K x 3
            a_point = a_point * weights * constraint_weight
            b_point = b_point * weights * constraint_weight
            a_point = a_point.reshape(-1, 6)  # 3K x 6
            b_point = b_point.reshape(-1, 1)  # 3K x 1
            return torch.cat([a, a_point], 0), torch.cat([b, b_point], 0)

        def add_plane_constraints(normals, constraint_weight):
            a_ = torch.cross(sources, normals, 1)
            a_ = torch.cat([a_, normals], -1)  # K x 6
            b_ = torch.sum(normals * targets - normals * sources, dim=1, keepdim=True)
            a_ = a_ * weights * constraint_weight
            b_ = b_ * weights * constraint_weight
            return torch.cat([a, a_], 0), torch.cat([b, b_], 0)

        if self.point_weight > 0:
            a, b = add_point_constraints(self.point_weight)

        if self.plane_weight > 0:
            a, b = add_plane_constraints(target_normals, self.plane_weight)

        if self.symmetric_weight > 0:
            symmetric_normals = source_normals + target_normals
            a, b = add_plane_constraints(symmetric_normals, self.symmetric_weight)

        ata = torch.matmul(a.t(), a)
        atb = torch.matmul(a.t(), b)
        pose_change = LinearSolverLU.apply(ata, atb).squeeze()
        pose_change = pose_to_matrix(pose_change)
        return torch.matmul(pose_change, pose)
