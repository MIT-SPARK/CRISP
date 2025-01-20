"""
Differentiable Pytorch ICP implementation
"""

import torch
from abc import ABC, abstractmethod

from crisp.utils.difficp.utils.geometry_utils import transform_points_by_matrix, pose_to_matrix
from .correspondence_function import get_correspondence_function
from .distance_function import get_distance_function
from .optimizer import LMOptimizer6DoF, LMOptimizerSMPL
from .procrustes_solver import solve_procrustes_svd, LinearizedProcrustesSolver


def combine_weights(weights):
    """
    combine a list of weight tensors by multiplication if not None
    :param weights: list of tensors, all tensors are of same shape or are None
    :return: multiplication of all non-None weight tensors, or None if all are None
    """
    total_weight = None
    for weight in weights:
        if weight is not None:
            if total_weight is None:
                total_weight = weight
            else:
                assert weight.shape == total_weight.shape
                total_weight = total_weight * weight
    return total_weight


class ICP(ABC):
    """Abstract Base Class of Differentiable Pytorch ICP"""

    def __init__(
        self,
        iters_max=100,
        mse_threshold=1e-5,
        corr_threshold=10,
        corr_type="nn",
        dist_type="point",
        weight_dist=False,
        weight_normal=False,
        rejection_dist=None,
        rejection_normal=None,
        differentiable=True,
        matching_temperature=1e-3,
        rejection_temperature=1e-5,
        w_point=0,
        w_plane=0,
        w_symmetric=0,
        verbose=False,
    ):
        """
        Initialize the ICP
        :param iters_max: maximum number of ICP iterations
        :param mse_threshold: MSE difference threshold to check convergence
        :param corr_threshold: correspondence difference threshold to check convergence
        :param corr_type: type of correspondence finding function
        :param dist_type: type of distance function
        :param weight_dist: whether to weight correspondences based on distances
        :param weight_normal: whether to weight correspondences based on normals
        :param rejection_dist: threshold to reject corrs with high distance
        :param rejection_normal: threshold to reject corrs with different normals
        :param differentiable: if False, set temperature parameters to 0 to use standard
            non-differentiable ICP (which is faster at inference time)
        :param matching_temperature: temperature parameter of correspondence finding
        :param rejection_temperature: temperature parameter of correspondence rejection
        :param w_point: weight of point-to-point constraints for dist_type="mixed"
        :param w_plane: weight of point-to-plane constraints for dist_type="mixed"
        :param w_symmetric: weight of symmetric ICP constraints for dist_type="mixed"
        :param verbose: print additional information if set to true
        """
        self.iters_max = iters_max
        self.mse_threshold = mse_threshold
        self.corr_threshold = corr_threshold
        self.corr_type = corr_type
        self.dist_type = dist_type
        self.weight_dist = weight_dist
        self.weight_normal = weight_normal
        self.rejection_dist = rejection_dist
        self.rejection_normal = rejection_normal
        self.verbose = verbose
        self.num_params = self.define_num_params()
        self.corr_fn = get_correspondence_function(
            self.corr_type,
            matching_temperature=matching_temperature if differentiable else 0,
            weight_dist=weight_dist,
            weight_normal=weight_normal,
            rejection_dist=rejection_dist,
            rejection_normal=rejection_normal,
            rejection_temperature=rejection_temperature if differentiable else 0,
        )

        def define_constraint_weight(constraint_type, constraint_value):
            if dist_type == "mixed":
                return constraint_value
            if dist_type == constraint_type:
                return 1
            return 0

        self.w_point = define_constraint_weight("point", w_point)
        self.w_plane = define_constraint_weight("plane", w_plane)
        self.w_symmetric = define_constraint_weight("symmetric", w_symmetric)
        self.dist_fn = get_distance_function(
            dist_type=dist_type,
            point_weight=self.w_point,
            plane_weight=self.w_plane,
            symmetric_weight=self.w_symmetric,
        )
        self.transform_fn = self.define_transform_fn()
        self.solver = self.define_solver()
        self.last_mse, self.last_num_corrs = None, None  # set in __call__()

    @abstractmethod
    def define_num_params(self):
        """define how many parameters to optimize for"""

    @abstractmethod
    def define_transform_fn(self):
        """define function to transform pointcloud based on given params"""

    @abstractmethod
    def define_solver(self):
        """define solver function, e.g. LMOptimizer"""

    def init_param_handling(self, init_params, dtype, device):
        """handle a given parameter initialization"""
        if init_params is None:
            return torch.zeros(self.num_params, dtype=dtype, device=device)
        size = init_params.size()
        if len(size) == 1 and size[0] == self.num_params:
            return init_params
        raise ValueError(
            f"initial params should have shape {self.num_params}, not {size}"
        )

    def _check_convergence(self, mse, num_corrs, weights):
        no_corrs = num_corrs == 0
        no_weights = weights is not None and torch.max(weights) == 0
        corrs_converged = abs(num_corrs - self.last_num_corrs) <= self.corr_threshold
        mse_converged = abs(mse - self.last_mse) <= self.mse_threshold
        self.last_mse = mse
        self.last_num_corrs = num_corrs
        converged = no_corrs or no_weights or (corrs_converged and mse_converged)
        return converged

    def initialize(self, sources, targets, source_normals, target_normals, init_pose):
        # assert sources and targets in N x 3 format, otherwise convert
        if source_normals is not None:
            assert sources.shape == source_normals.shape
        if target_normals is not None:
            assert targets.shape == target_normals.shape
        if len(sources.shape) == 3:
            sources, source_normals = depth_map_matrix_to_pointcloud(sources, source_normals)
        if len(targets.shape) == 3 and self.corr_type == "nn":
            targets, target_normals = depth_map_matrix_to_pointcloud(targets, target_normals)
            assert len(targets.shape) == 2
        assert len(sources.shape) == 2
        assert len(targets.shape) in (2, 3)
        assert len(sources) > 0 and len(targets) > 0
        if sources.shape[-1] != 3:
            sources = sources.transpose(1, 0).contiguous()
            assert sources.shape[-1] == 3
        if len(targets.shape) == 2 and targets.shape[-1] != 3:
            targets = targets.transpose(1, 0).contiguous()
            assert targets.shape[-1] == 3
        # handle init params and return tensors
        init_pose = self.init_param_handling(
            init_pose,
            dtype=targets.dtype,
            device=targets.device,
        )
        mse, self.last_mse, self.last_num_corrs = [0] * 3
        return sources, targets, source_normals, target_normals, init_pose, mse

    def __call__(
        self,
        sources,
        targets,
        init_pose=None,
        source_normals=None,
        target_normals=None,
        cam_intr=None,
        cam_pose=None,
        **unused_kwargs,
    ):
        """
        Run ICP to find 6DOF translation between given set of source and target points
        :param sources: source points, shape Nx3
        :param targets: target points, shape Mx3
        :param init_pose: initial transformation (shape domain specific)
        :param source_normals: normals of each source point, shape Nx3
        :param target_normals: normals of each target point, shape Mx3
        :param source_weights: predefined weights for each source point, shape Nx1
        :param target_weights: predefined weights for each target point, shape Mx1
        :param cam_intr: camera intrinsics matrix, shape 4x4
        :param cam_pose: camera pose (extrinsics) matrix, shape 4x4
        :return: best transformation between source and target (shape domain specific)
        """
        sources, targets, source_normals, target_normals, pose, mse = self.initialize(
            sources, targets, source_normals, target_normals, init_pose
        )
        for i in range(self.iters_max):
            src = self.transform_fn(sources, pose)
            src, tgt, src_normals, tgt_normals, _, _, weights = self.corr_fn(
                sources=src,
                targets=targets,
                source_normals=source_normals,
                target_normals=target_normals,
                cam_intr=cam_intr,
                cam_pose=cam_pose,
            )
            residuals = self.dist_fn(
                src,
                tgt,
                source_normals=src_normals,
                target_normals=tgt_normals,
                weights=weights,
            )
            mse = torch.mean(residuals ** 2)
            num_corrs = len(src)
            if self.verbose:
                print(i, mse, num_corrs, pose)
            if self._check_convergence(mse.detach(), num_corrs, weights):
                return pose, i, mse
            pose = self.solver(
                pose,
                src,
                tgt,
                source_normals=src_normals,
                target_normals=tgt_normals,
                weights=weights,
            )
        if self.verbose:
            print(self.iters_max, self.last_mse, pose)
        return pose, self.iters_max, mse


class ICP6DoF(ICP):
    """
    Rigid 6DoF ICP
    optimizes for a 3x3 rotation matrix and a 3x1 translation
    """

    def __init__(self, *args, solver_type="svd", **kwargs):
        self.solver_type = solver_type
        super().__init__(*args, **kwargs)

    def define_num_params(self):
        return 6

    def init_param_handling(self, init_params, dtype, device):
        """overwrite init param handling to also support euler angles inputs"""
        if init_params is None:
            return torch.eye(4, dtype=dtype, device=device)
        size = init_params.size()
        if len(size) == 2 and size[0] == size[1] == 4:
            return init_params
        return pose_to_matrix(super().init_param_handling(init_params, dtype, device))

    def define_transform_fn(self):
        return transform_points_by_matrix

    def define_solver(self):
        """choose solver function based on self.solver_type"""
        if self.solver_type == "svd" and self.dist_type != "point":
            raise ValueError(
                "SVD procrustes solver can only be used with point-to-point distance."
            )
        return {
            "lm": LMOptimizer6DoF(
                jacobian_type="analytic",
                verbose=self.verbose,
                dist_type=self.dist_type,
                w_point=self.w_point,
                w_plane=self.w_plane,
                w_symmetric=self.w_symmetric,
            ),
            "svd": solve_procrustes_svd,
            "linear": LinearizedProcrustesSolver(
                point_weight=self.w_point,
                plane_weight=self.w_plane,
                symmetric_weight=self.w_symmetric,
            ),
        }[self.solver_type]


class ICPSMPL(ICP):
    """
    Non-rigid ICP for SMPL body pose/shape parameters
    Optimizes for 23*3 pose parameters and 10 shape parameters
    """

    def __init__(self, smpl_model, *args, global_pose=None, **kwargs):
        self.smpl_model = smpl_model
        self.global_pose = global_pose
        super().__init__(*args, **kwargs)

    def define_num_params(self):
        return 79

    def define_transform_fn(self):
        def build_smpl(points, params):
            """build a model with given smpl parameters"""
            pose = params[:69].reshape(1, -1)
            shape = params[69:].reshape(1, -1)
            vertices = self.smpl_model(betas=shape, body_pose=pose).vertices.squeeze()
            if self.global_pose is not None:
                return transform_points_by_matrix(vertices, self.global_pose)
            return vertices

        return build_smpl

    def define_solver(self):
        return LMOptimizerSMPL(
            jacobian_type="numeric",
            smpl_model=self.smpl_model,
            verbose=self.verbose,
            mse_threshold=1e-5,
            pose_threshold=1e-5,
            regularized=True,
            global_pose=self.global_pose,
        )
