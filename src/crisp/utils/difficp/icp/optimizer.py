"""Differentiable Optimizers"""

import torch
from abc import ABC, abstractmethod
from torch import sin, cos

from crisp.utils.difficp.utils.geometry_utils import (
    transform_points_by_pose,
    pose_to_matrix,
    transform_points_by_matrix,
)
from .distance_function import get_distance_function
from .linear_solver import LinearSolverLU


class LMOptimizer(ABC):
    """Differentiable Levenberg-Marquardt Optimizer"""

    def __init__(
        self,
        lm_lambda=1e-5,
        lm_v=1.5,
        iters_max=1000,
        mse_threshold=0,
        pose_threshold=0,
        jacobian_type="numeric",
        verbose=False,
    ):
        """
        Initialize optimizer
        :param lm_lambda: initial trust region size
        :param lm_v: trust region update factor
        :param iters_max: maximum number of LM iterations
        :param mse_threshold: MSE difference threshold to check convergence
        :param pose_threshold: pose difference threshold to check convergence
        :param verbose: if true, print progress information
        """
        self.lm_lambda = lm_lambda
        self.lm_v = lm_v
        self.iters_max = iters_max
        self.mse_threshold = mse_threshold
        self.pose_threshold = pose_threshold
        self.jacobian_type = jacobian_type
        self.dist_fn = self.define_dist_fn()
        self.jacobian_method = self.define_jacobian_method()
        self.verbose = verbose

    @abstractmethod
    def define_dist_fn(self):
        """define distance function that assign an error for (pose, sources, targets)"""

    def define_jacobian_method(self):
        """define method to construct jacobian"""
        return {
            "autograd": self.autograd_jacobian,
            "numeric": self.numeric_jacobian,
        }[self.jacobian_type]

    def __call__(self, pose, sources, targets, **kwargs):
        """
        solve point-to-point procrustes using LM Optimizer
        :param pose: initial pose (alpha, beta, gamma, tx, ty, tz), shape 6
        :param sources: source points, shape Kx3
        :param targets: target points, shape Kx3
        :param kwargs: kwargs passed to both self.dist_fn and self.jacobian_method
        :return: best transformation between source and target, shape 6
        """
        sources_ = sources.reshape(-1, 3)
        targets_ = targets.reshape(-1, 3)

        def pose_to_mse(pose_):
            """Calculate mean squared error over residuals for a given pose"""
            dists_ = self.dist_fn(pose_, sources_, targets_, **kwargs)
            mse_ = torch.mean(dists_ ** 2)
            return mse_, dists_

        lm_lambda, lm_v = self.lm_lambda, self.lm_v  # init lambda and v
        for i in range(self.iters_max):
            last_mse, dists = pose_to_mse(pose)
            jacobian = self.jacobian_method(sources_, targets_, pose, **kwargs)
            gradient = torch.matmul(jacobian.t(), dists.view(-1, 1))
            jtj = torch.matmul(jacobian.t(), jacobian)

            def perform_lm_update(lambda_, grad_=gradient, jtj_=jtj):
                """Perform a LM pose update using trust region size lambda"""
                hessian = jtj_ + lambda_ * torch.diag(torch.diag(jtj_))
                delta = LinearSolverLU.apply(hessian, grad_).view(-1)
                return pose - delta

            # Marquardt's trust region update guideline:
            # 1) if lambda/v leads to lower error, set lambda=lambda/v and update pose
            # 2) elif lambda leads to lower error, keep lambda and update pose
            # 3) if both don't lower the error, do lambda=lambda*v until improvement
            lm_lambda /= lm_v
            new_pose = perform_lm_update(lm_lambda)
            mse = pose_to_mse(new_pose)[0]
            j = 0
            while mse > last_mse:
                j += 1
                lm_lambda *= lm_v
                new_pose = perform_lm_update(lm_lambda)
                mse = pose_to_mse(new_pose)[0]
                if self.verbose:
                    print("\t\t", j, mse, last_mse, lm_lambda)
            pose_diff = torch.mean(torch.abs(pose - new_pose))
            pose = new_pose

            # stop iteration if mse or pose difference becomes too small
            mse_diff = torch.abs(last_mse - mse)
            if self.verbose:
                print("\t", i, mse, mse_diff, pose_diff)
            if mse_diff <= self.mse_threshold or pose_diff <= self.pose_threshold:
                return pose
        return pose

    def numeric_jacobian(self, sources, targets, pose, eps=1e-5, **kwargs):
        """
        Calculate numeric jacobian using central differences
        :param sources: source points, Kx3 matrix of x-y-z coordinates
        :param targets: target points, Kx3 matrix of x-y-z coordinates
        :param pose: 6DoF pose [alpha, beta, gamma, tx, ty, tz]
        :param eps: epsilon in central differences
        :return: jacobian matrix, shape 3Kx6
        """
        jacobian = []
        for i in range(len(pose)):
            pose1 = pose.clone()
            pose2 = pose.clone()
            pose1[i] += eps
            pose2[i] -= eps
            dist1 = self.dist_fn(pose1, sources, targets, **kwargs)
            dist2 = self.dist_fn(pose2, sources, targets, **kwargs)
            grad = (dist1 - dist2) / (2 * eps)
            jacobian.append(grad.view(-1))
        return torch.stack(jacobian, -1)

    def autograd_jacobian(self, sources, targets, pose, **kwargs):
        return torch.autograd.functional.jacobian(
            lambda pose_: self.dist_fn(pose_, sources, targets, **kwargs).view(-1), pose
        )


class LMOptimizer6DoF(LMOptimizer):
    def __init__(
        self,
        *args,
        dist_type="point",
        w_point=0,
        w_plane=0,
        w_symmetric=0,
        **kwargs,
    ):
        self.dist_type = dist_type
        self.w_point = w_point
        self.w_plane = w_plane
        self.w_symmetric = w_symmetric
        self.dist_fn_6dof = get_distance_function(
            dist_type=dist_type,
            point_weight=w_point,
            plane_weight=w_plane,
            symmetric_weight=w_symmetric,
        )
        super().__init__(*args, **kwargs)

    def __call__(self, pose, *args, **kwargs):
        zero_pose = torch.zeros(6, dtype=pose.dtype, device=pose.device)
        pose_change = super().__call__(zero_pose, *args, **kwargs)
        pose_change = pose_to_matrix(pose_change)
        return torch.matmul(pose_change, pose)

    def define_dist_fn(self):
        def dist_6dof(pose, sources, targets, **kwargs):
            sources_ = transform_points_by_pose(sources, pose)
            return self.dist_fn_6dof(sources_, targets, **kwargs)

        return dist_6dof

    def define_jacobian_method(self):
        return {
            "autograd": self.autograd_jacobian,
            "numeric": self.numeric_jacobian,
            "analytic": self.analytic_jacobian,
        }[self.jacobian_type]

    def analytic_jacobian(
        self,
        sources,
        targets,
        pose,
        source_normals=None,
        target_normals=None,
        weights=None,
        **_,
    ):
        """
        Calculate analytic jacobian for 6DoF point-to-point alignment
        :param sources: source points, Kx3 matrix of x-y-z coordinates
        :param targets: target points, Kx3 matrix of x-y-z coordinates
        :param pose: 6DoF pose [alpha, beta, gamma, tx, ty, tz]
        :param source_normals: normals of each source point, shape Kx3
        :param target_normals: normals of each target point, shape Kx3
        :param weights: correspondence weights, shape Kx1
        :return: jacobian matrix, shape LKx6 for L in {1, 2, 3, 4, 5}
        """
        sources_ = transform_points_by_pose(sources, pose)
        num_correspondences = sources_.shape[0]
        ones = torch.ones(num_correspondences, device=pose.device, dtype=pose.dtype)
        zeros = torch.zeros(num_correspondences, device=pose.device, dtype=pose.dtype)
        alpha, beta, gamma = pose[:3]
        sina, cosa = sin(alpha), cos(alpha)
        sinb, cosb = sin(beta), cos(beta)
        sing, cosg = sin(gamma), cos(gamma)
        x, y, z = sources[:, 0], sources[:, 1], sources[:, 2]  # K

        dldalpha = [
            (
                (cosg * sinb * cosa + sing * sina) * y
                + (-cosg * sinb * sina + sing * cosa) * z
            ),
            (
                (sing * sinb * cosa - sing * sina) * y
                + (-sing * sinb * cosa - cosg * cosa) * z
            ),
            ((cosb * cosa) * y + (-cosb * sina) * z),
        ]
        dldalpha = torch.stack(dldalpha, -1)  # Kx3
        dldbeta = [
            ((-cosg * sinb) * x + (cosg * cosb * sina) * y + (cosg * cosb * cosa) * z),
            ((-sing * sinb) * x + (sing * cosb * sina) * y + (sing * cosb * cosa) * z),
            ((-cosg) * x + (-sinb * sina) * y + (-sinb * cosa) * z),
        ]
        dldbeta = torch.stack(dldbeta, -1)  # Kx3
        dldgamma = [
            (
                (-sing * cosb) * x
                + (-sing * sinb * sina - cosg * cosa) * y
                + (-sing * sinb * cosa + cosg * sina) * z
            ),
            (
                (cosg * cosb) * x
                + (cosg * sinb * sina - sing * cosa) * y
                + (cosg * sinb * cosa + sing * sina) * z
            ),
            zeros,
        ]
        dldgamma = torch.stack(dldgamma, -1)  # Kx3
        dldtx = torch.stack([ones, zeros, zeros], -1)  # Kx3
        dldty = torch.stack([zeros, ones, zeros], -1)  # Kx3
        dldtz = torch.stack([zeros, zeros, ones], -1)  # Kx3
        grads = [dldalpha, dldbeta, dldgamma, dldtx, dldty, dldtz]  # each Kx3

        if weights is not None:
            weights = weights.view(-1, 1)  # Kx1
            grads = [grad * weights for grad in grads]  # each Kx3

        jacobian = torch.tensor([], device=sources.device)
        if self.w_point > 0:
            jacobian_point = torch.stack([grad.reshape(-1) for grad in grads], -1)
            jacobian = torch.cat([jacobian, jacobian_point * self.w_point], 0)
        if self.w_plane > 0:
            jacobian_plane = torch.stack(
                [torch.sum(grad * target_normals, dim=-1) for grad in grads], -1
            )
            jacobian = torch.cat([jacobian, jacobian_plane * self.w_plane], 0)
        if self.w_symmetric > 0:
            symmetric_normals = source_normals + target_normals
            jacobian_symmetric = torch.stack(
                [torch.sum(grad * symmetric_normals, dim=-1) for grad in grads], -1
            )
            jacobian = torch.cat([jacobian, jacobian_symmetric * self.w_symmetric], 0)

        return jacobian / max(self.w_point, self.w_plane, self.w_symmetric)


class LMOptimizerSMPL(LMOptimizer):
    def __init__(
        self, smpl_model, *args, regularized=False, global_pose=None, **kwargs
    ):
        self.smpl_model = smpl_model
        self.regularized = regularized
        self.global_pose = global_pose
        super().__init__(*args, **kwargs)

    def define_dist_fn(self):
        class SMPLDist:
            def __init__(self, smpl_model, global_pose=None):
                self.smpl_model = smpl_model
                self.global_pose = global_pose
                self.num_pose_params = 23 * 3
                self.num_shape_params = 10 * 3

            def __call__(self, pose, sources, targets, **unused_kwargs):
                pose_ = pose[: self.num_pose_params].view(1, -1)
                shape_ = pose[self.num_pose_params :].view(1, -1)
                output = self.smpl_model(betas=shape_, body_pose=pose_)
                sources_ = output.vertices.detach().squeeze()
                if self.global_pose is not None:
                    sources_ = transform_points_by_matrix(sources_, self.global_pose)
                return (sources_ - targets).reshape(-1)

        dist_fn = SMPLDist(self.smpl_model, self.global_pose)
        if self.regularized:
            return self._regularize_dist_fn(dist_fn)
        return dist_fn

    def define_jacobian_method(self):
        jacobian_method = {
            "autograd": self.autograd_jacobian,
            "numeric": self.numeric_jacobian,
        }[self.jacobian_type]
        if self.regularized and self.jacobian_type == "analytic":
            return self._regularize_jacobian(jacobian_method)
        return jacobian_method

    @staticmethod
    def _regularize_dist_fn(dist_fn):
        def wrapper(pose, sources, targets, **kwargs):
            dist = dist_fn(pose, sources, targets, **kwargs)
            reg = pose ** 2
            return torch.cat([dist, reg])

        return wrapper

    @staticmethod
    def _regularize_jacobian(jacobian_method):
        def wrapper(sources, targets, pose, **kwargs):
            jacobian = jacobian_method(sources, targets, pose, **kwargs)
            reg = torch.diag(pose) * 2
            return torch.cat([jacobian, reg], dim=0)

        return wrapper
