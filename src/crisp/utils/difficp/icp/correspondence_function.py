"""Functions for ICP Correspondence Matching and Weighting/Rejection"""

import torch

from crisp.utils.difficp.utils.geometry_utils import global_to_screen_projection


def get_correspondence_function(
    matching_type,
    matching_temperature=0,
    weight_dist=False,
    weight_normal=False,
    rejection_dist=None,
    rejection_normal=None,
    rejection_temperature=0,
):
    """get correspondence function as matching + rejection"""
    matching_function = get_matching_function(matching_type, matching_temperature)

    def rejection_function(*args, **kwargs):
        return correspondence_weighting_and_rejection(
            *args,
            **kwargs,
            weight_dist=weight_dist,
            weight_normal=weight_normal,
            rejection_dist=rejection_dist,
            rejection_normal=rejection_normal,
            temperature=rejection_temperature,
        )

    def match_and_reject(
        sources,
        targets,
        source_normals=None,
        target_normals=None,
        source_tensors=None,
        target_tensors=None,
        **matching_kwargs,
    ):
        source_tensors = [] if source_tensors is None else source_tensors
        target_tensors = [] if target_tensors is None else target_tensors
        source_tensors.append(source_normals)
        target_tensors.append(target_normals)
        sources, targets, source_tensors, target_tensors = matching_function(
            sources=sources,
            targets=targets,
            source_tensors=source_tensors,
            target_tensors=target_tensors,
            **matching_kwargs,
        )
        source_normals = source_tensors.pop()
        target_normals = target_tensors.pop()
        return rejection_function(
            sources=sources,
            targets=targets,
            source_normals=source_normals,
            target_normals=target_normals,
            source_tensors=source_tensors,
            target_tensors=target_tensors,
        )

    return match_and_reject


def get_matching_function(matching_type, matching_temperature=0):
    """choose correspondence function based on matching_type"""
    if matching_type == "nn":
        return lambda *args, **kwargs: match_points_nn(
            *args,
            **kwargs,
            temperature=matching_temperature,
        )
    if matching_type == "projective":
        return match_points_projective
    raise ValueError(f"matching_type {matching_type} not recognized.")


def filter_tensor(tensor, indices):
    """index a tensor with given indices, or return None if tensor is None"""
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor[indices]


def filter_tensor_list(tensors, indices):
    """index all tensors in a list with given indices, or return None if list is None"""
    if tensors is None:
        return None
    return [filter_tensor(tensor, indices) for tensor in tensors]


class NearestNeighbors:
    """
    Pytorch Differentiable 1-Nearest Neighbor
    used by 1) calling nn=NearestNeighbor(targets) 2) neighbors=nn(sources)
    optionally, selection can be applied to other tensors with similar shape as targets
        by using other_neighbors=nn.apply(other_tensor)
    selection depends on temperature parameter:
        if temperature=0, we perform standard non-differentiable nearest neighbors with
            hard argmin selection
        if temperature>0, we perform differentiable soft nearest neighbors as linear
            combination of all target points based on softmax over negative distances
    """

    def __init__(self, targets, temperature=0):
        """
        :param targets: target points in which we want to find the nearest neighbors
        :param temperature: temperature parameter >=0, defines smoothness of function
        """
        self.targets = targets
        self.temperature = temperature
        self.do_soft_selection = temperature > 0
        self.dists = None
        self.weights = None

    @staticmethod
    def pairwise_dist(x, y):
        """calculate the pairwise distances between two sets of points (x,y)"""
        xx = torch.mm(x, x.t())  # NxN
        yy = torch.mm(y, y.t())  # MxM
        zz = torch.mm(x, y.t())  # NxM
        diag_xx = xx.diag().unsqueeze(1)  # Nx1
        diag_yy = yy.diag().unsqueeze(0)  # 1xM
        pairwise_dists = diag_xx + diag_yy - 2 * zz  # NxM
        return pairwise_dists  # NxM

    @staticmethod
    def pairwise_dist_lowmem(x, y):
        xx = torch.sum(x * x, dim=-1, keepdim=True)  # Nx1
        yy = torch.sum(y * y, dim=-1, keepdim=True).t()  # 1xM
        zz = torch.matmul(x, y.t())  # NxM
        zz *= -2  # NxM
        zz += xx  # NxM
        zz += yy  # NxM
        return zz

    def differentiable_selection(self, dists, temperature=0):
        """
        computes the continuous deterministic relaxation of 1-nearest-neighbors
        based on the paper Neural Nearest Neighbors Networks by Ploetz et al.
        :param dists: pair-wise distances from sources to targets, shape NxM
        :param temperature: defines smoothness selection, scalar >= 0
        :return: nearest neighbor probabilities for each source, shape N x M
        """
        alpha = -dists / temperature  # N x M
        weights = torch.softmax(alpha, dim=-1)  # N x M
        return weights

    def apply(self, targets):
        """
        apply previously found selection to given targets (needed to handle normals/...)
        :param targets: targets, must have same first dimension size as self.targets
        :return: closest target for each source point
        """
        if targets is None:
            return None
        assert targets.shape[0] == self.targets.shape[0]
        if self.do_soft_selection:
            return torch.mm(self.weights, targets)
        return targets[torch.argmin(self.dists, dim=-1)]

    def __call__(self, sources, *args, **kwargs):
        """
        find the nearest target for each source point
        :param sources: source points, shape Nx3
        :return: closest target for each source point, shape Nx3
        """
        self.dists = self.pairwise_dist(sources, self.targets)  # N x M
        if self.do_soft_selection:
            self.weights = self.differentiable_selection(self.dists, self.temperature)
        return self.apply(self.targets)


def match_points_nn(
    sources, targets, source_tensors=None, target_tensors=None, temperature=0, **kwargs
):
    """
    Find nearest neighbors in target_points for all source_points
    :param sources: source points, Nx3 tensor of x-y-z coordinates
    :param targets: target points, Mx3 tensor of x-y-z coordinates
    :param source_tensors: list of additional tensors corresponding to sources.
        these tensors also need to be filtered, each tensor is of shape Nx?
    :param target_tensors: list of additional tensors corresponding to targets.
        these tensors also need to be filtered, each tensor is of shape Mx?
    :param temperature: defines smoothness of nearest neighbor selection, scalar >= 0
    :return:
        - sources, shape Kx3
        - targets, shape Kx3
        - source_tensors, list of tensors, each tensor of shape Kx?
        - target_tensors, list of tensors, each tensor of shape Kx?
    """
    knn = NearestNeighbors(targets, temperature=temperature)
    sources = sources.clone()
    targets = knn(sources)  # N x M
    if source_tensors is not None:
        source_tensors = [x.clone() if x is not None else None for x in source_tensors]
    if target_tensors is not None:
        target_tensors = [knn.apply(tensor) for tensor in target_tensors]
    return sources, targets, source_tensors, target_tensors


def match_points_projective(
    sources,
    targets,
    source_tensors=None,
    target_tensors=None,
    cam_intr=None,
    cam_pose=None,
    **kwargs,
):
    """
    Match points via projective correlation
    :param sources: source points, Nx3 matrix of x-y-z coordinates
    :param targets: target depth map, HxWx3 matrix containing x-y-z coordinates
    :param cam_intr: camera intrinsics matrix, shape 4x4
    :param cam_pose: camera pose (extrinsics) matrix, shape 4x4
    :param source_tensors: list of additional tensors corresponding to sources.
        these tensors also need to be filtered, each tensor is of shape Nx?
    :param target_tensors: list of additional tensors corresponding to targets.
        these tensors also need to be filtered, each tensor is of shape HxWx?
    :return:
        - sources, shape Kx3
        - targets, shape Kx3
        - source_tensors, list of tensors, each tensor of shape Kx?
        - target_tensors, list of tensors, each tensor of shape Kx?
    """
    height, width = targets.shape[:2]
    sources_screen = global_to_screen_projection(sources, cam_intr, cam_pose)
    xy_screen = sources_screen[:, :2].round().to(torch.int64)
    x_screen, y_screen = xy_screen[:, 0], xy_screen[:, 1]
    x_in_img = torch.logical_and(0 <= x_screen, x_screen < width)
    y_in_img = torch.logical_and(0 <= y_screen, y_screen < height)
    p_in_img = torch.logical_and(x_in_img, y_in_img)
    y_screen_or_0 = torch.where(p_in_img, y_screen, torch.zeros_like(y_screen))
    x_screen_or_0 = torch.where(p_in_img, x_screen, torch.zeros_like(x_screen))
    zero = torch.zeros_like(sources[:, 0])
    matched_target = targets[y_screen_or_0, x_screen_or_0, 2]
    proj_z = torch.where(p_in_img, matched_target, zero)
    source_indices = torch.nonzero(proj_z).reshape(-1)
    if len(source_indices) == 0:
        sources = torch.tensor([])
        targets = torch.tensor([])
        source_tensors = [torch.tensor([]) for _ in range(len(source_tensors))]
        target_tensors = [torch.tensor([]) for _ in range(len(target_tensors))]
        return sources, targets, source_tensors, target_tensors
    target_indices = y_screen[source_indices] * width + x_screen[source_indices]
    dists_to_camera = torch.abs(sources[source_indices, 2])
    corr_dict = {}
    _, sorted_indices = torch.sort(dists_to_camera, descending=True)
    for i in sorted_indices:
        corr_dict[target_indices[i].item()] = source_indices[i].item()
    dtype, device = source_indices.dtype, source_indices.device
    source_indices = torch.tensor(list(corr_dict.values()), dtype=dtype, device=device)
    target_indices = torch.tensor(list(corr_dict.keys()), dtype=dtype, device=device)
    sources = sources[source_indices]
    targets = targets.reshape(-1, 3)[target_indices]
    source_tensors = filter_tensor_list(tensors=source_tensors, indices=source_indices)
    target_tensors = filter_tensor_list(tensors=target_tensors, indices=target_indices)
    return sources, targets, source_tensors, target_tensors


def tensor_dot(tensor1, tensor2, dim=-1, keepdim=False):
    """
    Element-wise dot product between two tensors of similar shapes
    used for correspondence rejection/weighting based on normals in ICP
    :param tensor1: first tensor
    :param tensor2: second tensor
    :param dim: dimension to reduce
    :param keepdim: whether the output tensor has dim retained or not
    :return: element-wise dot product between the tensors over dimension dim
    """
    assert tensor1.shape == tensor2.shape
    return torch.sum(tensor1 * tensor2, dim=dim, keepdim=keepdim)


def correspondence_weighting_by_distance(sources, targets):
    """
    Calculate correspondence weights in [0,1] based on distance between points
    :param sources: source points, shape Kx3
    :param targets: target points, shape Kx3
    :return: one weight per correspondence as 1-d(s,t)/max(d), shape Kx1
    """
    dists = torch.sum(torch.abs((sources - targets)), dim=-1, keepdim=True)
    weights = 1 - dists / torch.max(dists)
    return weights


def correspondence_weighting_by_normal(source_normals, target_normals):
    """
    Calculate correspondence weights in [0,1] based on normal dot product
    :param source_normals: normal of each source point, shape Kx3
    :param target_normals: normal of each target point, shape Kx3
    :return: one weight per correspondence as max(dot(normal1, normal2), 0), shape Kx1
    """
    normaldot = tensor_dot(source_normals, target_normals, keepdim=True)
    weights = torch.nn.functional.relu(normaldot)
    return weights


def correspondence_rejection_by_distance(sources, targets, threshold):
    """
    Soft correspondence rejection based on absolute distances
    :param sources: source points, shape Kx3
    :param targets: target points, shape Kx3
    :param threshold: threshold for max non-rejected distance
    :return: boolean tensor, True where not rejected, shape K
    """
    dists = torch.sum(torch.abs((sources - targets)), dim=-1)
    not_rejected_inds = dists < threshold
    return not_rejected_inds


def correspondence_rejection_by_normal(source_normals, target_normals, threshold):
    """
    Correspondence rejection based on normal dot product
    :param source_normals: normal for each source point, shape Kx3
    :param target_normals: normal for each target point, shape Kx3
    :param threshold: threshold for min non-rejected normal dot product
    :return: boolean tensor, True where not rejected, shape K
    """
    normal_dot = tensor_dot(source_normals, target_normals)
    not_rejected_inds = normal_dot > threshold
    return not_rejected_inds


def soft_correspondence_rejection_by_distance(sources, targets, threshold, temperature):
    """
    Soft correspondence rejection based on sigmoids over absolute distances
    :param sources: source points, shape Kx3
    :param targets: target points, shape Kx3
    :param threshold: threshold for max non-rejected distance
    :param temperature: defines smoothness of rejection function, scalar >= 0
    :return: weights, one weight in [0,1] per correspondence, shape Kx1
    """
    dists = torch.sum(torch.abs((sources - targets)), dim=-1, keepdim=True)
    weights = torch.sigmoid((threshold - dists) / temperature)
    return weights


def soft_correspondence_rejection_by_normal(
    source_normals, target_normals, threshold, temperature
):
    """
    Soft correspondence rejection based on sigmoids over normal dot product
    :param source_normals: normal for each source point, shape Kx3
    :param target_normals: normal for each target point, shape Kx3
    :param threshold: threshold for min non-rejected normal dot product
    :param temperature: defines smoothness of rejection function, scalar >= 0
    :return: weights, one weight in [0,1] per correspondence, shape Kx1
    """
    normal_dot = tensor_dot(source_normals, target_normals, keepdim=True)
    weights = torch.sigmoid((normal_dot - threshold) / temperature)
    return weights


def correspondence_weighting_and_rejection(
    sources,
    targets,
    source_normals=None,
    target_normals=None,
    source_tensors=None,
    target_tensors=None,
    weight_dist=False,
    weight_normal=False,
    rejection_dist=None,
    rejection_normal=None,
    temperature=0,
):
    """
    Correspondence rejection, depending on temperature parameter:
        if temperature=0, we perform standard non-differentiable hard rejection
        if temperature>0, we perform differentiable soft rejection
    :param sources: source points, shape Kx3
    :param targets: target points, shape Kx3
    :param source_normals: normal for each source point, shape Kx3
    :param target_normals: normal for each target point, shape Kx3
    :param source_tensors: list of additional tensors corresponding to sources.
        these tensors also need to be filtered, each tensor is of shape Kx?
    :param target_tensors: list of additional tensors corresponding to targets.
        these tensors also need to be filtered, each tensor is of shape Kx?
    :param weight_dist: whether to weight correspondences based on distances
    :param weight_normal: whether to weight correspondences based on normals
    :param rejection_dist: threshold for max non-rejected distance
    :param rejection_normal: threshold for min non-rejected normal dot product
    :param temperature: defines smoothness of rejection function, scalar >= 0
    :return:
        - sources of shape Lx3, L<=K
        - targets of shape Lx3, L<=K
        - source_normals of shape Lx3, L<=K
        - target_normals of shape Lx3, L<=K
        - source_tensors, list of tensors, each tensor of shape Lx?, L<=K
        - target_tensors, list of tensors, each tensor of shape Lx?, L<=K
        - correspondence_weights, one weight in [0,1] per correspondence, shape Lx1
    """
    assert sources.shape == targets.shape
    if source_normals is not None:
        assert sources.shape == source_normals.shape
    if target_normals is not None:
        assert targets.shape == target_normals.shape

    weights = None
    do_rejection = rejection_dist is not None or rejection_normal is not None
    do_weighting = weight_dist or weight_normal

    # hard correspondence rejection
    if do_rejection and temperature <= 0:
        not_rejected = torch.ones(len(sources), dtype=torch.bool, device=sources.device)
        if rejection_dist is not None:
            not_rejected_ = correspondence_rejection_by_distance(
                sources, targets, rejection_dist
            )
            not_rejected = torch.logical_and(not_rejected, not_rejected_)
        if rejection_normal is not None:
            not_rejected_ = correspondence_rejection_by_normal(
                source_normals, target_normals, rejection_normal
            )
            not_rejected = torch.logical_and(not_rejected, not_rejected_)
        sources = sources[not_rejected]
        targets = targets[not_rejected]
        source_normals = filter_tensor(source_normals, not_rejected)
        target_normals = filter_tensor(target_normals, not_rejected)
        source_tensors = filter_tensor_list(source_tensors, not_rejected)
        target_tensors = filter_tensor_list(target_tensors, not_rejected)

    # soft correspondence rejection
    elif do_rejection and temperature > 0:
        weights = torch.ones_like(sources[:, :1])
        if rejection_dist is not None:
            weights = weights * soft_correspondence_rejection_by_distance(
                sources, targets, rejection_dist, temperature
            )
        if rejection_normal is not None:
            weights = weights * soft_correspondence_rejection_by_normal(
                source_normals, target_normals, rejection_normal, temperature
            )

    # correspondence weighting
    if do_weighting and len(sources > 0):
        weights = torch.ones_like(sources[:, :1]) if weights is None else weights
        if weight_dist:
            weights = weights * correspondence_weighting_by_distance(sources, targets)
        if weight_normal:
            weights = weights * correspondence_weighting_by_normal(
                source_normals, target_normals
            )

    return (
        sources,
        targets,
        source_normals,
        target_normals,
        source_tensors,
        target_tensors,
        weights,
    )
