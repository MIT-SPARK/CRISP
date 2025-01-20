import logging
import torch
from pytorch3d import ops

from crisp.utils.math import sq_half_chamfer_dists


# from casper3d.certifiability import certifiability


def keypoints_loss_batch_average(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)
    returns Tensor with one scalar value
    """

    lossMSE = torch.nn.MSELoss(reduction="none")
    kp_loss = lossMSE(kp, kp_)
    kp_loss = kp_loss.sum(dim=1).sum(dim=1).mean()  # Note: this is not normalized by number of keypoints

    return kp_loss


def keypoints_loss(kp, kp_):
    """
    kp  : torch.tensor of shape (B, 3, N)
    kp_ : torch.tensor of shape (B, 3, N)
    returns Tensor with length B
    """

    lossMSE = torch.nn.MSELoss(reduction="none")

    return lossMSE(kp, kp_).sum(1).mean(1).unsqueeze(-1)


def avg_kpt_distance_regularizer(kp):
    """ Penalize keypoints if two are close. Note that you need to subtract the result from loss.

    Args:
        kp: (B, 3, N)

    Returns:

    """
    # equivalent to:
    # kp_contiguous = torch.transpose(kp, -1, -2).contiguous()
    # euclidian_dists = torch.cdist(kp_contiguous, kp_contiguous, p=2)
    # euclidian_dists_squared = torch.square(euclidian_dists)

    # use the expanded formula for | x_i - x_j |^2 = |x_i|^2 + |j_i|^2 - 2 x_i^T x_j
    B = kp.shape[0]
    n = kp.shape[-1]
    G = torch.transpose(kp, -1, -2) @ kp
    diag_G = torch.diagonal(G, dim1=-2, dim2=-1).unsqueeze(-1)
    D = diag_G + torch.transpose(diag_G, -1, -2) - 2 * G
    # taking the average. note that the matrix is symmetrical.
    # the diagonal elements are all zero, so we divide by n*(n-1)
    return torch.sum(D) / ((n ** 2 - n) * B)


def bounded_avg_kpt_distance_loss(kp, eps_bound=1e-1):
    """ Penalize keypoints if two are closer than a threshold.
    See https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf

    Args:
        kp: (B, 3, N)

    Returns: average bounded keypoint loss of the entire batch
    """
    B = kp.shape[0]
    n = kp.shape[-1]
    G = torch.transpose(kp, -1, -2) @ kp
    diag_G = torch.diagonal(G, dim1=-2, dim2=-1).unsqueeze(-1)
    D = diag_G + torch.transpose(diag_G, -1, -2) - 2 * G
    bounded_D = torch.maximum(torch.tensor(0), eps_bound ** 2 - D)
    # note: because the diagonal entries of D are squared distances from keypoint i to itself,
    # we want to remove them from the final summation (we only want to penalize if keypoint i & j =\= i
    # are close).
    bounded_D.diagonal(dim1=-2, dim2=-1).zero_()
    return torch.sum(bounded_D) / ((n ** 2 - n) * B)


def rotation_loss(R, R_):
    device_ = R.device

    err_mat = R @ R_.transpose(-1, -2) - torch.eye(3, device=device_)
    lossMSE = torch.nn.MSELoss(reduction="mean")

    return lossMSE(err_mat, torch.zeros_like(err_mat))


def translation_loss(t, t_):
    """
    t   : torch.tensor of shape (B, 3, N)
    t_  : torch.tensor of shape (B, 3, N)

    """

    lossMSE = torch.nn.MSELoss(reduction="mean")

    return lossMSE(t, t_)


def shape_loss(c, c_):
    """
    c   : torch.tensor of shape (B, K, 1)
    c_  : torch.tensor of shape (B, K, 1)

    """

    lossMSE = torch.nn.MSELoss(reduction="mean")

    return lossMSE(c, c_)


def sdf_loss(pc, sdf_func, pc_padding=None, max_loss=False):
    """Loss function based on SDF

    Args:
        pc: torch.tensor of shape (B, 3, n)
        sdf_func: a ParamDeclarativeFunction that returns SDF values; takes in (B, 3, n) and returns (B, 1, n)
        max_loss: indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    Returns:
        loss: torch.tensor of shape (B, 1)
    """
    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

    # sdf/sdf_sq is of shape (B, 1, n)
    sdf = sdf_func.forward(pc)
    sdf_sq = torch.square(sdf)
    sdf_sq = sdf_sq.squeeze(1) * torch.logical_not(pc_padding)
    if max_loss:
        loss = sdf_sq.max(dim=1)[0]
    else:
        a = torch.logical_not(pc_padding)
        loss = sdf_sq.sum(dim=1) / a.sum(dim=1)

    return loss.unsqueeze(-1)


def chamfer_loss(pc, pc_, pc_padding=None, max_loss=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1) * torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)

    ## use pytorch to compute distance
    # sq_dist_torch = torch.cdist(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), compute_mode='donot_use_mm_for_euclid_dist').topk(k=1, dim=2, largest=False, sorted=False).values
    # sq_dist_torch = sq_dist_torch * sq_dist_torch
    # sq_dist_torch = sq_dist_torch.squeeze(-1) * torch.logical_not(pc_padding)
    # breakpoint()

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
        # loss_torch = sq_dist_torch.max()
        # logging.debug(f"chamfer loss torch3d: {loss.item()}")
        # logging.debug(f"chamfer loss pytorch: {loss_torch.item()}")
        # breakpoint()
    else:
        loss = sq_dist.sum(dim=1) / a.sum(dim=1)

    return loss.unsqueeze(-1)


def half_chamfer_loss_clamped(pc, pc_, thres, pc_padding=None, max_loss=False):
    """

    Args:
        pc: torch.tensor of shape (B, 3, n)
        pc_: torch.tensor of shape (B, 3, m)
        thres: threshold on the distance; below which we consider as inliers and above which we take a constant cost
        pc_padding: torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
        max_loss: indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    Returns:

    """
    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = (pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    # sq_dist = sq_dist.squeeze(-1) * torch.logical_not(pc_padding)
    sq_dist = sq_dist.squeeze(-1)
    # thresholding mask
    thres_mask = torch.le(sq_dist, thres ** 2)
    aa = torch.logical_and(torch.logical_not(pc_padding), thres_mask)

    sq_dist = sq_dist * aa

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
    else:
        loss = sq_dist.sum(dim=1) / aa.sum(dim=1)

    return loss.unsqueeze(-1)


# supervised training and validation losses
def supervised_training_loss(input, output):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
    pc_loss = pc_loss.mean()

    kp_loss = keypoints_loss_batch_average(input[1], output[1])

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()

    return pc_loss + kp_loss + R_loss + t_loss


def supervised_validation_loss(input, output):
    """
    inputs:
        input   : tuple of length 4 : input[0]  : torch.tensor of shape (B, 3, m) : input_point_cloud
                                      input[1]  : torch.tensor of shape (B, 3, N) : keypoints_true
                                      input[2]  : torch.tensor of shape (B, 3, 3) : rotation_true
                                      input[3]  : torch.tensor of shape (B, 3, 1) : translation_true
        output  : tuple of length 4 : output[0]  : torch.tensor of shape (B, 3, m) : predicted_point_cloud
                                      output[1]  : torch.tensor of shape (B, 3, N) : detected/corrected_keypoints
                                      output[2]  : torch.tensor of shape (B, 3, 3) : rotation
                                      output[3]  : torch.tensor of shape (B, 3, 1) : translation

    outputs:
    loss    : torch.tensor of shape (1,)

    """

    pc_loss = chamfer_loss(pc=input[0], pc_=output[0])
    pc_loss = pc_loss.mean()

    kp_loss = keypoints_loss_batch_average(input[1], output[1])

    R_loss = rotation_loss(input[2], output[2]).mean()
    t_loss = translation_loss(input[3], output[3]).mean()

    return pc_loss + kp_loss + R_loss + t_loss
    # return kp_loss


# ToDo:
# # self-supervised training and validation losses
# def certify(input_point_cloud, predicted_point_cloud, corrected_keypoints, predicted_model_keypoints,
#             epsilon=0.01, clamp_thres=0.1):
#     """
#     inputs:
#     input_point_cloud           : torch.tensor of shape (B, 3, m)
#     predicted_point_cloud       : torch.tensor of shape (B, 3, n)
#     corrected_keypoints         : torch.tensor of shape (B, 3, N)
#     predicted_model_keypoints   : torch.tensor of shape (B, 3, N)
#     epsilon                     :
#     clamp_thres                 :
#
#     outputs:
#     certificate     : torch.tensor of shape (B, 1)  : dtype = torch.bool
#
#     """
#     cert_model = certifiability(epsilon=epsilon, clamp_thres=clamp_thres)
#     out = cert_model.forward(X=input_point_cloud,
#                              Z=predicted_point_cloud,
#                              kp_=corrected_keypoints,
#                              kp=predicted_model_keypoints)
#
#     return out


def self_supervised_multimodel_training_loss(inputs, outputs, certi, theta=25.0):
    """

    Args:
        inputs: keys=method
        outputs:
        certi:
        theta:

    Returns:

    """
    # inputs:
    # 1. c3po_multi -> object_batched_pcs -> obj label

    methods = sorted(inputs.keys())
    objects_labels = sorted(certi[methods[0]].keys())

    # need to store:
    # for each obj:
    # - array with size (B): integer inside indicates whether it has no certified results (-1),
    #                        or a certified result

    # certification loop
    for obj_label in objects_labels:
        obj_loss_type = -1
        # (num methods, B)
        # OR all rows: if zero, then no certified
        # XOR all rows: if zero, then both certified
        # each method AND (XOR all rows): elements only certified in one method
        for method in methods:
            # size (B,)
            obj_certi = certi[method][obj_label]

    # For each model, check amount certified
    # If non certified, set all losses to zero
    # Else:
    #   if only one method has certified results:
    #   - treat that one as the supervised input
    #   if both methods has certified results:
    #   - use self-supervised loss on both

    # output:
    # loss -> (number of methods, number of objects)
    breakpoint()
    return None


def self_supervised_training_loss(input_point_cloud, predicted_point_cloud, keypoint_correction, certi, theta=25.0):
    """
    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    keypoint_correction     : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    loss    : torch.tensor of shape (1,)

    """
    device_ = input_point_cloud.device
    theta = torch.tensor([theta]).to(device=device_)

    if certi.sum() == 0:
        logging.info("NO DATA POINT CERTIFIABLE IN THIS BATCH")
        pc_loss = torch.tensor([0.0]).to(device=device_)
        kp_loss = torch.tensor([0.0]).to(device=device_)
        fra_certi = torch.tensor([0.0]).to(device=device_)
        pc_loss.requires_grad = True
        kp_loss.requires_grad = True
        fra_certi.requires_grad = True

    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]  # not to be used for training

        # pc loss
        pc_loss = chamfer_loss(
            pc=input_point_cloud, pc_=predicted_point_cloud
        )  # Using normal chamfer loss here, as the max chamfer is used in certification
        pc_loss = pc_loss * certi
        pc_loss = pc_loss.sum() / num_certi

        lossMSE = torch.nn.MSELoss(reduction="none")
        if keypoint_correction is None:
            kp_loss = torch.zeros(pc_loss.shape)
        else:
            kp_loss = lossMSE(keypoint_correction, torch.zeros_like(keypoint_correction))
            kp_loss = kp_loss.sum(dim=1).mean(dim=1)  # (B,)
            kp_loss = kp_loss * certi
            kp_loss = kp_loss.mean()

    # return pc_loss + theta*kp_loss, pc_loss, kp_loss, fra_certi   # pointnet
    return (
        theta * pc_loss + kp_loss,
        pc_loss,
        kp_loss,
        fra_certi,
    )  # point_transformer: we will try this, as the first gave worse performance for pointnet.


def self_supervised_training_robust_loss(
        input_point_cloud, predicted_point_cloud, keypoint_correction, certi, clamp_thres=100, theta=25.0
):
    """A robust version of self-supervised training loss. Will try to clamp outliers

    inputs:
    input_point_cloud       : torch.tensor of shape (B, 3, m)
    predicted_point_cloud   : torch.tensor of shape (B, 3, n)
    keypoint_correction     : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    loss    : torch.tensor of shape (1,)

    """
    device_ = input_point_cloud.device
    theta = torch.tensor([theta]).to(device=device_)

    if certi.sum() == 0:
        logging.info("NO DATA POINT CERTIFIABLE IN THIS BATCH")
        pc_loss = torch.tensor([0.0]).to(device=device_)
        kp_loss = torch.tensor([0.0]).to(device=device_)
        fra_certi = torch.tensor([0.0]).to(device=device_)
        pc_loss.requires_grad = True
        kp_loss.requires_grad = True
        fra_certi.requires_grad = True

    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]  # not to be used for training

        # 1. calculate the squared chamfer distances
        valid_pc_mask = (input_point_cloud != torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3
        sq_pc_dists = sq_half_chamfer_dists(pc=input_point_cloud, pc_=predicted_point_cloud)

        # 2. calculate the clamp mask
        pc_thres_mask = torch.le(sq_pc_dists, clamp_thres ** 2)
        valid_pc_mask = torch.logical_and(valid_pc_mask, pc_thres_mask)
        num_valid_pc_points = valid_pc_mask.sum(dim=1)
        pc_loss = sq_pc_dists * valid_pc_mask / num_valid_pc_points

        # 3. backprop only the non clamp & certified instances
        pc_loss = pc_loss * certi
        pc_loss = pc_loss.sum() / num_certi

        lossMSE = torch.nn.MSELoss(reduction="none")
        if keypoint_correction is None:
            kp_loss = torch.zeros(pc_loss.shape)
        else:
            kp_loss = lossMSE(keypoint_correction, torch.zeros_like(keypoint_correction))
            kp_loss = kp_loss.sum(dim=1).mean(dim=1)  # (B,)
            kp_loss = kp_loss * certi
            kp_loss = kp_loss.mean()

    # return pc_loss + theta*kp_loss, pc_loss, kp_loss, fra_certi   # pointnet
    return (
        theta * pc_loss + kp_loss,
        pc_loss,
        kp_loss,
        fra_certi,
    )  # point_transformer: we will try this, as the first gave worse performance for pointnet.


def self_supervised_validation_loss(input_pc, predicted_pc, certi=None):
    """
    inputs:
        input_pc        : torch.tensor of shape (B, 3, m) : input_point_cloud
        predicted_pc    : torch.tensor of shape (B, 3, n) : predicted_point_cloud
        certi           : None or torch.tensor(dtype=torch.bool) of shape (B,)  : certification

    outputs:
        loss    : torch.tensor of shape (1,). Negative of the amount certified.

    """

    if certi == None:
        pc_loss = chamfer_loss(pc=input_pc, pc_=predicted_pc)
        vloss = pc_loss.mean()
    else:
        # fra certi
        num_certi = certi.sum()
        fra_certi = num_certi / certi.shape[0]
        vloss = -fra_certi

    return vloss


if __name__ == "__main__":
    print("hi")
