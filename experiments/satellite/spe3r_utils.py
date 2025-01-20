import numpy as np
import torch
import trimesh
from pytorch3d.loss import chamfer_distance

from crisp.utils.evaluation_metrics import translation_error, rotation_error
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh


def l1_l2_chamfer_dists(est_cam_cad, gt_cam_cad):
    # update metrics
    bs = est_cam_cad.shape[0]
    cf_l1, _ = chamfer_distance(
        est_cam_cad,
        gt_cam_cad,
        point_reduction="mean",
        batch_reduction=None,
        norm=1,
    )
    cf_l2, _ = chamfer_distance(
        est_cam_cad,
        gt_cam_cad,
        point_reduction="mean",
        batch_reduction=None,
        norm=2,
    )
    cf_l1 = cf_l1.cpu().numpy().reshape(bs, -1)
    cf_l2 = cf_l2.cpu().numpy().reshape(bs, -1)
    return cf_l1, cf_l2


def calculate_metrics(
    model,
    gt_cam_T_world,
    est_cam_T_world,
    gt_shape_names,
    batched_shape_code,
    shape_bounds,
    object_pc,
    calculate_pred_recons_metrics=True,
):
    bs = gt_cam_T_world.shape[0]
    cam_t_cad = est_cam_T_world[..., :3, -1]
    cam_R_world = est_cam_T_world[..., :3, :3]
    trans_err = translation_error(cam_t_cad.unsqueeze(-1), gt_cam_T_world[:, :3, -1].unsqueeze(-1))
    rot_err = rotation_error(cam_R_world, gt_cam_T_world[..., :3, :3])

    # transformed things
    est_cam_cad = est_cam_T_world[:, :3, :3] @ torch.transpose(object_pc, 1, 2) + est_cam_T_world[:, :3, 3].reshape(
        -1, 3, 1
    )
    gt_cam_cad = gt_cam_T_world[:, :3, :3] @ torch.transpose(object_pc, 1, 2) + gt_cam_T_world[:, :3, 3].reshape(
        -1, 3, 1
    )

    pose_cf_l1, pose_cf_l2 = l1_l2_chamfer_dists(torch.transpose(est_cam_cad, 1, 2), torch.transpose(gt_cam_cad, 1, 2))

    # marching cubes
    cad_chamfer_l1, cad_chamfer_l2 = [None for _ in range(bs)], [None for _ in range(bs)]
    if calculate_pred_recons_metrics:
        # marching cubes
        for j in range(bs):
            shape_code = batched_shape_code[j, ...]

            def model_fn(coords):
                return model.recons_net.forward(shape_code=shape_code.unsqueeze(0), coords=coords)

            cube_scale = shape_bounds[1] - shape_bounds[0]
            (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                model_fn=model_fn,
                N=64,
                max_batch=64**3,
                cube_center=np.array([0, 0, 0]),
                cube_scale=cube_scale,
            )

            pred_mesh = convert_sdf_samples_to_mesh(
                sdf_grid=sdf_grid,
                voxel_grid_origin=voxel_grid_origin,
                voxel_size=voxel_size,
                offset=None,
                scale=None,
            )
            sampled_mesh, _ = trimesh.sample.sample_surface(pred_mesh, 2500)

            cad_cf_l1, cad_cf_l2 = l1_l2_chamfer_dists(
                torch.tensor(sampled_mesh).unsqueeze(0).cuda().float(), object_pc[j].unsqueeze(0)
            )
            cad_chamfer_l1[j] = cad_cf_l1.item()
            cad_chamfer_l2[j] = cad_cf_l2.item()

    metrics = []
    for j in range(bs):
        metrics.append(
            {
                "model_name": gt_shape_names[j],
                "eT": trans_err[j].cpu().item(),
                "eR": rot_err[j].cpu().item(),
                "pose_chamfer_l1": pose_cf_l1[j].item(),
                "pose_chamfer_l2": pose_cf_l2[j].item(),
                "chamfer_l1": cad_chamfer_l1[j],
                "chamfer_l2": cad_chamfer_l2[j],
            }
        )

    return metrics
