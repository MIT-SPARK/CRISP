import torch
import numpy as np
import pyvista as pv
import open3d as o3d

from crisp.datasets.unified_objects import cam2cad_transformation_from
from crisp.utils.evaluation_metrics import rotation_error, add_s_error, chamfer_dist
from crisp.utils.math import instance_depth_to_point_cloud_torch, make_se3, sq_half_chamfer_dists
from crisp.utils.geometry import sample_pts_and_normals

from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    gen_pyvista_voxel_slices,
    visualize_meshes_pyvista,
    visualize_sdf_slices_pyvista,
)


def error_metrics(
    gt_objs_data,
    depths,
    masks,
    intrinsics,
    objects_info,
    object_pc_ds,
    payload,
    model,
    calculate_pred_recons_metrics=False,
    cube_scale=None,
    vis_pcs=False,
    use_ICP=False,
    normalized_recons=True,
    vis_transformed_cad=False,
):
    """Create coord"""
    if cube_scale is None:
        cube_scale = 1.0

    """Compute relevant error metrics"""
    with torch.no_grad():
        results = []
        B = payload["cam_s_nocs"].shape[0]
        assert B == len(payload["frame_index"])
        print(f"Cert mask: {payload['cert_mask']}")
        for i in range(B):
            fi, oi = payload["frame_index"][i], payload["obj_index"][i]
            depths_pts, _ = instance_depth_to_point_cloud_torch(
                depths[fi, 0, ...],
                intrinsics[fi, 0, ...],
                instance_mask=masks[fi, 0, ...] == gt_objs_data[fi][oi]["id_in_segm"],
            )

            device = payload["cam_R_nocs"].device

            # gt pose and scale
            obj_label = gt_objs_data[fi][oi]["label"]
            cam_T_gt_cad = torch.tensor(gt_objs_data[fi][oi]["TCO"]).to(device).float()
            recons_s_blender, recons_t_blender, blender_R_cad, blender_s_cad = (
                torch.tensor(objects_info[obj_label]["recons_s_blender"]).to(device),
                torch.tensor(objects_info[obj_label]["recons_t_blender"]).to(device),
                torch.tensor(objects_info[obj_label]["blender_R_cad"]).to(device),
                torch.tensor(objects_info[obj_label]["blender_s_cad"]).to(device),
            )
            assert torch.isclose(blender_s_cad[0], blender_s_cad[1]) and torch.isclose(
                blender_s_cad[0], blender_s_cad[1]
            )

            # estimated R & t
            # recons_T_blender = make_se3(torch.eye(3).to(device), recons_t_blender)
            # blender_T_cad = make_se3(blender_R_cad, torch.zeros(3).to(device))
            # recons_T_cad = recons_T_blender @ blender_T_cad
            cam_s_est_recons = payload["cam_s_nocs"][i, ...]
            cam_R_est_recons = payload["cam_R_nocs"][i, ...]
            cam_t_est_recons = payload["cam_t_nocs"][i, ...]
            cam_s_est_cad, cam_R_est_cad, cam_t_est_cad = cam2cad_transformation_from(
                cam_s_est_recons,
                cam_R_est_recons,
                cam_t_est_recons,
                recons_s_blender,
                recons_t_blender,
                blender_R_cad,
                torch.mean(blender_s_cad),
            )

            # estimated scale
            # note:
            # cam_s_est_cad gt = 1 for correct estimate
            cam_s_est_recons = payload["cam_s_nocs"][i, ...]
            cam_s_est_blender = cam_s_est_recons * recons_s_blender

            # ADD-S scores (gt transformed CAD vs. est transformed CAD)
            o_mesh = torch.tensor(object_pc_ds.get_pc(obj_label)).to(device).float()

            # from cad to recons frame
            recons_cadpc_gt = torch.diag(blender_s_cad) @ blender_R_cad @ o_mesh
            recons_cadpc_gt = recons_s_blender * recons_cadpc_gt + recons_t_blender.reshape(3, 1)
            if not normalized_recons:
                recons_cadpc_gt = recons_cadpc_gt / recons_s_blender

            # CAD model transfromed to camera frame with estimated transformation
            cam_cadpc_est = cam_s_est_recons * cam_R_est_recons @ recons_cadpc_gt + cam_t_est_recons.reshape(3, 1)

            # CAD model transformed to camera frame with groundtruth transformation
            cam_cadpc_gt = torch.mean(blender_s_cad) * cam_T_gt_cad[:3, :3] @ o_mesh + cam_T_gt_cad[:3, -1].reshape(
                3, 1
            )

            # pre & post corrector NOCS (checking the amount of correction on NOCS)
            # pre crt metrics
            cam_precrtnocs_est = None
            adds_precrt_nocs_depths = None
            if "precrt_nocs" in payload.keys():
                cam_precrtnocs_est = cam_s_est_recons * cam_R_est_recons @ payload["precrt_nocs"].detach()[
                    i, ...
                ] + cam_t_est_recons.reshape(3, 1)
                adds_precrt_nocs_depths = torch.sqrt(
                    sq_half_chamfer_dists(cam_precrtnocs_est.unsqueeze(0), depths_pts.unsqueeze(0)).detach().cpu()
                )

            # post crt metrics
            cam_postcrtnocs_est = None
            adds_postcrt_nocs_depths = None
            if "postcrt_nocs" in payload.keys():
                cam_postcrtnocs_est = cam_s_est_recons * cam_R_est_recons @ payload["postcrt_nocs"].detach()[
                    i, ...
                ] + cam_t_est_recons.reshape(3, 1)
                adds_postcrt_nocs_depths = torch.sqrt(
                    sq_half_chamfer_dists(cam_postcrtnocs_est.unsqueeze(0), depths_pts.unsqueeze(0)).detach().cpu()
                )

            # CAD model ADDS
            adds = (
                torch.sqrt(sq_half_chamfer_dists(cam_cadpc_gt.unsqueeze(0), cam_cadpc_est.unsqueeze(0))).detach().cpu()
            )

            if vis_transformed_cad:
                print(f"Object label: {obj_label}.")
                print("Visualizing CAD transformed to NOCS frame (precrt - red, postcrt - blue)")
                visualize_pcs_pyvista(
                    [
                        payload["precrt_nocs"][i, ...].detach(),
                        payload["postcrt_nocs"][i, ...].detach(),
                        recons_cadpc_gt.detach(),
                    ],
                    colors=["crimson", "blue", "black"],
                    pt_sizes=[5.0, 5.0, 5.0],
                    bg_color="white",
                    show_axes=False,
                )
                print("Visualizing est transformed precrt NOCS (red), postcrt NOCS (blue) and depths (black)")
                visualize_pcs_pyvista(
                    [cam_precrtnocs_est.detach(), cam_postcrtnocs_est.detach(), depths_pts],
                    colors=["crimson", "blue", "black"],
                    pt_sizes=[5.0, 5.0, 5.0],
                )
                print("Visualizing GT transformed CAD pc, est transformed CAD pc and depths")
                visualize_pcs_pyvista(
                    [cam_cadpc_est.detach(), cam_cadpc_gt.detach(), depths_pts],
                    colors=["crimson", "blue", "black"],
                    pt_sizes=[5.0, 5.0, 5.0],
                    bg_color="white",
                    show_axes=False,
                )

            # error
            R_err = rotation_error(cam_R_est_cad, cam_T_gt_cad[:3, :3]).cpu().item()
            t_err = torch.linalg.norm(cam_t_est_cad.flatten() - cam_T_gt_cad[:3, -1]).cpu().item()
            if normalized_recons:
                s_err = (cam_s_est_blender - 1).cpu().item()
            else:
                s_err = 0

            metrics = {
                "obj_label": obj_label,
                "R_err": R_err,
                "t_err": t_err,
                "s_err": s_err,
                # CAD-CAD ADDS
                "adds_mean": torch.mean(adds).item(),
                "adds_std": torch.std(adds).item(),
                "adds_median": torch.median(adds).item(),
                "cert_flag": payload["cert_mask"][i],
                "cert_percent": (payload["cert_mask"].sum() / payload["cert_mask"].shape[0]).cpu().item(),
            }

            if adds_precrt_nocs_depths is not None:
                metrics.update(
                    {
                        # NOCS-DEPTHS ADDS
                        "adds_nocs_depths_precrt_mean": torch.mean(adds_precrt_nocs_depths).item(),
                        "adds_nocs_depths_precrt_std": torch.std(adds_precrt_nocs_depths).item(),
                        "adds_nocs_depths_precrt_median": torch.median(adds_precrt_nocs_depths).item(),
                    }
                )

            if adds_postcrt_nocs_depths is not None:
                metrics.update(
                    {
                        # NOCS-DEPTHS ADDS
                        "adds_nocs_depths_postcrt_mean": torch.mean(adds_postcrt_nocs_depths).item(),
                        "adds_nocs_depths_postcrt_std": torch.std(adds_postcrt_nocs_depths).item(),
                        "adds_nocs_depths_postcrt_median": torch.median(adds_postcrt_nocs_depths).item(),
                    }
                )

            if "precrt_cert_mask" in payload.keys():
                metrics["precrt_cert_percent"] = (
                    (payload["precrt_cert_mask"].sum() / payload["precrt_cert_mask"].shape[0]).cpu().item()
                )

            if calculate_pred_recons_metrics:
                shape_code = payload["shape_code"][i, ...].unsqueeze(0)

                def model_fn(coords):
                    return model.recons_net.forward(shape_code=shape_code, coords=coords)

                (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                    model_fn=model_fn,
                    N=128,
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

                shape_gt = recons_cadpc_gt.detach().cpu().numpy().T
                shape_est, _ = sample_pts_and_normals(pred_mesh, shape_gt.shape[0], interp=False)

                d1, p1 = fpd(shape_gt)
                d2, p2 = fpd(shape_est)

                shape_gt = shape_gt / d1
                shape_est = shape_est / d2

                if use_ICP:
                    min_cd = 100000000
                    min_shape_est = None

                    shape_gt_o3d = o3d.geometry.PointCloud()
                    shape_gt_o3d.points = o3d.utility.Vector3dVector(shape_gt)
                    shape_est_o3d = o3d.geometry.PointCloud()
                    shape_est_o3d.points = o3d.utility.Vector3dVector(shape_est)

                    for angle in range(4):
                        initial_angle = angle * np.pi / 2
                        initial_pose = np.array(
                            [
                                [np.cos(initial_angle), -np.sin(initial_angle), 0, 0],
                                [np.sin(initial_angle), np.cos(initial_angle), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                        )

                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            shape_est_o3d,  # source
                            shape_gt_o3d,  # target
                            0.01,  # threshold
                            initial_pose,  # initial transformation
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )
                        icp_R = reg_p2p.transformation[:3, :3]
                        icp_t = reg_p2p.transformation[:3, 3]

                        temp_shape_est = shape_est @ icp_R.T + icp_t

                        cd = chamfer_dist(
                            torch.transpose(torch.tensor(shape_gt.astype("float32")), 0, 1).unsqueeze(0),
                            torch.transpose(torch.tensor(temp_shape_est.astype("float32")), 0, 1).unsqueeze(0),
                        )

                        if cd < min_cd:
                            min_cd = cd
                            min_shape_est = temp_shape_est

                    cd = min_cd
                    shape_est = min_shape_est

                else:
                    cd = chamfer_dist(
                        torch.transpose(torch.tensor(shape_gt.astype("float32")), 0, 1).unsqueeze(0),
                        torch.transpose(torch.tensor(shape_est.astype("float32")), 0, 1).unsqueeze(0),
                    )

                if vis_pcs:
                    print("Visualize generate shape (blue) and gt shape (red)")
                    visualize_pcs_pyvista(
                        [shape_gt, shape_est],
                        colors=["red", "blue"],
                        pt_sizes=[5.0, 5.0],
                    )
                    print(cd)

                metrics.update({"shape_chamfer_distance": cd.item()})

            results.append(metrics)

    return results


def shape_metrics(
    pipeline,
    gt_objs_data,
    depths,
    masks,
    intrinsics,
    objects_info,
    object_pc_ds,
    payload,
    normalized_recons=True,
    vis_transformed_cad=False,
    N=256,
    max_batch=64**3,
    cube_center=None,
    cube_scale=None,
    visualize_pcs=False,
):
    if cube_center is None:
        cube_center = np.array([0, 0, 0])
    if cube_scale is None:
        cube_scale = 2.5

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner of the grid, not the middle
    voxel_origin = cube_center - cube_scale / 2
    voxel_size = cube_scale / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    # first 3 columns: coordinates; last column: SDF values
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    with torch.no_grad():
        results = []
        B = payload["cam_s_nocs"].shape[0]
        assert B == len(payload["frame_index"])
        for i in range(B):
            shape_code = payload["shape_code"][i, ...].unsqueeze(0)

            head = 0
            while head < num_samples:
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].unsqueeze(0).cuda()
                samples[head : min(head + max_batch, num_samples), 3] = (
                    pipeline.model.recons_net(shape_code, coords=sample_subset).squeeze().detach().cpu()  # .squeeze(1)
                )
                head += max_batch

            sdf_values = samples[:, 3]
            sdf_values = sdf_values.reshape(N, N, N)

            pred_mesh = convert_sdf_samples_to_mesh(
                sdf_grid=sdf_values,
                voxel_grid_origin=voxel_origin,
                voxel_size=voxel_size,
                offset=None,
                scale=None,
            )

            device = payload["cam_R_nocs"].device
            fi, oi = payload["frame_index"][i], payload["obj_index"][i]
            # gt pose and scale
            obj_label = gt_objs_data[fi][oi]["label"]

            recons_s_blender, recons_t_blender, blender_R_cad, blender_s_cad = (
                torch.tensor(objects_info[obj_label]["recons_s_blender"]).to(device),
                torch.tensor(objects_info[obj_label]["recons_t_blender"]).to(device),
                torch.tensor(objects_info[obj_label]["blender_R_cad"]).to(device),
                torch.tensor(objects_info[obj_label]["blender_s_cad"]).to(device),
            )
            assert torch.isclose(blender_s_cad[0], blender_s_cad[1]) and torch.isclose(
                blender_s_cad[0], blender_s_cad[1]
            )

            o_mesh = torch.tensor(object_pc_ds.get_pc(obj_label)).to(device).float()

            # from cad to recons frame
            recons_cadpc_gt = torch.diag(blender_s_cad) @ blender_R_cad @ o_mesh
            recons_cadpc_gt = recons_s_blender * recons_cadpc_gt + recons_t_blender.reshape(3, 1)
            if not normalized_recons:
                recons_cadpc_gt = recons_cadpc_gt / recons_s_blender

            shape_gt = recons_cadpc_gt.detach().cpu().numpy().T
            # shape_est = pred_mesh.vertices
            # rand_idx = np.random.choice(shape_est.shape[0], size = shape_gt.shape[0], replace=False)
            # shape_est = shape_est[rand_idx, :]
            shape_est, _ = sample_pts_and_normals(pred_mesh, shape_gt.shape[0], interp=False)

            d1, p1 = fpd(shape_gt)
            d2, p2 = fpd(shape_est)

            shape_gt = shape_gt / d1
            shape_est = shape_est / d2

            shape_gt_o3d = o3d.geometry.PointCloud()
            shape_gt_o3d.points = o3d.utility.Vector3dVector(shape_gt)
            shape_est_o3d = o3d.geometry.PointCloud()
            shape_est_o3d.points = o3d.utility.Vector3dVector(shape_est)

            reg_p2p = o3d.pipelines.registration.registration_icp(
                shape_est_o3d,  # source
                shape_gt_o3d,  # target
                1,  # threshold
                np.diag(np.ones(4)),  # initial transformation
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            icp_R = reg_p2p.transformation[:3, :3]
            icp_t = reg_p2p.transformation[:3, 3]

            shape_est = shape_est @ icp_R.T + icp_t

            cd = chamfer_dist(
                torch.transpose(torch.tensor(shape_gt.astype("float32")), 0, 1).unsqueeze(0),
                torch.transpose(torch.tensor(shape_est.astype("float32")), 0, 1).unsqueeze(0),
            )
            results += [cd]
            if visualize_pcs:
                visualize_pcs_pyvista(
                    [shape_gt, shape_est],
                    colors=["red", "blue"],
                    pt_sizes=[5.0, 5.0],
                )
                print(cd)

    return results


def fpd(pc):
    # pc: N x 3
    max_d = 0
    N = pc.shape[0]
    sample_N = int(np.sqrt(N))
    rand_idx = np.random.choice(N, size=sample_N, replace=False)

    furthest_pair = (None, None)

    for i in range(sample_N):
        for j in range(i + 1, sample_N):
            idx = rand_idx[i]
            jdx = rand_idx[j]

            d = np.linalg.norm(pc[idx, :] - pc[jdx, :])

            if d > max_d:
                max_d = d
                furthest_pair = (pc[idx, :], pc[jdx, :])

    return max_d, furthest_pair
