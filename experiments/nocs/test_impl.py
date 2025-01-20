import trimesh
import os
import time
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
import torchvision.transforms.functional as F


# local lib imports
from crisp.models.joint import JointShapePoseNetwork
import crisp.models.pipeline
from crisp.models.pipeline import Pipeline, get_reg_inlier_thres_from_nocs
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.certifier import FrameCertifier
from crisp.models.loss_functions import (
    nocs_loss,
    nocs_loss_clamped,
    siren_udf_loss,
    siren_sdf_fast_loss,
    metric_sdf_loss,
)
from crisp.models.registration import (
    umeyama_ransac_batched,
    umeyama_ransac_batched_inlier_thres_target,
    arun_ransac_batched,
)
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    gen_pyvista_voxel_slices,
    visualize_meshes_pyvista,
    visualize_sdf_slices_pyvista,
    visualize_meshes_pyvista,
    imgs_show,
)
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh
from crisp.utils.math import se3_inverse_batched_torch, make_se3_batched
from crisp.utils import diff_operators
from experiments.nocs.train_impl import run_pipeline


def test_model(
    fabric,
    model,
    dataloader,
    loss_fn,
    normalized_recons=True,
    vis_rgb_image=True,
    vis_sdf_sample_points=True,
    vis_pred_nocs_registration=True,
    vis_pred_nocs_and_cad=True,
    vis_pred_recons=True,
    vis_pred_sdf=True,
    vis_gt_sanity_test=True,
    vis_gt_nocs_heatmap=True,
    calculate_pred_recons_metrics=True,
    export_all_pred_recons_mesh=True,
    export_average_pred_recons_mesh=True,
    dump_vis=True,
    artifacts_save_dir="./artifacts",
):
    """Test the joint model"""
    safely_make_folders([artifacts_save_dir])
    model.eval()
    if vis_pred_nocs_and_cad:
        dataloader.dataset.data_to_output.extend(["normalized_mesh", "object_pc"])

    fabric.print("Testing...")
    all_metrics = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            (rgb_img, depth, segmap, gt_nocs, depth_pc, depth_pc_mask, object_pc, gt_obj_pose, coords, metadata) = (
                batch["rgb"],
                batch["depth"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["depth_pc"],
                batch["depth_pc_mask"],
                batch["object_pc"],
                batch["obj_pose"],
                batch["coords"],
                batch["metadata"],
            )
            B = rgb_img.shape[0]
            # the mask should be already for the correct object
            mask = segmap.unsqueeze(1)

            nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)

            # mask out non-object points
            nocs_map_flat = nocs_map.view(B, 3, -1).float()
            depth_pc_flat = depth_pc.view(B, 3, -1).float()
            mask_flat = torch.broadcast_to(
                torch.logical_and(depth_pc_mask, mask).view(B, 1, -1).bool(), depth_pc_flat.shape
            )
            masked_nocs, masked_depth_pcs = [], []
            for ii in range(B):
                masked_nocs.append(torch.unflatten(nocs_map_flat[ii, mask_flat[ii]], 0, (3, -1)))
                masked_depth_pcs.append(torch.unflatten(depth_pc_flat[ii, mask_flat[ii]], 0, (3, -1)))

            # est transformation
            est_poses_no_scale, est_poses_scale = [], []
            for ii in range(B):
                nocs_pts, depth_pts = masked_nocs[ii].unsqueeze(0), masked_depth_pcs[ii].unsqueeze(0)

                # get inlier threshold
                source_centroid = (torch.sum(nocs_pts, dim=2, keepdim=False) / nocs_pts.shape[2]).unsqueeze(2)
                centered_source = nocs_pts - source_centroid
                source_diameter = 2 * torch.amax(torch.linalg.norm(centered_source, dim=1), dim=1)
                reg_inlier_thres = 0.1 * source_diameter

                # no scale
                cam_R_nocs, cam_t_nocs, _, status = arun_ransac_batched(
                    nocs_pts, depth_pts, masks=None, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=500
                )
                est_poses_no_scale.append((1, cam_R_nocs[0, ...], cam_t_nocs[0, ...]))

                # scaled
                cam_s_nocs, cam_R_nocs, cam_t_nocs, _, status = umeyama_ransac_batched(
                    nocs_pts, depth_pts, masks=None, inlier_thres=reg_inlier_thres, confidence=0.99, max_iters=100
                )
                est_poses_scale.append((cam_s_nocs.item(), cam_R_nocs[0, ...], cam_t_nocs[0, ...]))

                # TODO: 3D IOU
                # TODO: Pose errors

            if dump_vis:
                print("dump vis")
                # save mesh
                for j in range(B):
                    shp = shape_code[j, ...]

                    def model_fn(coords):
                        return model.recons_net.forward(shape_code=shp.unsqueeze(0), coords=coords)

                    (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                        model_fn=model_fn,
                        N=64,
                        max_batch=64 ** 3,
                        cube_center=np.array([0, 0, 0]),
                        cube_scale=0.5,
                    )
                    pred_mesh = convert_sdf_samples_to_mesh(
                        sdf_grid=sdf_grid,
                        voxel_grid_origin=voxel_grid_origin,
                        voxel_size=voxel_size,
                        offset=None,
                        scale=None,
                    )
                    mesh_name = f"mesh_{metadata[j]['image_id']}_{metadata[j]['inst_id']}_{metadata[j]['obj_label']}.ply"
                    pred_mesh.export(os.path.join(artifacts_save_dir, mesh_name))
                    c_rgb_image = rgb_img[j, ...].cpu()
                    image_pil = F.to_pil_image(c_rgb_image)
                    rgb_name = f"rgb_{metadata[j]['image_id']}_{metadata[j]['inst_id']}_{metadata[j]['obj_label']}.jpg"
                    image_pil.save(os.path.join(artifacts_save_dir, rgb_name))

            if calculate_pred_recons_metrics:
                with torch.no_grad():
                    cf_l2s = []
                    for j in range(B):
                        shp = shape_code[j, ...]

                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shp.unsqueeze(0), coords=coords)

                        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                            model_fn=model_fn,
                            N=64,
                            max_batch=64**3,
                            cube_center=np.array([0, 0, 0]),
                            cube_scale=1,
                        )
                        pred_mesh = convert_sdf_samples_to_mesh(
                            sdf_grid=sdf_grid,
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=voxel_size,
                            offset=None,
                            scale=None,
                        )

                        sampled_mesh, _ = trimesh.sample.sample_surface(pred_mesh, 2500)
                        cf_l2, _ = chamfer_distance(
                            torch.tensor(sampled_mesh).unsqueeze(0).cuda().float(),
                            object_pc[j].unsqueeze(0),
                            point_reduction="mean",
                            batch_reduction=None,
                            norm=2,
                        )
                        cf_l2s.append(cf_l2.cpu().numpy().item())

            # update metrics
            for ii in range(B):
                all_metrics.append(
                    {
                        "gt_class_id": metadata[ii]["class_id"],
                        "gt_class_name": metadata[ii]["class_name"],
                        "gt_s": gt_obj_pose[ii][0],
                        "gt_R": gt_obj_pose[ii][1],
                        "gt_t": gt_obj_pose[ii][2],
                        "est_s": est_poses_scale[ii][0],
                        "est_R": est_poses_no_scale[ii][1],
                        "est_t": est_poses_no_scale[ii][2],
                        "cf_l2": cf_l2s[ii],
                    }
                )

            if vis_pred_nocs_registration:
                print(
                    "Visualize transformed CAD with depth points. "
                    "Est. transformed CAD: green, "
                    "Est. transformed NOCS: blue, "
                    "Depth: red"
                )
                for j in range(B):
                    assert not normalized_recons
                    # est transformed est nocs
                    noscale_cam_nocs_p = est_poses_no_scale[j][1] @ masked_nocs[j] + est_poses_no_scale[j][2]
                    scaled_cam_nocs_p = (
                        est_poses_scale[j][0] * est_poses_scale[j][1] @ masked_nocs[j] + est_poses_scale[j][2]
                    )

                    # gt transformed gt nocs
                    if normalized_recons:
                        nocs_pts = (gt_nocs[j, :3, ...] - 0.5) * 2
                    else:
                        nocs_pts = gt_nocs[j, :3, ...]
                    cad_p = nocs_pts.reshape(3, -1)

                    # some post processing ...
                    gt_obj_T = np.eye(4)
                    gt_obj_T[:3, :3] = gt_obj_pose[j][1]
                    gt_obj_T[:3, :3] *= gt_obj_pose[j][0]
                    gt_obj_T[:3, 3] = gt_obj_pose[j][2]
                    z_180_RT = np.zeros((4, 4), dtype=np.float32)
                    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                    z_180_RT[3, 3] = 1
                    gt_obj_T = z_180_RT @ gt_obj_T

                    gt_cam_nocs = gt_obj_T[:3, :3] @ cad_p.cpu().numpy() + gt_obj_T[:3, 3].reshape((3, 1))

                    visualize_pcs_pyvista(
                        [noscale_cam_nocs_p, scaled_cam_nocs_p, masked_depth_pcs[j], gt_cam_nocs],
                        colors=["green", "blue", "crimson", "black"],
                        pt_sizes=[5.0, 5.0, 5.0, 5.0],
                    )

            if i % 500 == 0:
                np.save(os.path.join(artifacts_save_dir, "metrics.npy"), all_metrics, allow_pickle=True)

            if vis_pred_nocs_and_cad:
                print("Visualize sampled coords and mesh.")
                normalized_mesh, object_pc = batch["normalized_mesh"], batch["object_pc"]
                for j in range(rgb_img.shape[0]):
                    visualize_meshes_pyvista(
                        [object_pc[j, ...], normalized_mesh[j][0]],
                        plotter_shape=(1, 1),
                        mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                    )

                print("Visualize NOCS and object mesh points. Note: we assume unnormalized NOCS.")
                # TODO: Fix scale bug here
                for j in range(rgb_img.shape[0]):
                    nocs_pts = (gt_nocs[j, ...] * mask[j, ...]).detach().cpu().numpy().reshape((3, -1))
                    visualize_meshes_pyvista(
                        [object_pc[j, ...], nocs_pts],
                        plotter_shape=(1, 1),
                        mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                    )

                    print("Visualize GT NOCS and est NOCS.")
                    visualize_meshes_pyvista(
                        [masked_nocs[j], nocs_pts],
                        plotter_shape=(1, 1),
                        mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                    )

            if vis_pred_nocs_and_cad:
                print("Visualizing model predicted NOCS with object points.")
                for j in range(B):
                    print("Masked depth pc (black)")
                    visualize_pcs_pyvista(
                        [masked_depth_pcs[j]],
                        colors=["lightblue"],
                        pt_sizes=[10.0],
                    )

                    print("Masked nocs (lightblue)")
                    visualize_pcs_pyvista(
                        [masked_nocs[j]],
                        colors=["lightblue"],
                        pt_sizes=[10.0],
                    )

            if vis_rgb_image:
                print("Visualize RGB and mask...")
                imgs_show(rgb_img)
                imgs_show(mask.int())

            if vis_pred_sdf:
                print("Visualize SDF slices...")
                with torch.no_grad():
                    for j in range(B):
                        shp = shape_code[j, ...]

                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shp.unsqueeze(0), coords=coords)

                        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                            model_fn=model_fn,
                            N=128,
                            max_batch=64**3,
                            cube_center=np.array([0, 0, 0]),
                            cube_scale=1,
                        )
                        pred_mesh = convert_sdf_samples_to_mesh(
                            sdf_grid=sdf_grid,
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=voxel_size,
                            offset=None,
                            scale=None,
                        )

                        nocs_pts = (gt_nocs[j, ...] * mask[j, ...]).detach().cpu().numpy().reshape((3, -1))
                        visualize_meshes_pyvista(
                            [pred_mesh, nocs_pts],
                            mesh_args=[{"opacity": 0.2, "color": "white"}, {"opacity": 0.2, "color": "red"}],
                        )

    np.save(os.path.join(artifacts_save_dir, "metrics.npy"), all_metrics, allow_pickle=True)

    return all_metrics


def test_pipeline(
    fabric,
    pipeline,
    test_dataloader,
    num_epochs,
    visualize=False,
    save_every_epoch=10,
    model_save_path="./model_ckpts",
    artifacts_save_dir="./artifacts",
    calculate_test_recons_metrics=True,
):
    all_metrics = run_pipeline(
        fabric=fabric,
        pipeline=pipeline,
        optimizer=None,
        train_dataloader=test_dataloader,
        num_epochs=num_epochs,
        visualize=visualize,
        save_every_epoch=save_every_epoch,
        model_save_path=model_save_path,
        ssl_train=False,
        calculate_test_metrics=True,
        calculate_test_recons_metrics=calculate_test_recons_metrics,
        artifacts_save_dir=artifacts_save_dir,
    )

    np.save(os.path.join(artifacts_save_dir, "metrics.npy"), all_metrics, allow_pickle=True)
