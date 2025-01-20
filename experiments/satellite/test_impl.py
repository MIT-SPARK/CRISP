import os
from collections import defaultdict

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import trimesh

import crisp.utils.sdf
from experiments.satellite.spe3r_utils import l1_l2_chamfer_dists, calculate_metrics
from experiments.unified_model.dataset_checks import single_batch_sanity_test
from experiments.satellite.train_impl import run_pipeline
from crisp.models.registration import align_nocs_to_depth, align_pts_to_depth_no_scale
from crisp.utils.evaluation_metrics import translation_error, rotation_error
from crisp.utils.file_utils import safely_make_folders
from crisp.utils.math import se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.sdf import create_sdf_samples_generic
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    gen_pyvista_voxel_slices,
    visualize_meshes_pyvista,
)


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
    dump_video=False,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    all_metrics = run_pipeline(
        fabric=fabric,
        pipeline=pipeline,
        optimizers=None,
        train_dataloader=test_dataloader,
        num_epochs=num_epochs,
        visualize=visualize,
        save_every_epoch=save_every_epoch,
        model_save_path=model_save_path,
        ssl_train=False,
        calculate_test_metrics=True,
        calculate_test_recons_metrics=calculate_test_recons_metrics,
        artifacts_save_dir=artifacts_save_dir,
        dump_video=dump_video,
    )

    np.save(os.path.join(artifacts_save_dir, "metrics.npy"), all_metrics, allow_pickle=True)


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
    artifacts_save_dir="./artifacts",
):
    """Test the following:
    1. Pose errors from NOCS registration
    2. Reconstruction error (deviation from 0 set)
    3. Visualize quality of shape reconstruction

    Set calculate_pred_recons_metrics=True to run marching cubes and generate predicted shapes
    """
    safely_make_folders([artifacts_save_dir])
    model.eval()
    count = 0
    total_rot_err, total_trans_err = 0, 0
    obj2shpcode = defaultdict(list)
    objcounter = defaultdict(lambda: 0)

    # data to save for comparing with baselines
    metrics = []

    fabric.print("Testing...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            (rgb_img, depth, segmap, gt_nocs, object_pc, cam_intrinsics, gt_world_T_cam, coords) = (
                batch["rgb"],
                batch["depth"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["object_pc"],
                batch["cam_intrinsics"],
                batch["cam_pose"],
                batch["coords"],
            )
            device = rgb_img.device
            bs = rgb_img.shape[0]
            mask = (segmap == 1).unsqueeze(1)
            nocs_map, sdf, batched_shape_code = model(img=rgb_img, mask=mask, coords=coords)

            # depth to point clouds
            H, W = rgb_img.shape[-2:]
            depth_map_x_indices = torch.arange(W).to(device)
            depth_map_y_indices = torch.arange(H).to(device)
            depth_map_grid_x, depth_map_grid_y = torch.meshgrid(depth_map_x_indices, depth_map_y_indices, indexing="xy")
            # pcs = depth_to_point_cloud_map_batched(
            #    depth, cam_intrinsics, grid_x=depth_map_grid_x, grid_y=depth_map_grid_y
            # )

            if vis_rgb_image:
                for j in range(bs):
                    rgb_img_cpu = rgb_img[j, ...].cpu()
                    rgb_img_cpu[0, ...] = rgb_img_cpu[0, ...] * 0.229 + 0.485
                    rgb_img_cpu[1, ...] = rgb_img_cpu[1, ...] * 0.224 + 0.456
                    rgb_img_cpu[2, ...] = rgb_img_cpu[2, ...] * 0.225 + 0.406
                    plt.imshow(rgb_img_cpu.permute(1, 2, 0).numpy())
                    plt.show()

            if vis_gt_nocs_heatmap:
                for j in range(bs):
                    nocs_pts = (gt_nocs[j, :3, ...] - 0.5) * 2
                    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
                    rgb_img_cpu = rgb_img[j, ...].cpu()
                    rgb_img_cpu[0, ...] = rgb_img_cpu[0, ...] * 0.229 + 0.485
                    rgb_img_cpu[1, ...] = rgb_img_cpu[1, ...] * 0.224 + 0.456
                    rgb_img_cpu[2, ...] = rgb_img_cpu[2, ...] * 0.225 + 0.406
                    axs[0, 0].imshow(rgb_img_cpu.permute(1, 2, 0).numpy())
                    axs[0, 0].set_title("image")
                    seaborn.heatmap(
                        data=nocs_pts[0].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[0, 1],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[0, 1].set_title("x heatmap")
                    seaborn.heatmap(
                        data=nocs_pts[1].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[1, 0],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[1, 0].set_title("y heatmap")
                    seaborn.heatmap(
                        data=nocs_pts[2].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[1, 1],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[1, 1].set_title("z heatmap")
                    plt.show()

            if vis_sdf_sample_points:
                visualize_pcs_pyvista(
                    [coords[0, ...], object_pc[0, ...]], colors=["crimson", "lightblue"], pt_sizes=[6.0, 2.0]
                )

            if vis_gt_sanity_test:
                voxel_res = dataloader.dataset.unified_objects.global_nonmnfld_points_voxel_res
                cube_scale = (
                    dataloader.dataset.unified_objects.sample_bounds[1]
                    - dataloader.dataset.unified_objects.sample_bounds[0]
                )
                voxel_origin = np.array([0, 0, 0]) - cube_scale / 2
                voxel_size = cube_scale / (voxel_res - 1)
                single_batch_sanity_test(
                    mask=mask.squeeze(1),
                    gt_nocs=gt_nocs,
                    sdf_grid=batch["sdf_grid"],
                    object_pc=object_pc,
                    normalized_mesh=batch["normalized_mesh"],
                    depth=depth.squeeze(1),
                    cam_intrinsics=cam_intrinsics,
                    gt_world_T_cam=gt_world_T_cam,
                    voxel_res=voxel_res,
                    voxel_origin=voxel_origin,
                    voxel_size=voxel_size,
                    normalized_recons=normalized_recons,
                    vis_gt_nocs_registration=True,
                    vis_gt_nocs_and_cad=True,
                    vis_sdf_and_mesh=True,
                )

            if normalized_recons:
                # nocs pose error
                s, cam_R_world, cam_t_cad, cam_Tsim_cad, _, _ = align_nocs_to_depth(
                    masks=mask,
                    nocs=nocs_map,
                    depth=depth.squeeze(1),
                    intrinsics=cam_intrinsics,
                    instance_ids=torch.arange(bs),
                    normalized_nocs=normalized_recons,
                    img_path=None,
                    verbose=False,
                )
            else:
                # unnormalized nocs
                cam_R_world, cam_t_cad, cam_Tsim_cad, _, _ = align_pts_to_depth_no_scale(
                    masks=mask,
                    pts=nocs_map,
                    depth=depth.squeeze(1),
                    intrinsics=cam_intrinsics,
                    instance_ids=torch.arange(bs),
                    inlier_thres=torch.tensor(0.1).to(nocs_map.device),
                    img_path=None,
                    verbose=False,
                )
                s = 1.0

            gt_cam_T_world = se3_inverse_batched_torch(gt_world_T_cam)
            trans_err = translation_error(cam_t_cad.unsqueeze(-1), gt_cam_T_world[:, :3, -1].unsqueeze(-1))
            rot_err = rotation_error(cam_R_world, gt_cam_T_world[..., :3, :3])
            # print(f"Pred. NOCS registration avg/median trans err: {torch.mean(trans_err)}/{torch.median(trans_err)}")
            # print(
            #    f"Pred. NOCS registration avg/median rotation err (rad): {torch.mean(rot_err)}/{torch.median(rot_err)}"
            # )
            # print(f"Pred. NOCS scale: {s}")

            # transformed things
            est_cam_cad = cam_Tsim_cad[:, :3, :3] @ torch.transpose(object_pc, 1, 2) + cam_Tsim_cad[:, :3, 3].reshape(
                -1, 3, 1
            )
            gt_cam_cad = gt_cam_T_world[:, :3, :3] @ torch.transpose(object_pc, 1, 2) + gt_cam_T_world[
                :, :3, 3
            ].reshape(-1, 3, 1)

            if vis_pred_nocs_and_cad:
                print("Visualizing model predicted NOCS with object points.")
                for j in range(bs):
                    # sanity checks & tests for each instance in batch
                    # retrieve original CAD points
                    # compare the original CAD coords and NOCS points
                    # cad points from NOCS
                    if normalized_recons:
                        nocs_pts = (nocs_map[j, :3, ...] - 0.5) * 2
                    else:
                        nocs_pts = nocs_map[j, :3, ...]
                    mask_j = torch.broadcast_to(mask[j], nocs_pts.shape)
                    nocs_pts_in = nocs_pts[mask_j].reshape(3, -1)  # in segmentation mask
                    nocs_pts_out = nocs_pts[~mask_j].reshape(3, -1)

                    # visualize NOCS and object mesh points
                    visualize_pcs_pyvista(
                        [nocs_pts_in, nocs_pts_out, object_pc[j, ...]],
                        colors=["lightblue", "red", "black"],
                        pt_sizes=[10.0, 10.0, 2.0],
                    )
                    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

                    rgb_img_cpu = rgb_img[j, ...].cpu()
                    rgb_img_cpu[0, ...] = rgb_img_cpu[0, ...] * 0.229 + 0.485
                    rgb_img_cpu[1, ...] = rgb_img_cpu[1, ...] * 0.224 + 0.456
                    rgb_img_cpu[2, ...] = rgb_img_cpu[2, ...] * 0.225 + 0.406
                    axs[0, 0].imshow(rgb_img_cpu.permute(1, 2, 0).numpy())
                    axs[0, 0].set_title("image")
                    seaborn.heatmap(
                        data=nocs_pts[0].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[0, 1],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[0, 1].set_title("x heatmap")
                    seaborn.heatmap(
                        data=nocs_pts[1].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[1, 0],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[1, 0].set_title("y heatmap")
                    seaborn.heatmap(
                        data=nocs_pts[2].cpu().numpy(),
                        cbar_kws={"pad": 0.02},
                        ax=axs[1, 1],
                        square=True,
                        xticklabels=False,
                        yticklabels=False,
                    )
                    axs[1, 1].set_title("z heatmap")
                    plt.show()

            if vis_pred_nocs_registration:
                print("Visualizing model predicted NOCS registration result.")
                for j in range(bs):
                    print(
                        "Visualize transformed CAD with depth points. "
                        "Est. transformed CAD: green, "
                        "Est. transformed NOCS: blue, "
                        "Depth: red"
                    )
                    depth_pts, idxs = instance_depth_to_point_cloud_torch(
                        depth[j, ...].squeeze(0), cam_intrinsics[j, ...], mask[j, ...].squeeze(0)
                    )
                    # NOCS transformed to depth
                    if normalized_recons:
                        nocs_pts = (nocs_map[j, :3, idxs[0], idxs[1]] - 0.5) * 2
                    else:
                        nocs_pts = nocs_map[j, :3, idxs[0], idxs[1]]
                    nocs_p = nocs_pts.reshape(3, -1)
                    cam_nocs_p = cam_Tsim_cad[j, :3, :3] @ nocs_p + cam_Tsim_cad[j, :3, 3].reshape(3, 1)

                    visualize_pcs_pyvista(
                        [est_cam_cad[j, ...], cam_nocs_p, gt_cam_cad[j, ...], depth_pts],
                        colors=["green", "lightblue", "yellow", "crimson"],
                        pt_sizes=[2.0, 10.0, 2.0, 2.0],
                    )

            # shape reconstruction
            shp_inf_flag = (
                vis_pred_recons or vis_pred_sdf or export_average_pred_recons_mesh or export_all_pred_recons_mesh
            )
            if shp_inf_flag:
                model.eval()
                for j in range(bs):
                    shape_code = batched_shape_code[0, ...]
                    if vis_pred_recons or vis_pred_sdf or export_all_pred_recons_mesh:

                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shape_code.unsqueeze(0), coords=coords)

                        cube_scale = (
                            dataloader.dataset.unified_objects.sample_bounds[1]
                            - dataloader.dataset.unified_objects.sample_bounds[0]
                        )
                        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                            model_fn=model_fn,
                            N=256,
                            max_batch=64**3,
                            cube_center=np.array([0, 0, 0]),
                            cube_scale=cube_scale,
                        )

                        pred_mesh = crisp.utils.sdf.convert_sdf_samples_to_mesh(
                            sdf_grid=sdf_grid,
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=voxel_size,
                            offset=None,
                            scale=None,
                        )

                        if vis_pred_sdf:
                            print("Visualize SDF field")
                            gt_mesh = trimesh.Trimesh(
                                vertices=batch["normalized_mesh"][j][0].numpy(force=True),
                                faces=batch["normalized_mesh"][j][1].numpy(force=True),
                            )

                            mesh_sdf_slices = gen_pyvista_voxel_slices(
                                sdf_grid.numpy(force=True), voxel_grid_origin, (voxel_size,) * 3
                            )
                            visualize_meshes_pyvista(
                                [pred_mesh, gt_mesh, mesh_sdf_slices],
                                mesh_args=[{"opacity": 0.2, "color": "white"}, {"opacity": 0.4, "color": "red"}, None],
                            )

            c_metrics = calculate_metrics(
                model=model,
                gt_cam_T_world=gt_cam_T_world,
                est_cam_T_world=cam_Tsim_cad,
                gt_shape_names=batch["model_name"],
                batched_shape_code=batched_shape_code,
                shape_bounds=dataloader.dataset.unified_objects.sample_bounds,
                object_pc=object_pc,
                calculate_pred_recons_metrics=calculate_pred_recons_metrics,
            )
            metrics.extend(c_metrics)

            # save test metrics
            if i % 500 == 0:
                np.save(os.path.join(artifacts_save_dir, "metrics.npy"), metrics, allow_pickle=True)

            total_rot_err += torch.sum(rot_err)
            total_trans_err += torch.sum(trans_err)
            count += bs
            torch.cuda.empty_cache()

    # final metrics
    np.save(os.path.join(artifacts_save_dir, "metrics.npy"), metrics, allow_pickle=True)

    total_rot_err /= count
    total_trans_err /= count
    fabric.print(f"Test rot/trans loss across {count} instances:: {total_rot_err}/{total_trans_err}")
    fabric.log("Test/total_rot_err", total_rot_err)
    fabric.log("Test/total_trans_err", total_trans_err)

    if export_average_pred_recons_mesh:
        print("Calculating average shape.")
        for l, lcs in obj2shpcode.items():
            mesh_folder = os.path.join(artifacts_save_dir, "meshes", f"{l}")
            safely_make_folders([mesh_folder])
            mesh_fname = os.path.join(mesh_folder, f"avg_{l}.ply")
            avg_lc = torch.stack(lcs).mean(axis=0).to(rgb_img.device)
            with torch.no_grad():

                def model_fn(coords):
                    return model.recons_net.forward(shape_code=torch.tensor(avg_lc).unsqueeze(0), coords=coords)

                sdf_grid, voxel_size, voxel_grid_origin = create_sdf_samples_generic(
                    model_fn=model_fn, N=256, max_batch=64**3, cube_center=np.array([0, 0, 0]), cube_scale=2.5
                )
                pred_mesh = crisp.utils.sdf.convert_sdf_samples_to_mesh(
                    sdf_grid=sdf_grid,
                    voxel_grid_origin=voxel_grid_origin,
                    voxel_size=voxel_size,
                    offset=None,
                    scale=None,
                )
                pred_mesh.export(mesh_fname)
