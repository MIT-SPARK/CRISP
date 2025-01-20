import gc
import os
import dataclasses
from functools import partial
import numpy as np
import torch
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser
import lightning as L
import lightning.fabric.strategies as LFS
import time
import lightning.fabric as LF
import datetime
from tqdm import tqdm
import torch.utils.data as tchdata
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pyvista as pv
import seaborn

# local lib imports
from crisp.datasets.unified_objects import (
    UnifiedObjects,
    unified_objects_collate_fn,
    ObjClassBatchSampler,
    DistributedObjClassBatchSampler,
)
from crisp.models.joint import JointShapePoseNetwork, apply_lipschitz_constraint
from crisp.models.registration import align_nocs_to_depth
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh
from crisp.models.nocs import *
from crisp.models.loss_functions import (
    nocs_loss,
    siren_udf_loss,
    siren_sdf_fast_loss,
    metric_sdf_loss,
    metric_sdf_loss_no_gradient,
    two_cls_contrastive_explicit_loss_helper,
)
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.math import se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    gen_pyvista_voxel_slices,
    visualize_meshes_pyvista,
    imgs_show,
)
from crisp.utils.evaluation_metrics import rotation_error, translation_error
from crisp.utils import diff_operators

from experiments.unified_model.dataset_checks import single_batch_sanity_test


def implicit_loss_helper(
    pred_sdf,
    gradient,
    gt_normals,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    shape_code,
    recons_loss_fn=None,
    opt=None,
    **kwargs,
):
    # nocs loss
    nocs_l = nocs_loss(pred_nocs=pred_nocs, exp_nocs=gt_nocs, mask=mask, threshold=opt.loss_nocs_threshold)

    # recons loss
    mnfld_grad, nonmnfld_grad = (gradient[:, :on_surface_cutoff, :], gradient[:, on_surface_cutoff:, :])
    recons_l, recons_loss_terms = recons_loss_fn(
        mnfld_pred=pred_sdf[:, :on_surface_cutoff, :],
        mnfld_grad=mnfld_grad,
        nonmnfld_pred=pred_sdf[:, on_surface_cutoff:, :],
        gradients=gradient,
        mnfld_normals=gt_normals[:, :on_surface_cutoff, :],
        sdf_weight=opt.recons_df_weight,
        inter_weight=opt.recons_inter_weight,
        normal_weight=opt.recons_normal_weight,
        grad_weight=opt.recons_grad_weight,
    )

    # TODO: Add batch consistent term for UDF
    shape_reg_loss = torch.sum(torch.square(shape_code))
    total_l = nocs_l + recons_l + shape_reg_loss
    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
        "shape_reg_loss": shape_reg_loss,
        "recons_loss_terms": recons_loss_terms,
    }


def explicit_loss_helper(
    pred_sdf,
    gt_sdf,
    gradient,
    gt_normals,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    shape_code,
    opt=None,
    **kwargs,
):
    # nocs loss
    nocs_l = nocs_loss(pred_nocs=pred_nocs, exp_nocs=gt_nocs, mask=mask, threshold=opt.loss_nocs_threshold)

    # sdf loss
    mnfld_grad, nonmnfld_grad = (gradient[:, :on_surface_cutoff, :], gradient[:, on_surface_cutoff:, :])
    mnfld_normals = None if gt_normals is None else gt_normals[:, :on_surface_cutoff, :]
    recons_l, recons_loss_terms = metric_sdf_loss(
        sdf_pred=pred_sdf.squeeze(2),
        sdf_gt=gt_sdf,
        mnfld_grad=mnfld_grad,
        gradients=gradient,
        mnfld_normals=mnfld_normals,
        sdf_weight=opt.recons_df_weight,
        normal_weight=opt.recons_normal_weight,
        grad_weight=opt.recons_grad_weight,
    )
    shape_reg_loss = torch.sum(torch.square(shape_code))
    total_l = (
        opt.nocs_loss_weight * nocs_l
        + opt.recons_loss_weight * recons_l
        + opt.shape_code_regularization_weight * shape_reg_loss
    )

    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
        "shape_reg_loss": shape_reg_loss,
        "recons_loss_terms": recons_loss_terms,
    }


def validation_loss_helper(
    pred_sdf,
    gt_sdf,
    on_surface_cutoff,
    global_nonmnfld_start,
    global_nonmnfld_count,
    pred_nocs,
    gt_nocs,
    mask,
    shape_code,
    opt=None,
):
    # nocs loss
    nocs_l = nocs_loss(pred_nocs=pred_nocs, exp_nocs=gt_nocs, mask=mask, threshold=opt.loss_nocs_threshold)

    recons_l = metric_sdf_loss_no_gradient(
        sdf_pred=pred_sdf.squeeze(2),
        sdf_gt=gt_sdf,
        sdf_weight=opt.recons_df_weight,
    )

    total_l = opt.nocs_loss_weight * nocs_l + opt.recons_loss_weight * recons_l

    return {
        "total_loss": total_l,
        "nocs_loss": nocs_l,
        "recons_loss": recons_l,
    }


def train(
    fabric,
    model: JointShapePoseNetwork,
    optimizer,
    scheduler,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    validate_per_n_epochs=50,
    log_every_iter=5,
    model_save_path="./model_ckpts",
    hparams=None,
    opt=None,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    on_surface_cutoff = train_dataloader.dataset.dataset.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.dataset.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.dataset.sample_global_nonmnfld_points_count

    model.train()
    best_val_loss = torch.tensor(np.inf)
    ctr_same_cls_weight = opt.contrastive_same_object_weight
    ctr_diff_cls_weight = opt.contrastive_different_object_weight
    for epoch in range(num_epochs):
        print(f"Training Epoch = {epoch} out of {num_epochs}.")
        start_time = time.time()
        avg_train_loss = 0

        if opt.contrastive_late_start and epoch < opt.contrastive_delay_epochs:
            ctr_same_cls_weight, ctr_diff_cls_weight = 0, 0

        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # nocs related
            rgb_img, segmap, gt_nocs, coords, gt_sdf = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["coords"],
                batch["sdf"],
            )
            # imgs_show(rgb_img, unnormalize=True)
            optimizer.zero_grad()

            # only 1 object per image in synthetic training set
            mask = (segmap == 1).unsqueeze(1)

            # forward pass
            coords = coords.clone().detach().requires_grad_(True)
            nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)
            gradient = diff_operators.gradient(sdf, coords)
            loss_terms = loss_fn(
                pred_sdf=sdf,
                gradient=gradient,
                gt_normals=None,
                on_surface_cutoff=on_surface_cutoff,
                global_nonmnfld_start=global_nonmnfld_start,
                global_nonmnfld_count=global_nonmnfld_count,
                shape_code=shape_code,
                pred_nocs=nocs_map,
                gt_nocs=gt_nocs,
                gt_sdf=gt_sdf,
                mask=mask,
                contrastive_same_object_weight=ctr_same_cls_weight,
                contrastive_different_object_weight=ctr_diff_cls_weight,
            )
            loss = loss_terms["total_loss"]

            if i % log_every_iter == 0:
                c_iter = epoch * len(train_dataloader) + i
                nocs_l, recons_l, recons_terms = (
                    loss_terms["nocs_loss"],
                    loss_terms["recons_loss"],
                    loss_terms["recons_loss_terms"],
                )
                fabric.log("Train/total_loss", loss.item(), step=c_iter)
                fabric.log("Train/nocs_loss/total", nocs_l.item(), step=c_iter)
                fabric.log("Train/recons_loss/total", recons_l.item(), step=c_iter)
                if "shape_reg_loss" in loss_terms.keys():
                    fabric.log("Train/shape_reg_loss/total", loss_terms["shape_reg_loss"].item(), step=c_iter)
                if "same_cls_loss" in loss_terms.keys():
                    fabric.log("Train/same_cls_loss/total", loss_terms["same_cls_loss"].item(), step=c_iter)
                if "diff_cls_loss" in loss_terms.keys():
                    fabric.log("Train/diff_cls_loss/total", loss_terms["diff_cls_loss"].item(), step=c_iter)

                if "sdf" in recons_terms.keys():
                    # df means distance field
                    fabric.log("Train/recons_loss/df", recons_terms["sdf"].item(), step=c_iter)
                if "inter" in recons_terms.keys():
                    fabric.log("Train/recons_loss/inter", recons_terms["inter"].item(), step=c_iter)
                if "normal" in recons_terms.keys():
                    fabric.log("Train/recons_loss/normal", recons_terms["normal"].item(), step=c_iter)
                if "grad" in recons_terms.keys():
                    fabric.log("Train/recons_loss/grad", recons_terms["grad"].item(), step=c_iter)
                for logger in fabric._loggers:
                    logger.log_hyperparams(hparams)

            # backward pass
            fabric.backward(loss)
            # fabric.clip_gradients(model, optimizer, clip_val=1.0)
            optimizer.step()
            avg_train_loss += loss.item()

            if opt.recons_lipschitz_normalization_type is not None:
                # projected gradient descent described in
                # https://arxiv.org/pdf/1804.04368
                # modify the weights of the layers without affecting gradients
                apply_cst = partial(
                    apply_lipschitz_constraint,
                    lipschitz_constraint_type=opt.recons_lipschitz_normalization_type,
                    lambda_=opt.recons_lipschitz_lambda,
                )
                model.recons_net.apply(apply_cst)

            if scheduler is not None:
                scheduler.step()

        end_time = time.time()

        print(f"Saving model at epoch={epoch}.")
        # if fabric.global_rank == 0:
        state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
        fabric.save(os.path.join(model_save_path, "checkpoint.pth"), state)
        # fabric.barrier()

        # if fabric.global_rank == 0:
        if epoch % validate_per_n_epochs == 0:
            print(f"epoch: {epoch}, loss: {loss}, per epoch time: {(end_time - start_time) / validate_per_n_epochs}")
            with torch.no_grad():
                val_loss = validate(fabric, model, val_dataloader, opt=opt)

            fabric.log("Val/total_loss", val_loss, step=epoch)
            if val_loss < best_val_loss:
                state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
                print(f"Current val loss = {val_loss} < Best val loss = {best_val_loss}. Saving model.")
                fabric.save(os.path.join(model_save_path, "checkpoint_best_val.pth"), state)
                best_val_loss = val_loss

        del rgb_img, coords, segmap, gt_nocs, gt_sdf, gradient, nocs_map, loss_terms
        torch.cuda.empty_cache()

        # fabric.barrier()


def validate(fabric, model, dataloader, opt):
    torch.cuda.empty_cache()
    on_surface_cutoff = dataloader.dataset.dataset.sample_surface_points_count
    global_nonmnfld_start = dataloader.dataset.dataset.global_nonmnfld_points_start
    global_nonmnfld_count = dataloader.dataset.dataset.sample_global_nonmnfld_points_count

    model.eval()
    total_loss = 0
    fabric.print("Validation...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # nocs related
        rgb_img, segmap, gt_nocs, coords, gt_sdf = (
            batch["rgb"],
            batch["instance_segmap"],
            batch["nocs"],
            batch["coords"],
            batch["sdf"],
        )

        # only 1 object per image in synthetic training set
        mask = (segmap == 1).unsqueeze(1)

        # forward pass
        nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)
        loss_terms = validation_loss_helper(
            pred_sdf=sdf,
            on_surface_cutoff=on_surface_cutoff,
            global_nonmnfld_start=global_nonmnfld_start,
            global_nonmnfld_count=global_nonmnfld_count,
            pred_nocs=nocs_map,
            shape_code=shape_code,
            gt_nocs=gt_nocs,
            gt_sdf=gt_sdf,
            mask=mask,
            opt=opt,
        )
        loss = loss_terms["total_loss"]

        total_loss += loss.cpu().float().item()
        del rgb_img, coords, segmap, gt_nocs, gt_sdf, nocs_map
        del loss, loss_terms
        torch.cuda.empty_cache()

    total_loss /= len(dataloader)
    fabric.print(f"Avg. validation loss across {len(dataloader)} batches: {total_loss}")
    model.train()
    return total_loss


def test(
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
    export_all_pred_recons_mesh=True,
    export_average_pred_recons_mesh=True,
    artifacts_save_dir="./artifacts",
):
    """Test the following:
    1. Pose errors from NOCS registration
    2. Reconstruction error (deviation from 0 set)
    3. Visualize quality of shape reconstruction
    """
    safely_make_folders([artifacts_save_dir])
    model.eval()
    count = 0
    total_rot_err, total_trans_err = 0, 0
    obj2shpcode = defaultdict(list)
    objcounter = defaultdict(lambda: 0)

    fabric.print("Testing...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            (rgb_img, depth, segmap, gt_nocs, object_pc, cam_intrinsics, gt_world_T_cam, coords, metadata) = (
                batch["rgb"],
                batch["depth"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["object_pc"],
                batch["cam_intrinsics"],
                batch["cam_pose"],
                batch["coords"],
                batch["metadata"],
            )
            bs = rgb_img.shape[0]
            mask = (segmap == 1).unsqueeze(1)
            nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)

            if vis_sdf_sample_points:
                visualize_pcs_pyvista(
                    [coords[0, ...], object_pc[0, ...]], colors=["crimson", "lightblue"], pt_sizes=[6.0, 2.0]
                )

            voxel_res = dataloader.dataset.dataset.global_nonmnfld_points_voxel_res
            cube_scale = dataloader.dataset.dataset.sample_bounds[1] - dataloader.dataset.dataset.sample_bounds[0]
            voxel_origin = np.array([0, 0, 0]) - cube_scale / 2
            voxel_size = cube_scale / (voxel_res - 1)
            if vis_gt_sanity_test:
                print("Sanity checking the dataset (visualizing the GT data)...")
                single_batch_sanity_test(
                    mask=mask.squeeze(1),
                    gt_nocs=gt_nocs,
                    sdf_grid=batch["sdf_grid"],
                    object_pc=object_pc,
                    normalized_mesh=batch["normalized_mesh"],
                    depth=depth,
                    cam_intrinsics=cam_intrinsics,
                    gt_world_T_cam=gt_world_T_cam,
                    voxel_res=voxel_res,
                    voxel_origin=voxel_origin,
                    voxel_size=voxel_size,
                    vis_gt_nocs_registration=True,
                    vis_gt_nocs_and_cad=True,
                    vis_sdf_and_mesh=True,
                    normalized_recons=normalized_recons,
                )

            # nocs pose error
            s, cam_R_world, cam_t_cad, cam_Tsim_cad, _, _ = align_nocs_to_depth(
                masks=mask,
                nocs=nocs_map,
                depth=depth,
                intrinsics=cam_intrinsics,
                instance_ids=torch.arange(bs),
                img_path=None,
                verbose=False,
                normalized_nocs=normalized_recons,
            )

            # this is necessary to handle the case where we are using unnormalized recons
            # because the unified objects datasets' NOCS are normalized, the returned GT pose
            # will have a scale in its rotation
            # R_det = scale^3 * det(R) = scale^3
            R_det = torch.linalg.det(gt_world_T_cam[:, :3, :3])
            scale = torch.pow(R_det, 1 / 3)
            gt_world_T_cam[:, :3, :3] = gt_world_T_cam[:, :3, :3] / scale.reshape(bs, 1, 1)
            print(f"GT pose scale: {scale}, inv scale: {1 / scale}")

            gt_cam_T_world = se3_inverse_batched_torch(gt_world_T_cam)
            trans_err = translation_error(cam_t_cad.unsqueeze(-1), gt_cam_T_world[:, :3, -1].unsqueeze(-1))
            rot_err = rotation_error(cam_R_world, gt_world_T_cam[..., :3, :3].permute((0, 2, 1)))
            print(f"Pred. NOCS registration avg/median trans err: {torch.mean(trans_err)}/{torch.median(trans_err)}")
            print(
                f"Pred. NOCS registration avg/median rotation err (rad): {torch.mean(rot_err)}/{torch.median(trans_err)}"
            )
            print(f"Pred. NOCS scale: {s}")

            if vis_rgb_image:
                for j in range(bs):
                    rgb_img_cpu = rgb_img[j, ...].cpu()
                    rgb_img_cpu[0, ...] = rgb_img_cpu[0, ...] * 0.229 + 0.485
                    rgb_img_cpu[1, ...] = rgb_img_cpu[1, ...] * 0.224 + 0.456
                    rgb_img_cpu[2, ...] = rgb_img_cpu[2, ...] * 0.225 + 0.406
                    plt.imshow(rgb_img_cpu.permute(1, 2, 0).numpy())
                    plt.show()

            if vis_pred_nocs_and_cad:
                print("Visualizing model predicted NOCS (light blue) with object points (black).")
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
                    print("Visualize transformed CAD (green), transformed NOCS (lightblue) with depth (red) points.")
                    depth_pts, idxs = instance_depth_to_point_cloud_torch(
                        depth[j, ...], cam_intrinsics[j, ...], mask[j, ...].squeeze(0)
                    )
                    # NOCS transformed to depth
                    if normalized_recons:
                        nocs_pts = (nocs_map[j, :3, idxs[0], idxs[1]] - 0.5) * 2
                    else:
                        nocs_pts = nocs_map[j, :3, idxs[0], idxs[1]]

                    nocs_p = nocs_pts.reshape(3, -1)
                    cam_nocs_p = cam_Tsim_cad[j, :3, :3] @ nocs_p + cam_Tsim_cad[j, :3, 3].reshape(3, 1)

                    # CAD transformed to depth
                    cad_p = cam_Tsim_cad[j, :3, :3] @ object_pc[j, ...].T + cam_Tsim_cad[j, :3, 3].reshape(3, 1)

                    visualize_pcs_pyvista(
                        [cad_p, cam_nocs_p, depth_pts],
                        colors=["green", "lightblue", "crimson"],
                        pt_sizes=[2.0, 10.0, 2.0],
                    )

            # shape reconstruction
            shp_inf_flag = (
                vis_pred_recons or vis_pred_sdf or export_average_pred_recons_mesh or export_all_pred_recons_mesh
            )
            if shp_inf_flag:
                model.eval()
                _, _, batched_shape_code = model.forward(img=rgb_img, mask=mask, coords=coords)
                for j in range(bs):
                    shape_code = batched_shape_code[0, ...]
                    objkey = str(metadata[j]["dataset_name"]) + "-" + str(metadata[j]["obj_name"])
                    print(f"Object name/key: {objkey}")
                    objcounter[objkey] += 1
                    obj2shpcode[objkey].append(shape_code)
                    if vis_pred_recons or vis_pred_sdf or export_all_pred_recons_mesh:

                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shape_code.unsqueeze(0), coords=coords)

                        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                            model_fn=model_fn,
                            N=256,
                            max_batch=64**3,
                            cube_center=np.array([0, 0, 0]),
                            cube_scale=cube_scale * 1.2,
                        )
                        pred_mesh = convert_sdf_samples_to_mesh(
                            sdf_grid=sdf_grid,
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=voxel_size,
                            offset=None,
                            scale=None,
                        )

                        if vis_pred_sdf:
                            print("Visualize SDF field")
                            mesh_sdf_slices = gen_pyvista_voxel_slices(
                                sdf_grid.numpy(force=True), voxel_grid_origin, (voxel_size,) * 3
                            )
                            visualize_meshes_pyvista(
                                [pred_mesh, object_pc[j, ...], mesh_sdf_slices],
                                mesh_args=[{"opacity": 0.2, "color": "white"}, {"opacity": 0.2, "color": "red"}, None],
                            )

                        if vis_pred_recons:
                            print("Visualize reconstruction...")
                            print(f"Showing {metadata[j]['dataset_name']}-{metadata[j]['obj_name']}")
                            pred_mesh.show()
                        if export_all_pred_recons_mesh:
                            mesh_dir = os.path.join(artifacts_save_dir, "meshes", f"{objkey}")
                            safely_make_folders([mesh_dir])
                            pred_mesh.export(
                                os.path.join(mesh_dir, f"{objkey}_{objcounter[objkey]}.obj"), file_type="obj"
                            )

            total_rot_err += torch.sum(rot_err)
            total_trans_err += torch.sum(trans_err)
            count += bs
            gc.collect()
            torch.cuda.empty_cache()

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
                pred_mesh = convert_sdf_samples_to_mesh(
                    sdf_grid=sdf_grid,
                    voxel_grid_origin=voxel_grid_origin,
                    voxel_size=voxel_size,
                    offset=None,
                    scale=None,
                )
                pred_mesh.export(mesh_fname)


@dataclass
class ExpSettings:
    dataset_dir: str
    shapenet_data_dir: str = None
    bop_data_dir: str = None
    replicacad_data_dir: str = None
    spe3r_data_dir: str = None
    uhumans_data_dir: str = None
    model_ckpts_save_dir: str = "./model_ckpts"
    checkpoint_path: str = None
    preload_to_mem: bool = False
    preload_sdf_to_mem: bool = True
    generate_obj_cache: bool = True
    force_recompute_sdf: bool = False

    # dataset
    pc_size: int = 60000
    per_batch_sample_surface_points_count: int = 1500
    per_batch_sample_local_nonmnfld_points_count: int = 1500
    per_batch_sample_global_nonmnfld_points_count: int = 4000
    global_nonmnfld_voxel_res: int = 128
    normalized_recons: bool = True
    dataset_debug_vis: bool = False
    random_split: bool = True
    dataloader_workers: int = 2

    # backbone model
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    freeze_pretrained_backbone_weights: bool = True
    backbone_model_path: str = None
    log_root_dir: str = "logs"
    backbone_input_res: tuple = (420, 420)

    # implicit recons model
    recons_num_layers: int = 5
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"
    recons_modulate_last_layer: bool = False
    recons_encoder_type: str = "ffn"
    recons_loss_type: str = "sdf"
    recons_df_loss_mode: str = "metric"
    recons_shape_code_normalization: str = None
    recons_shape_code_norm_scale: float = 10
    recons_lipschitz_normalization_type: str = None
    recons_lipschitz_lambda: float = 2

    # nocs model
    nocs_network_type: str = "dpt"
    nocs_channels: int = 256
    nocs_lateral_layers_type: str = "spaced"

    # training
    loss_nocs_threshold: float = 0.1
    num_epochs: int = 500
    validate_per_n_epochs: int = 20
    batch_size: int = 20
    lr: float = 3e-4
    nocs_lr: float = 1e-4
    recons_lr: float = 3e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine-anneal"
    cosine_anneal_period: int = 20000
    cosine_anneal_T_mult: int = 2

    # distributed
    num_devices: int = 1
    num_nodes: int = 1

    # overall loss
    nocs_loss_weight: float = 5e1
    recons_loss_weight: float = 1.0
    nocs_min: float = float("-inf")
    nocs_max: float = float("inf")

    # loss for recons module
    recons_df_weight: float = 3e3
    recons_inter_weight: float = 2e2
    recons_normal_weight: float = 0
    recons_grad_weight: float = 5e1

    # shape code regularization
    shape_code_regularization_weight: float = 0

    # contrastive regularization
    use_contrastive_regularization: bool = False
    num_classes_per_batch: int = 2
    contrastive_same_object_weight: float = 1.0
    contrastive_different_object_weight: float = 1.0
    contrastive_late_start: bool = False
    contrastive_delay_epochs: int = 150

    # loading model & testing
    test_only: bool = False
    test_on_unseen_objects: bool = False
    test_gt_nocs_pose: bool = False
    unseen_shapenet_render_data_dir: str = None
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    resume_from_ckpt: bool = False
    checkpoint_path: str = None

    vis_sdf_sample_points: bool = False
    vis_pred_nocs_registration: bool = False
    vis_pred_nocs_and_cad: bool = False
    vis_pred_recons: bool = False
    vis_gt_sanity_test: bool = False
    vis_rgb_images: bool = False
    vis_pred_sdf: bool = False
    export_all_pred_recons_mesh: bool = False
    export_average_pred_recons_mesh: bool = True
    artifacts_save_dir: str = "artifacts"

    # seed
    fixed_random_seed: bool = True

    # automatically populated if missing
    exp_id: str = None


def main(opt: ExpSettings):
    if opt.fixed_random_seed:
        LF.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    artifacts_dump_path = os.path.join(opt.artifacts_save_dir, opt.exp_id)
    print("Experiment/checkpoint/logs dump path: " + exp_dump_path)

    # one logger to log root dir, another to the exp folder
    tb_logger_root = LF.loggers.TensorBoardLogger(
        root_dir=os.path.join(opt.log_root_dir, "tensorboard"), name="joint_model_training", flush_secs=10, max_queue=5
    )
    tb_logger_exp = LF.loggers.TensorBoardLogger(root_dir=exp_dump_path, name="log", flush_secs=10, max_queue=5)
    strat = LFS.DDPStrategy(timeout=datetime.timedelta(seconds=6000))
    fabric = L.Fabric(
        accelerator="auto",
        loggers=[tb_logger_root, tb_logger_exp],
        strategy=strat,
        devices=opt.num_devices,
        num_nodes=opt.num_nodes,
    )
    fabric.launch()

    # dataloaders
    shape_ds = UnifiedObjects(
        folder_path=opt.dataset_dir,
        shapenet_dataset_path=opt.shapenet_data_dir,
        bop_dataset_path=opt.bop_data_dir,
        spe3r_dataset_path=opt.spe3r_data_dir,
        replicacad_dataset_path=opt.replicacad_data_dir,
        uhumans_objs_dataset_path=opt.uhumans_data_dir,
        preload_to_mem=opt.preload_to_mem,
        preload_sdf_to_mem=opt.preload_sdf_to_mem,
        generate_obj_cache=opt.generate_obj_cache,
        pc_size=opt.pc_size,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        sample_bounds=(-1.0, 1.0),
        debug_vis=opt.dataset_debug_vis,
        normalized_recons=opt.normalized_recons,
        force_recompute_sdf=opt.force_recompute_sdf,
        data_to_output=[
            "rgb",
            "nocs",
            "coords",
            "instance_segmap",
            "metadata",
            "cam_intrinsics",
            "cam_pose",
            "sdf",
        ],
    )

    if opt.random_split:
        print("Using random splits.")
        splits = tchdata.random_split(shape_ds, [0.1, 0.1, 0.8])
    else:
        print("Using fixed splits.")
        print("Using all for training & testing.")
        splits = [
            tchdata.Subset(shape_ds, list(range(len(shape_ds)))),
            tchdata.Subset(shape_ds, list(range(max(int(0.1 * len(shape_ds)), 1)))),
            tchdata.Subset(shape_ds, list(range(len(shape_ds)))),
        ]

    if not opt.use_contrastive_regularization:
        test_dl, val_dl, train_dl = [
            tchdata.DataLoader(
                x,
                shuffle=True,
                num_workers=opt.dataloader_workers,
                batch_size=opt.batch_size,
                collate_fn=unified_objects_collate_fn,
                drop_last=True,
            )
            for x in splits
        ]
        test_dl, val_dl, train_dl = fabric.setup_dataloaders(test_dl, val_dl, train_dl, use_distributed_sampler=True)
    else:
        print("Using contrastive regularization.")
        B = int(opt.batch_size / opt.num_classes_per_batch)
        print(f"Using batch size of {B} for each class.")

        dls = []
        for x in splits:
            bsampler = DistributedObjClassBatchSampler(
                per_obj_class_batch_size=B,
                num_classes_per_batch=opt.num_classes_per_batch,
                unified_ds=x,
                num_replicas=fabric.world_size,
                rank=fabric.global_rank,
            )

            dl = tchdata.DataLoader(
                x,
                num_workers=opt.dataloader_workers,
                collate_fn=unified_objects_collate_fn,
                batch_sampler=bsampler,
            )
            dls.append(dl)
        test_dl, val_dl, train_dl = fabric.setup_dataloaders(*dls, use_distributed_sampler=False)

    # joint nocs & recons model
    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=opt.recons_num_layers,
        recons_hidden_dim=256,
        recons_modulate_last_layer=opt.recons_modulate_last_layer,
        backbone_model=opt.backbone_model_name,
        local_backbone_model_path=opt.backbone_model_path,
        freeze_pretrained_weights=opt.freeze_pretrained_backbone_weights,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
        nocs_channels=opt.nocs_channels,
        recons_encoder_type=opt.recons_encoder_type,
        lateral_layers_type=opt.nocs_lateral_layers_type,
        backbone_input_res=opt.backbone_input_res,
        normalize_shape_code=opt.recons_shape_code_normalization,
        recons_shape_code_norm_scale=opt.recons_shape_code_norm_scale,
    )
    optim = torch.optim.Adam(
        model.get_lr_params_list(nocs_lr=opt.nocs_lr, recons_lr=opt.recons_lr), lr=opt.lr, weight_decay=opt.weight_decay
    )

    hparams = dataclasses.asdict(opt)
    if opt.test_only or opt.resume_from_ckpt:
        print("Loading model checkpoint.")
        state = fabric.load(opt.checkpoint_path)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        hparams = state["hparams"]

    model, optim = fabric.setup(model, optim)

    if opt.scheduler == "cosine-anneal":
        scheduler = CosineAnnealingWarmRestarts(optim, T_0=opt.cosine_anneal_period, T_mult=opt.cosine_anneal_T_mult)
    elif opt.scheduler == "none":
        scheduler = None
    else:
        raise NotImplementedError

    # loss closure to modify behavior & hyperparams of loss functions
    if opt.recons_loss_type == "sdf":
        recons_loss_fn = siren_sdf_fast_loss
    elif opt.recons_loss_type == "udf":
        recons_loss_fn = siren_udf_loss

    if opt.use_contrastive_regularization:
        print("Using contrastive regularization for losses.")
        assert opt.num_classes_per_batch == 2

        def loss_fn(**kwargs):
            return two_cls_contrastive_explicit_loss_helper(**kwargs, opt=opt)

    else:
        if opt.recons_df_loss_mode == "implicit":

            def loss_fn(**kwargs):
                return implicit_loss_helper(**kwargs, recons_loss_fn=recons_loss_fn, opt=opt)

        elif opt.recons_df_loss_mode == "metric":

            def loss_fn(**kwargs):
                return explicit_loss_helper(**kwargs, opt=opt)

        else:
            raise ValueError(f"Invalid recons_df_loss_mode: {opt.recons_df_loss_mode}")

    if not opt.test_only:
        train(
            fabric=fabric,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            loss_fn=loss_fn,
            num_epochs=opt.num_epochs,
            validate_per_n_epochs=opt.validate_per_n_epochs,
            model_save_path=exp_dump_path,
            hparams=hparams,
            opt=opt,
        )

    test_dl.dataset.dataset.data_to_output.extend(["sdf_grid", "normalized_mesh", "object_pc", "depth"])
    test(
        fabric,
        model,
        test_dl,
        loss_fn=None,
        normalized_recons=opt.normalized_recons,
        vis_sdf_sample_points=opt.vis_sdf_sample_points,
        vis_pred_nocs_registration=opt.vis_pred_nocs_registration,
        vis_pred_nocs_and_cad=opt.vis_pred_nocs_and_cad,
        vis_pred_recons=opt.vis_pred_recons,
        vis_pred_sdf=opt.vis_pred_sdf,
        vis_gt_sanity_test=opt.vis_gt_sanity_test,
        vis_rgb_image=opt.vis_rgb_images,
        artifacts_save_dir=opt.artifacts_save_dir,
        export_all_pred_recons_mesh=opt.export_all_pred_recons_mesh,
        export_average_pred_recons_mesh=opt.export_average_pred_recons_mesh,
    )

    return


if __name__ == "__main__":
    """Joint model training"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    # generate random timestamped experiment ID
    if opt.exp_id is None:
        opt.exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    if not opt.test_only:
        safely_make_folders([exp_dump_path])
        parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))

    main(ExpSettings(**opt))
