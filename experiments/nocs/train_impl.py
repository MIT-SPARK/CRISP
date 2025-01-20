import os
import time
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch
from tqdm import tqdm

# local lib imports
from crisp.models.joint import JointShapePoseNetwork
import crisp.models.pipeline
from crisp.models.pipeline import Pipeline
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.models.certifier import FrameCertifier
from crisp.models.loss_functions import (
    nocs_loss,
    nocs_loss_clamped,
    siren_udf_loss,
    siren_sdf_fast_loss,
    metric_sdf_loss,
)
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.visualization_utils import visualize_sdf_slices_pyvista, visualize_meshes_pyvista, imgs_show
from matplotlib import pyplot as plt
from crisp.utils.math import se3_inverse_batched_torch, make_se3_batched
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh
from crisp.utils import diff_operators


def run_pipeline(
    fabric,
    pipeline: crisp.models.pipeline.Pipeline,
    optimizer,
    train_dataloader,
    num_epochs,
    visualize=False,
    save_every_epoch=10,
    model_save_path="./model_ckpts",
    artifacts_save_dir="./artifacts",
    ssl_train=True,
    calculate_test_metrics=False,
    calculate_test_recons_metrics=False,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    on_surface_cutoff = train_dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

    pipeline.train()
    best_val_loss = torch.tensor(np.inf)
    ssl_iter = 0
    all_metrics = []
    for epoch in range(num_epochs):
        print(f"Training Epoch = {epoch} out of {num_epochs}.")
        start_time = time.time()
        best_cert_percent = 0
        avg_train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            c_iter = epoch * len(train_dataloader) + i
            # TODO: Update this to use the NOCS dataset
            rgb_img, segmap, depth, K, model_name = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["depth"],
                batch["cam_intrinsics"].unsqueeze(1),
                batch["model_name"],
            )
            frames_info = [{"model_name": model_name[ii]} for ii in range(rgb_img.shape[0])]

            # one object per image
            objs_info = [[{"bbox": None, "id_in_segm": 1, "label": model_name[ii]}] for ii in range(rgb_img.shape[0])]

            # the mask should be already for the correct object
            mask = segmap.unsqueeze(1)

            # forward pass
            payload = pipeline(
                rgbs=rgb_img, masks=mask, depths=depth, intrinsics=K, objs=objs_info, frames_info=frames_info
            )

            # test metrics for logging
            if calculate_test_metrics:
                gt_cam_T_world = se3_inverse_batched_torch(batch["cam_pose"]).detach()
                est_cam_T_world = make_se3_batched(R=payload["cam_R_nocs"], t=payload["cam_t_nocs"]).detach()
                c_metrics = calculate_metrics(
                    model=pipeline.model,
                    gt_cam_T_world=gt_cam_T_world,
                    est_cam_T_world=est_cam_T_world,
                    gt_shape_names=batch["model_name"],
                    batched_shape_code=payload["shape_code"].detach(),
                    shape_bounds=train_dataloader.dataset.unified_objects.sample_bounds,
                    object_pc=batch["object_pc"],
                    calculate_pred_recons_metrics=calculate_test_recons_metrics,
                )
                all_metrics.extend(c_metrics)

                # save test metrics
                if i % 100 == 0:
                    np.save(os.path.join(artifacts_save_dir, "metrics.npy"), all_metrics, allow_pickle=True)

            if visualize:
                print("Visualize pipeline outputs...")
                if "precrt_nocs" in payload.keys() and "postcrt_nocs" in payload.keys():
                    print("Visualize corrected NOCS and CAD transformed into NOCS frame...")
                    B = payload["postcrt_nocs"].shape[0]
                    for i in range(B):
                        print(f"Obj index = {i}")
                        frame_i, obj_i = payload["frame_index"][i], payload["obj_index"][i]
                        gt_obj_name = batch["model_name"][i]
                        gt_recons_pc = train_dataloader.dataset.unified_objects.distinct_objects["spe3r"][gt_obj_name][
                            "surface_points"
                        ]

                        visualize_pcs_pyvista(
                            [
                                payload["postcrt_nocs"][i, ...].detach(),
                                payload["precrt_nocs"][i, ...].detach(),
                                gt_recons_pc.detach(),
                            ],
                            colors=["crimson", "blue", "green"],
                            pt_sizes=[10.0, 10.0, 5.0],
                        )

                        # shape + nocs
                        visualize_sdf_slices_pyvista(
                            payload["shape_code"][i, ...].unsqueeze(0),
                            pipeline.model.recons_net,
                            additional_meshes=[gt_recons_pc.detach()],
                        )

            # self-supervised backprop if we have enough certified samples
            all_ssl_loss = None
            if ssl_train:
                all_ssl_loss, nocs_ssl_steps, recons_ssl_steps = pipeline.ssl_step(optimizer, fabric=fabric)

            cert_percent = torch.sum(payload["cert_mask"]) / payload["cert_mask"].shape[0]
            fabric.log("Train/cert_percent", cert_percent, step=c_iter)
            print(f"c_iter={c_iter}, SSL_iter={ssl_iter}, cert%={cert_percent}, best cert%={best_cert_percent}")
            if best_cert_percent < cert_percent:
                best_cert_percent = cert_percent
            if all_ssl_loss is not None:
                for ssl_loss in all_ssl_loss:
                    ssl_iter += 1
                    fabric.log("Train/total_loss", ssl_loss["total_loss"], step=ssl_iter)
                    fabric.log("Train/nocs_loss/total", ssl_loss["nocs_loss"], step=ssl_iter)
                    fabric.log("Train/shape_loss/total", ssl_loss["shape_loss"], step=ssl_iter)

        if ssl_train:
            state = {"model": pipeline.model, "optimizer": optimizer, "epoch": epoch, "ssl-iter": ssl_iter}
            fabric.save(os.path.join(model_save_path, f"checkpoint.pth"), state)

            if epoch % save_every_epoch == 0:
                print(f"Saving model at epoch={epoch}.")
                fabric.save(os.path.join(model_save_path, f"checkpoint_epoch={epoch}.pth"), state)

        end_time = time.time()
        torch.cuda.empty_cache()

    # save the final result
    if ssl_train:
        state = {"model": pipeline.model, "optimizer": optimizer, "epoch": epoch, "ssl-iter": ssl_iter}
        print(f"Saving final model at epoch={epoch}.")
        fabric.save(os.path.join(model_save_path, f"checkpoint_epoch={epoch}.pth"), state)

    return all_metrics


def train_ssl(
    fabric,
    pipeline,
    optimizers,
    scheduler,
    train_dataloader,
    val_dataloader,
    num_epochs,
    visualize=False,
    validate_per_n_epochs=1,
    save_every_epoch=1,
    model_save_path="./model_ckpts",
    hparams=None,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    return run_pipeline(
        fabric,
        pipeline,
        optimizers,
        train_dataloader,
        num_epochs,
        visualize=visualize,
        save_every_epoch=save_every_epoch,
        model_save_path=model_save_path,
        ssl_train=True,
        ssl_type=opt.ssl_type,
        calculate_test_metrics=False,
        opt=opt,
    )


def train_sl(
    fabric,
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    validate_per_n_epochs=50,
    model_save_path="./model_ckpts",
    log_every_iter=1,
    visualize=None,
    hparams=None,
):
    """Main train loop for supervised model."""
    if visualize is not None:
        train_dataloader.dataset.data_to_output.extend(["normalized_mesh", "object_pc"])

    on_surface_cutoff = train_dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

    model.train()
    best_val_loss = torch.tensor(np.inf)
    for epoch in range(num_epochs):
        print(f"Training Epoch = {epoch} out of {num_epochs}.")
        start_time = time.time()
        avg_train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            c_iter = epoch * len(train_dataloader) + i

            # nocs related
            rgb_img, segmap, gt_nocs, coords, gt_sdf = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["coords"],
                batch["sdf"],
            )
            B = rgb_img.shape[0]
            optimizer.zero_grad()

            # the mask should be already for the correct object
            mask = segmap.unsqueeze(1)

            if "rgb" in visualize:
                print("Visualize RGB and mask...")
                imgs_show(rgb_img)
                imgs_show(mask.int())

            if "object_pc" in visualize:
                print("Visualize sampled coords and mesh.")
                normalized_mesh, object_pc = batch["normalized_mesh"], batch["object_pc"]
                for j in range(rgb_img.shape[0]):
                    visualize_meshes_pyvista(
                        [object_pc[j, ...], normalized_mesh[j][0]],
                        plotter_shape=(1, 1),
                        mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                    )

                print("Visualize NOCS and object mesh points. Note: we assume unnormalized NOCS.")
                for j in range(rgb_img.shape[0]):
                    nocs_pts = (gt_nocs[j, ...] * mask[j, ...]).detach().cpu().numpy().reshape((3, -1))
                    visualize_meshes_pyvista(
                        [object_pc[j, ...], nocs_pts],
                        plotter_shape=(1, 1),
                        mesh_args=[{"opacity": 0.2, "color": "blue"}, {"opacity": 0.2, "color": "red"}],
                    )

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
                pred_nocs=nocs_map,
                gt_nocs=gt_nocs,
                gt_sdf=gt_sdf,
                mask=mask,
            )
            loss = loss_terms["total_loss"]

            if "sdf" in visualize:
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
                            cube_scale=10,
                        )
                        pred_mesh = convert_sdf_samples_to_mesh(
                            sdf_grid=sdf_grid,
                            voxel_grid_origin=voxel_grid_origin,
                            voxel_size=voxel_size,
                            offset=None,
                            scale=None,
                        )

                        visualize_meshes_pyvista(
                            [pred_mesh],
                            mesh_args=[{"opacity": 0.2, "color": "white"}],
                        )

            if i % log_every_iter == 0:
                nocs_l, recons_l, recons_terms = (
                    loss_terms["nocs_loss"],
                    loss_terms["recons_loss"],
                    loss_terms["recons_loss_terms"],
                )
                fabric.log("Train/total_loss", loss.item(), step=c_iter)
                fabric.log("Train/nocs_loss/total", nocs_l.item(), step=c_iter)
                fabric.log("Train/recons_loss/total", recons_l.item(), step=c_iter)
                if "sdf" in recons_terms.keys():
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

            # log gradients
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            fabric.log("Train/grad_norm", total_norm, step=c_iter)

            fabric.clip_gradients(model, optimizer, clip_val=1.0)

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            fabric.log("Train/grad_norm_after_clip", total_norm, step=c_iter)

            optimizer.step()
            avg_train_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
        end_time = time.time()

        del rgb_img, coords, segmap, gt_nocs
        torch.cuda.empty_cache()

        if epoch % 2 == 0:
            print(f"epoch: {epoch}, loss: {loss}, per epoch time: {(end_time - start_time) / validate_per_n_epochs}")
            state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
            fabric.save(os.path.join(model_save_path, "checkpoint.pth"), state)

        # TODO: Fix nocs_camera_val key error
        # if epoch % validate_per_n_epochs == 0:
        #    print(f"epoch: {epoch}, loss: {loss}, per epoch time: {(end_time - start_time) / validate_per_n_epochs}")
        #    val_loss = validate_sl(fabric, model, val_dataloader, loss_fn)
        #    fabric.log("Val/total_loss", val_loss, step=epoch)
        #    if val_loss < best_val_loss:
        #        state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
        #        print(f"Current val loss = {val_loss} < Best val loss = {best_val_loss}. Saving model.")
        #        fabric.save(os.path.join(model_save_path, "checkpoint.pth"), state)
        #        best_val_loss = val_loss
    return


def validate_sl(fabric, model, dataloader, loss_fn):
    """Validation function for supervised model."""
    torch.cuda.empty_cache()
    on_surface_cutoff = dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

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

        # the mask should be already for the correct object
        mask = segmap.unsqueeze(1)

        # forward pass
        coords = coords.clone().detach().requires_grad_(True)
        nocs_map, sdf, _ = model(img=rgb_img, mask=mask, coords=coords)
        gradient = diff_operators.gradient(sdf, coords)
        loss, nocs_l, recons_l, recons_terms = loss_fn(
            pred_sdf=sdf,
            gradient=gradient,
            gt_normals=None,
            on_surface_cutoff=on_surface_cutoff,
            global_nonmnfld_start=global_nonmnfld_start,
            global_nonmnfld_count=global_nonmnfld_count,
            pred_nocs=nocs_map,
            gt_nocs=gt_nocs,
            gt_sdf=gt_sdf,
            mask=mask,
        )

        total_loss += loss.item()

    del rgb_img, coords, segmap, gt_nocs
    torch.cuda.empty_cache()

    total_loss /= len(dataloader)
    fabric.print(f"Avg. validation loss across {len(dataloader)} batches: {total_loss}")
    return total_loss
