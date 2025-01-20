import dataclasses
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils import data as tchdata
from tqdm import tqdm

# project local imports
import crisp.models.loss_functions as losses
from crisp.backend.slam import poses2odom
from crisp.datasets import bop
from crisp.models.pipeline import Pipeline
from crisp.utils.math import make_scaled_se3_inverse_batched
from crisp.utils.file_utils import safely_make_folders
from crisp.utils.visualization_utils import visualize_pcs_pyvista, visualize_sdf_slices_pyvista

from experiments.ycbv_object_slam.metrics import error_metrics


def train_sep(pipeline, optim, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path):
    """Main train function if using a separate dataloader for each scene"""
    all_results = []
    c_iter = 0
    for epoch in range(opt.num_epochs):
        for s_id in opt.scenes_to_load:
            # train on a specific trajectory
            dataset = bop.BOPSceneDataset(
                ds_name="ycbv",
                split=opt.split,
                bop_ds_dir=Path(opt.bop_ds_dir),
                load_depth=True,
                scenes_to_load=(s_id,),
            )
            ssl_dl = fabric.setup_dataloaders(
                tchdata.DataLoader(
                    dataset,
                    shuffle=False,
                    num_workers=0,
                    batch_size=opt.dataloader_batch_size,
                    collate_fn=dataset.collate_fn,
                )
            )

            # for each epoch:
            #   for each mini batch:
            #       correct
            #       certify
            #       buffer the certified samples (save to disk?)
            #       backprop self-supervision if buffer full? (no need to save everything then load again to do training)
            #   after finishing entire scene:
            #       RPGO
            #       certify
            #       put into buffer
            #       self-supervision
            metric_folder = os.path.join(exp_dump_path, f"traj_data", f"traj_{s_id}")
            safely_make_folders([metric_folder])
            traj_results = train_on_single_traj(
                fabric=fabric,
                pipeline=pipeline,
                optimizer=optim,
                train_dataloader=ssl_dl,
                objects_info=objects_info[opt.ds_name],
                object_pc_ds=obj_pc_ds,
                start_iter=c_iter,
                metric_dump_path=os.path.join(metric_folder, f"traj_results.npy"),
                checkpoint_dump_path=exp_dump_path,
                hparams=dataclasses.asdict(opt),
            )
            all_results.extend(traj_results)

            # save metrics
            np.save(os.path.join(metric_folder, f"traj_results.npy"), all_results)

            c_iter += len(ssl_dl)

    return


def train_on_single_traj(
    fabric,
    pipeline: Pipeline,
    optimizer,
    train_dataloader,
    objects_info,
    object_pc_ds,
    start_iter=0,
    start_ssl_iter=0,
    metric_dump_path="artifacts",
    checkpoint_dump_path="./",
    hparams=None,
):
    """Run this for each trajectory"""
    all_results = []
    ssl_iter = start_ssl_iter
    best_cert_percent = 0
    for logger in fabric._loggers:
        logger.log_hyperparams(hparams)

    for i, (rgbs, masks, depths, intrinsics, world_T_cam, gt_objs_data, frames_info) in tqdm(
        enumerate(train_dataloader)
    ):
        #   for each mini batch:
        #       detect
        #       correct
        #       certify
        #       buffer the certified samples (save to disk?)
        #       backprop self-supervision if buffer full? (no need to save everything then load again to do training)
        #   after finishing entire scene:
        #       RPGO
        #       certify
        #       put into buffer
        #       self-supervision
        print(f"in batch {i}")
        payload = pipeline.forward(rgbs, masks, depths, intrinsics, gt_objs_data, frames_info)

        # add odometry: view_id is the unique identifier to the pose node in the factor graph
        with torch.no_grad():
            odoms = poses2odom(world_T_cam, index_mapping_fn=lambda x: frames_info[x]["view_id"])
            pipeline.update_pgo_odometry(odometry=odoms)

            # add objects
            obj_det_view_ids = [
                frames_info[payload["frame_index"][x]]["view_id"] for x in range(len(payload["frame_index"]))
            ]
            pipeline.update_pgo_objs(
                object_labels=[
                    gt_objs_data[fid][x]["label"] for fid, x in zip(payload["frame_index"], payload["obj_index"])
                ],
                cam_R_obj=payload["cam_R_nocs"],
                cam_t_obj=payload["cam_t_nocs"],
                index2frame_fn=lambda x: obj_det_view_ids[x],
            )

        # self-supervised backprop if we have enough certified samples
        all_ssl_loses = pipeline.ssl_step(optimizer)

        c_iter = start_iter + i
        cert_percent = torch.sum(payload["cert_mask"]) / payload["cert_mask"].shape[0]
        fabric.log("Train/cert_percent", cert_percent, step=c_iter)
        print(f"c_iter={c_iter}, SSL_iter={ssl_iter}, cert%={cert_percent}, best cert%={best_cert_percent}")
        for ssl_loses in all_ssl_loses:
            ssl_iter += 1
            fabric.log("Train/total_loss", ssl_loses["total_loss"], step=ssl_iter)
            fabric.log("Train/nocs_loss/total", ssl_loses["nocs_loss"], step=ssl_iter)
            fabric.log("Train/shape_loss/total", ssl_loses["shape_loss"], step=ssl_iter)

        if best_cert_percent < cert_percent:
            state = {"model": pipeline.model, "optimizer": optimizer, "iter": c_iter, "ssl-iter": ssl_iter}
            print(f"Current cert percent = {cert_percent} > Best cert percent = {best_cert_percent}. Saving model.")
            fabric.save(os.path.join(checkpoint_dump_path, "checkpoint.pth"), state)
            best_cert_percent = cert_percent

        # gt error metrics
        results = error_metrics(
            gt_objs_data=gt_objs_data,
            depths=depths,
            masks=masks,
            intrinsics=intrinsics,
            objects_info=objects_info,
            object_pc_ds=object_pc_ds,
            payload=payload,
        )
        all_results.extend(results)

        # save metrics
        np.save(metric_dump_path, all_results)

        del payload
        torch.cuda.empty_cache()

    end_iter = ssl_iter
    ## pgo and multi-frame shape corrector
    # pipeline.optimize_pgo()
    # postcrt_shps = pipeline.correct_multiframe_shape()
    return all_results, end_iter


def train_mixed(pipeline, optims, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path):
    """Main train function if using a separate dataloader for each scene"""
    # load all trajectories
    dataset = bop.BOPSceneDataset(
        ds_name="ycbv",
        split=opt.split,
        bop_ds_dir=Path(opt.bop_ds_dir),
        load_depth=True,
        scenes_to_load=opt.scenes_to_load,
        bop19_targets=opt.bop19_targets,
    )
    ssl_dl = fabric.setup_dataloaders(
        tchdata.DataLoader(
            dataset,
            shuffle=True,
            num_workers=opt.dataloader_num_workers,
            batch_size=opt.dataloader_batch_size,
            collate_fn=dataset.collate_fn,
        )
    )

    for logger in fabric._loggers:
        logger.log_hyperparams(dataclasses.asdict(opt))

    all_results = []
    start_ssl_nocs_iter, start_ssl_shape_iter = 0, 0
    for epoch in range(opt.num_epochs):
        metric_folder = os.path.join(exp_dump_path, f"traj_data")
        safely_make_folders([metric_folder])
        print(f"Epoch: {epoch}")
        traj_results, end_ssl_nocs_iter, end_ssl_shape_iter = train_one_epoch(
            fabric=fabric,
            pipeline=pipeline,
            optimizers=optims,
            train_dataloader=ssl_dl,
            objects_info=objects_info[opt.ds_name],
            object_pc_ds=obj_pc_ds,
            unified_shape_ds=unified_shape_ds,
            ssl_type=opt.ssl_type,
            start_iter=epoch * len(ssl_dl),
            metric_dump_path=os.path.join(metric_folder, f"traj_results.npy"),
            checkpoint_dump_path=exp_dump_path,
            visualize=opt.visualize,
            start_ssl_nocs_iter=start_ssl_nocs_iter,
            start_ssl_shape_iter=start_ssl_shape_iter,
            opt=opt,
        )
        all_results.extend(traj_results)

        state = {
            "model": pipeline.model,
            "optimizers": optims,
            "ssl-nocs-iter": end_ssl_nocs_iter,
            "ssl-shape-iter": end_ssl_shape_iter,
        }
        if epoch % opt.save_every_epoch == 0:
            print(f"Saving model at epoch={epoch}.")
            fabric.save(os.path.join(exp_dump_path, f"checkpoint_epoch={epoch}.pth"), state)
        fabric.save(os.path.join(exp_dump_path, f"checkpoint.pth"), state)

        # save metrics
        np.save(os.path.join(metric_folder, f"traj_results.npy"), all_results)

        start_ssl_nocs_iter = end_ssl_nocs_iter
        start_ssl_shape_iter = end_ssl_shape_iter

        # cert schedule update
        if opt.cert_schedule_period is not None:
            if epoch != 0 and epoch % opt.cert_schedule_period == 0:
                pipeline.frame_certifier.depths_eps *= opt.cert_schedule_multiplier

    return


def contains_one_traj(frames_info):
    """Return True if frames_info contains one trajectory only"""
    traj_ids = set([x["scene_id"] for x in frames_info])
    return len(traj_ids) == 1


def train_one_epoch(
    fabric,
    pipeline: Pipeline,
    optimizers,
    train_dataloader,
    objects_info,
    object_pc_ds,
    unified_shape_ds,
    ssl_type="v1",
    start_iter=0,
    start_ssl_nocs_iter=0,
    start_ssl_shape_iter=0,
    mf_shp_ssl=False,
    metric_dump_path="artifacts",
    checkpoint_dump_path="./",
    save_every_iter=4000,
    visualize=False,
    opt=None,
):
    """Run this for each trajectory"""
    all_results = []
    best_cert_percent = 0
    ssl_nocs_iter, ssl_shape_iter = start_ssl_nocs_iter, start_ssl_shape_iter
    for i, (rgbs, masks, depths, intrinsics, world_T_cam, gt_objs_data, frames_info) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        print(f"in batch {i}")
        payload = pipeline.forward(rgbs, masks, depths, intrinsics, gt_objs_data, frames_info)

        if visualize:
            print("Visualize pipeline outputs...")
            B = payload["cam_s_nocs"].shape[0]
            if "precrt_shp_code" in payload.keys():
                print("Visualize original & corrected shape code generated meshes.")
                for i in range(B):
                    print(f"Cert? = {payload['cert_mask'][i]}")
                    frame_i, obj_i = payload["frame_index"][i], payload["obj_index"][i]
                    gt_obj_name = gt_objs_data[frame_i][obj_i]["label"]
                    gt_recons_pc = unified_shape_ds.distinct_objects["ycbv"][gt_obj_name]["surface_points"]

                    print("Visualize uncorrected shape code SDF slices & mesh.")
                    visualize_sdf_slices_pyvista(
                        payload["precrt_shp_code"].to(device="cuda")[i, ...].unsqueeze(0),
                        pipeline.model.recons_net,
                        cube_scale=pipeline.debug_settings["vis_sdf_grid_scale"],
                        additional_meshes=[gt_recons_pc],
                    )

                    print("Visualize corrected shape code SDF slices & mesh & point cloud.")
                    nocs_T_depth = make_scaled_se3_inverse_batched(
                        payload["cam_s_nocs"], payload["cam_R_nocs"], payload["cam_t_nocs"]
                    )
                    depth_coords = (
                        torch.bmm(nocs_T_depth[..., :3, :3], payload["pcs"]) + nocs_T_depth[..., :3, -1].reshape(-1, 3, 1)
                    ).transpose(-1, -2)

                    visualize_sdf_slices_pyvista(
                        payload["shape_code"][i, ...].unsqueeze(0),
                        pipeline.model.recons_net,
                        cube_scale=pipeline.debug_settings["vis_sdf_grid_scale"],
                        additional_meshes=[gt_recons_pc, depth_coords[i, ...]],
                    )

            if "precrt_nocs" in payload.keys() and "postcrt_nocs" in payload.keys():
                print("Visualize corrected NOCS and CAD transformed into NOCS frame...")
                for i in range(B):
                    print(f"Obj index = {i}")
                    frame_i, obj_i = payload["frame_index"][i], payload["obj_index"][i]
                    gt_obj_name = gt_objs_data[frame_i][obj_i]["label"]
                    gt_recons_pc = unified_shape_ds.distinct_objects["ycbv"][gt_obj_name]["surface_points"]

                    visualize_pcs_pyvista(
                        [
                            payload["postcrt_nocs"][i, ...].detach(),
                            payload["precrt_nocs"][i, ...].detach(),
                            gt_recons_pc.detach(),
                        ],
                        colors=["crimson", "blue", "green"],
                        pt_sizes=[10.0, 10.0, 5.0],
                    )

        c_iter = start_iter + i
        B, num_cert = payload["cert_mask"].shape[0], torch.sum(payload["cert_mask"])
        cert_percent = num_cert / payload["cert_mask"].shape[0]
        fabric.log("Train/cert_percent", cert_percent, step=c_iter)
        print(
            f"c_iter={c_iter}, SSL_nocs_iter={ssl_nocs_iter}, SSL_recons_iter={ssl_shape_iter}, cert%={cert_percent}, best cert%={best_cert_percent}"
        )
        if ssl_type == "v1":
            # self-supervised backprop if we have enough certified samples
            all_ssl_loss = pipeline.ssl_step(
                ssl_nocs_optimizer=optimizers["nocs"], ssl_recons_optimizer=optimizers["recons"], fabric=fabric
            )

            if all_ssl_loss is not None:
                for ssl_loss in all_ssl_loss["nocs_losses"]:
                    ssl_nocs_iter += 1
                    fabric.log(f"Train/nocs_loss", ssl_loss, step=ssl_nocs_iter)

                for ssl_loss in all_ssl_loss["shape_losses"]:
                    ssl_shape_iter += 1
                    fabric.log(f"Train/shape_loss", ssl_loss, step=ssl_shape_iter)

                if best_cert_percent < cert_percent:
                    state = {
                        "model": pipeline.model,
                        "optimizers": optimizers,
                        "iter": c_iter,
                        "ssl-nocs-iter": ssl_nocs_iter,
                        "ssl-recons-iter": ssl_shape_iter,
                    }
                    print(
                        f"Current cert percent = {cert_percent} > Best cert percent = {best_cert_percent}. Saving model."
                    )
                    fabric.save(os.path.join(checkpoint_dump_path, "checkpoint.pth"), state)
                    best_cert_percent = cert_percent

        elif ssl_type == "v2":
            if num_cert > 0:
                for optim in optimizers.values():
                    optim.zero_grad()

                # SDF network conditioned by shape code
                def f_sdf_conditioned(x):
                    return pipeline.model.recons_net.forward(shape_code=payload["shape_code"], coords=x)

                # directly use SDF-INPUT + SDF-NOCS loss
                l1 = losses.inv_transformed_depth_sdf_loss(
                    nocs=payload["postcrt_nocs"],
                    f_sdf=f_sdf_conditioned,
                    depth_pcs=payload["pcs"],
                    masks=payload["masks"],
                    reg_inlier_thres=payload["reg_inlier_thres"],
                    max_ransac_iters=100,
                    trim_quantile=opt.ssl_sdf_input_trim_quantile,
                    reduction="sum",
                )
                l1 = (l1 * payload["cert_mask"].reshape((B,))).sum() / num_cert
                l2 = losses.sdf_nocs_loss(
                    f_sdf_conditioned=f_sdf_conditioned,
                    nocs=payload["postcrt_nocs"],
                    weights=payload["masks"],
                    reduction="sum",
                )
                l2 = (l2 * payload["cert_mask"].reshape((B,))).sum() / num_cert
                l = opt.ssl_sdf_input_weight * l1 + opt.ssl_sdf_nocs_weight * l2
                fabric.backward(l)
                for optim in optimizers.values():
                    optim.step()

                fabric.log(f"Train/total_loss", l.item(), step=c_iter)
                fabric.log(f"Train/sdf_input_loss", l1.item(), step=c_iter)
                fabric.log(f"Train/sdf_nocs_loss", l2.item(), step=c_iter)
        else:
            raise ValueError(f"Unknown SSL type: {ssl_type}")

        #if i % save_every_iter == 0:
        #    state = {
        #        "model": pipeline.model,
        #        "optimizers": optimizers,
        #        "iter": c_iter,
        #        "ssl-nocs-iter": ssl_nocs_iter,
        #        "ssl-shape-iter": ssl_shape_iter,
        #    }
        #    print(f"Saving model at iter={c_iter}.")
        #    fabric.save(os.path.join(checkpoint_dump_path, f"checkpoint_iter={c_iter}.pth"), state)

        # gt error metrics
        results = error_metrics(
            gt_objs_data=gt_objs_data,
            depths=depths,
            masks=masks,
            intrinsics=intrinsics,
            model=None,
            calculate_pred_recons_metrics=False,
            objects_info=objects_info,
            object_pc_ds=object_pc_ds,
            payload=payload,
        )
        all_results.extend(results)

        # save metrics
        np.save(metric_dump_path, all_results)

        del payload, all_ssl_loss
        torch.cuda.empty_cache()

    if pipeline.mf_shape_code_corrector is not None:
        pipeline.mf_shape_code_corrector.clear_buffer()

    return all_results, ssl_nocs_iter, ssl_shape_iter


def train_mixed_mf_shp(pipeline, optim, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path):
    """
    Main train function if using a separate dataloader for each scene
    """
    raise NotImplementedError
    # load all trajectories
    dataset = bop.BOPSceneDataset(
        ds_name="ycbv",
        split=opt.split,
        bop_ds_dir=Path(opt.bop_ds_dir),
        load_depth=True,
        scenes_to_load=opt.scenes_to_load,
    )
    ssl_dl = fabric.setup_dataloaders(
        tchdata.DataLoader(
            dataset,
            shuffle=True,
            num_workers=opt.dataloader_num_workers,
            batch_size=opt.dataloader_batch_size,
            collate_fn=dataset.collate_fn,
        )
    )

    for logger in fabric._loggers:
        logger.log_hyperparams(dataclasses.asdict(opt))

    all_results = []
    for epoch in range(opt.num_epochs):
        metric_folder = os.path.join(exp_dump_path, f"traj_data")
        safely_make_folders([metric_folder])
        traj_results = train_one_epoch(
            fabric=fabric,
            pipeline=pipeline,
            optimizer=optim,
            train_dataloader=ssl_dl,
            objects_info=objects_info[opt.ds_name],
            object_pc_ds=obj_pc_ds,
            unified_shape_ds=unified_shape_ds,
            metric_dump_path=os.path.join(metric_folder, f"traj_results.npy"),
            checkpoint_dump_path=exp_dump_path,
            visualize=opt.visualize,
            opt=opt,
        )
        all_results.extend(traj_results)

        # save metrics
        np.save(os.path.join(metric_folder, f"traj_results.npy"), all_results)

    return
