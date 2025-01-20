import dataclasses
import os
import copy
from pathlib import Path

import PIL
from PIL import Image
import PIL.ImageOps
import numpy as np
import torch
from torch.utils import data as tchdata
from torchvision.transforms import functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

# project local imports
from crisp.backend.slam import poses2odom
from crisp.datasets import bop
from crisp.models.pipeline import Pipeline
from crisp.models.corrector import JointCorrector
from crisp.utils.file_utils import safely_make_folders
from crisp.utils.visualization_utils import (
    visualize_pcs_pyvista,
    visualize_sdf_slices_pyvista,
    imgs_show,
    save_screenshots_from_shape_codes,
    save_joint_visualizations,
    get_meshes_from_shape_codes,
    generate_video_from_meshes,
    generate_orbiting_video_from_meshes,
)
from experiments.ycbv_object_slam.metrics import error_metrics, shape_metrics


def test_sep(
    pipeline,
    objects_info,
    obj_pc_ds,
    unified_shape_ds,
    fabric,
    opt,
    exp_dump_path,
    use_torch_profile=False,
    dump_vis=False,
    dump_video=False,
):
    """Main train function if using a separate dataloader for each scene"""
    if use_torch_profile:
        opt.num_epochs = 1

    all_results = []
    for s_id in opt.scenes_to_load:
        # train on a specific trajectory
        dataset = bop.BOPSceneDataset(
            ds_name="ycbv",
            split=opt.split,
            bop_ds_dir=Path(opt.bop_ds_dir),
            load_depth=True,
            scenes_to_load=(s_id,),
            bop19_targets=opt.bop19_targets,
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

        metric_folder = os.path.join(exp_dump_path, f"traj_data", f"traj_{s_id}")
        safely_make_folders([metric_folder])
        traj_results = test_on_single_traj(
            fabric=fabric,
            pipeline=pipeline,
            train_dataloader=ssl_dl,
            objects_info=objects_info[opt.ds_name],
            object_pc_ds=obj_pc_ds,
            log_every_iter=5,
            metric_dump_path=os.path.join(metric_folder, f"traj_results.npy"),
            hparams=dataclasses.asdict(opt),
            torch_profile=use_torch_profile,
            visualize=opt.visualize_test,
            dump_vis=dump_vis,
            dump_video=dump_video,
        )
        all_results.extend(traj_results)

        if pipeline.output_degen_condition_number:
            mf_degen_conds = pipeline.get_mf_degen_condition_number()
            acc_nocs = pipeline.get_mf_acc_nocs()
            np.save(os.path.join(metric_folder, f"mf_degen_conds.npy"), mf_degen_conds)
            np.save(os.path.join(metric_folder, f"mf_acc_nocs.npy"), acc_nocs)

        # save metrics
        np.save(os.path.join(metric_folder, f"traj_results.npy"), all_results)

        del dataset, ssl_dl
        torch.cuda.empty_cache()

        if use_torch_profile:
            break

    return


def test_on_single_traj(
    fabric,
    pipeline: Pipeline,
    train_dataloader,
    objects_info,
    object_pc_ds,
    start_iter=0,
    log_every_iter=5,
    metric_dump_path="artifacts",
    vis_dump_path="visualizations",
    hparams=None,
    torch_profile=False,
    visualize=False,
    dump_vis=False,
    dump_video=False,
    profile_lengths=1,
):
    """Run this for each trajectory"""
    all_results = []

    prof = None
    warmup = float("inf")
    if torch_profile:
        print("Use PyTorch profiler")
        warmup = 1
        schdule = torch.profiler.schedule(wait=0, warmup=warmup, active=profile_lengths)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schdule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces/no_grad_ransac"),
            record_shapes=False,
            profile_memory=False,
        )
        prof.start()

    for i, (rgbs, masks, depths, intrinsics, world_T_cam, gt_objs_data, frames_info) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        with record_function("pipeline.forward"):
            payload = pipeline.forward(rgbs, masks, depths, intrinsics, gt_objs_data, frames_info)

        if visualize:
            # visualize rgb
            print("Visualize RGB and mask...")
            imgs_show(rgbs)
            imgs_show(masks)

        if dump_video:
            print("Dump video")
            # generate meshes
            pred_meshes = get_meshes_from_shape_codes(pipeline.model, payload["shape_code"], mesh_recons_scale=0.35)

            # transform meshes to camera frame
            transformed_meshes = []
            for ii, m in enumerate(pred_meshes):
                mesh = copy.deepcopy(m)
                s = payload["cam_s_nocs"][ii, ...].item()
                R = payload["cam_R_nocs"][ii, :3, :3].detach().cpu().numpy()
                t = (
                    payload["cam_t_nocs"][ii, ...]
                    .reshape(
                        3,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                mesh.vertices = np.transpose(s * R @ np.transpose(mesh.vertices) + t.reshape(3, 1))
                transformed_meshes.append(mesh)

            # generate video
            def cam_traj_fn(i, pl):
                if i == 0:
                    pl.camera_position = "xy"
                    pl.camera.azimuth = 10
                    pl.camera.roll = 180
                    pl.camera.elevation = 30
                else:
                    pl.camera.azimuth = 10 + i * 2
                    pl.camera.roll = 180 - i * 0.025
                    pl.camera.elevation = 30 - i * 0.25

            # create a folder with frame name
            # generate video
            assert rgbs.shape[0] == 1
            frame_info = frames_info[0]
            folder_name = f"artifacts/video_dump_{frame_info['scene_id']:06d}_{frame_info['frame_id']:06d}"
            safely_make_folders([folder_name])
            video_path = os.path.join(folder_name, f"mesh_video.mp4")
            generate_video_from_meshes(transformed_meshes, cam_traj_fn, 100, video_path)

            # dump rgb
            rgb_img_cpu = rgbs[0, ...].cpu()
            rgb_pil = F.to_pil_image(rgb_img_cpu)
            rgb_path = os.path.join(folder_name, f"rgb_{frame_info['frame_id']:06d}.jpg")
            rgb_pil.save(rgb_path)

            # dump rgb with mask
            mask_pil = F.to_pil_image(masks[0, ...].int() * 255).convert("1")
            # invert_mask = PIL.ImageOps.invert(mask_pil)
            rgb_masked = Image.blend(mask_pil.convert("RGB"), rgb_pil, 0.5)
            rgb_masked_path = os.path.join(folder_name, f"rgb_masked_{frame_info['frame_id']:06d}.jpg")
            rgb_masked.save(rgb_masked_path)

        if dump_vis:
            print("dump vis")
            # save mesh
            bs = payload["cam_s_nocs"].shape[0]
            mesh_screenshot_names = []
            mesh_export_names = []
            dump_path = os.path.join(vis_dump_path, "vis_dump")
            safely_make_folders([dump_path])
            for i in range(bs):
                c_frame_idx = payload["frame_index"][i]
                c_obj_idx = payload["obj_index"][i]
                c_obj_info = gt_objs_data[c_frame_idx][c_obj_idx]
                label = c_obj_info["label"]
                frame_info = frames_info[c_frame_idx]
                name = f"mesh_{frame_info['frame_id']}_{frame_info['view_id']}_{label}.jpg"
                mesh_name = f"mesh_{frame_info['frame_id']}_{frame_info['view_id']}_{label}.ply"
                mesh_export_names.append(os.path.join(dump_path, str(frame_info["scene_id"]), "mesh", mesh_name))
                mesh_screenshot_names.append(os.path.join(dump_path, str(frame_info["scene_id"]), "mesh", name))
            pred_meshes = save_screenshots_from_shape_codes(
                pipeline.model, payload["shape_code"], mesh_screenshot_names, mesh_export_names
            )

            print("dump nocs vis")
            save_joint_visualizations(
                pred_meshes, payload, rgbs, masks, gt_objs_data, frames_info, intrinsics, export_folder=dump_path
            )

            del pred_meshes
            torch.cuda.empty_cache()

        ## add odometry: view_id is the unique identifier to the pose node in the factor graph
        # with torch.no_grad():
        #    odoms = poses2odom(world_T_cam, index_mapping_fn=lambda x: frames_info[x]["view_id"])
        #    pipeline.update_pgo_odometry(odometry=odoms)

        #    # add objects
        #    obj_det_view_ids = [
        #        frames_info[payload["frame_index"][x]]["view_id"] for x in range(len(payload["frame_index"]))
        #    ]
        #    pipeline.update_pgo_objs(
        #        object_labels=[
        #            gt_objs_data[fid][x]["label"] for fid, x in zip(payload["frame_index"], payload["obj_index"])
        #        ],
        #        cam_R_obj=payload["cam_R_nocs"],
        #        cam_t_obj=payload["cam_t_nocs"],
        #        index2frame_fn=lambda x: obj_det_view_ids[x],
        #    )

        # gt error metrics
        # shape_results = shape_metrics(
        #     pipeline,
        #     gt_objs_data,
        #     depths,
        #     masks,
        #     intrinsics,
        #     objects_info,
        #     object_pc_ds,
        #     payload,
        #     normalized_recons=True,
        #     vis_transformed_cad=False,
        # )
        # breakpoint()
        # results = error_metrics(
        #     gt_objs_data=gt_objs_data,
        #     depths=depths,
        #     masks=masks,
        #     intrinsics=intrinsics,
        #     objects_info=objects_info,
        #     object_pc_ds=object_pc_ds,
        #     payload=payload,
        #     normalized_recons=pipeline.normalized_recons,
        #     vis_transformed_cad=visualize
        # )
        results = error_metrics(
            gt_objs_data=gt_objs_data,
            depths=depths,
            masks=masks,
            intrinsics=intrinsics,
            objects_info=objects_info,
            object_pc_ds=object_pc_ds,
            payload=payload,
            model=pipeline.model,
            cube_scale=0.251,
            vis_pcs=False,
            use_ICP=False,
            normalized_recons=pipeline.normalized_recons,
            calculate_pred_recons_metrics=True,
            vis_transformed_cad=False,
        )
        if pipeline.output_degen_condition_number:
            results = [
                {
                    "ftf_min_eig": payload["sf_degen_conds"][i]["FTF_min_eig"],
                    "ftf_cond": payload["sf_degen_conds"][i]["FTF_cond"],
                    **v,
                }
                for i, v in enumerate(results)
            ]
            if "oc_score" in payload.keys():
                results = [{"oc_score": payload["oc_score"][i], **v} for i, v in enumerate(results)]
        if pipeline.profile_runtime:
            results.append({"sf_corrector_time": payload["sf_corrector_time"]})
            results.append({"mf_corrector_time": payload["mf_corrector_time"]})
            results.append({"model_inference_time": payload["model_inference_time"]})

        all_results.extend(results)

        # save metrics
        np.save(metric_dump_path, all_results)

        if torch_profile:
            prof.step()
            if i >= profile_lengths + warmup:
                break

        del rgbs, masks, depths, intrinsics, gt_objs_data, frames_info, payload
        torch.cuda.empty_cache()

    if torch_profile:
        prof.stop()

    return all_results


def test_corrector_on_single_traj_with_gt_nocs(
    pipeline: Pipeline,
    corrector: JointCorrector,
    train_dataloader,
    nocs_ds,
    objects_info,
    object_pc_ds,
    metric_dump_path="artifacts",
    visualize_precrt_results=False,
    visualize_postcrt_results=False,
):
    """Run this for each trajectory"""
    all_results = []

    for i, (rgbs, masks, depths, intrinsics, world_T_cam, gt_objs_data, frames_info) in tqdm(
        enumerate(train_dataloader), total=len(train_dataloader)
    ):
        assert len(gt_objs_data) == len(frames_info) == 1
        # pipeline forward pass does not use corrector
        payload = pipeline.forward(rgbs, masks, depths, intrinsics, gt_objs_data, frames_info)

        # get ground truth NOCS
        nocs_data = []
        for bid in range(len(frames_info)):
            for oid in range(len(gt_objs_data[bid])):
                c_nocs_data = nocs_ds.__getitem__(frames_info[bid]["frame_id"], obj_idx=oid)
                nocs_data.append({"gt_nocs": c_nocs_data["nocs"]})

        # run preprocessor to get object-wise GT NOCS
        # gt_nocs_processed = pipeline._preprocess(
        #    rgbs, masks, depths, intrinsics, gt_objs_data, frames_info, additional_data=nocs_data
        # )[-1]
        gt_nocs = torch.stack([entry["gt_nocs"] for entry in nocs_data]).squeeze(1).cpu()

        # sampled nocs
        B = payload["precrt_nocs"].shape[0]
        sampled_indices = payload["sampled_indices"].cpu()
        sampled_gt_nocs = torch.gather(gt_nocs.view(B, 3, -1), 2, sampled_indices.unsqueeze(1).expand((B, 3, -1)))

        if visualize_precrt_results:
            print("Visualize pred NOCS (red) and GT NOCS (blue) and GT PC (green)")
            for i in range(B):
                print(f"Obj index = {i}")
                if i < 3:
                    continue
                obj_label = gt_objs_data[0][i]["label"]
                gt_recons_pc = nocs_ds.unified_objects.distinct_objects["ycbv"][obj_label]["surface_points"]

                visualize_pcs_pyvista(
                    [
                        payload["precrt_nocs"][i, ...].detach(),
                        sampled_gt_nocs[i, ...].detach(),
                        gt_recons_pc.detach(),
                    ],
                    colors=["crimson", "blue", "green"],
                    pt_sizes=[10.0, 10.0, 5.0],
                )

                # shape + nocs
                visualize_sdf_slices_pyvista(
                    payload["shape_code"][i, ...].unsqueeze(0),
                    pipeline.model.recons_net,
                    cube_scale=0.251,
                    additional_meshes=[gt_recons_pc.detach()],
                )

        for i in range(B):
            if i < 3:
                continue
            # run corrector with ground truth NOCS as input
            results = corrector.solve_inv_depths_sdf_input_nocs_scale_free(
                nocs=sampled_gt_nocs[i, ...].detach().to(pipeline.device).unsqueeze(0),
                shape_code=payload["shape_code"][i, ...].detach().unsqueeze(0),
                pcs=payload["pcs"][i, ...].detach().unsqueeze(0),
                masks=payload["masks"][i, ...].unsqueeze(0),
                registration_inlier_thres=payload["reg_inlier_thres"][i, ...].detach().unsqueeze(0),
                loss_multiplier=100,
                loss_type="sdf-input-nocs-no-scale",
            )
            postcrt_nocs = payload["precrt_nocs"] + results["nocs_correction"]
            postcrt_shape_code = payload["shape_code"] + results["shape_correction"]

            if visualize_postcrt_results:
                print("Visualize post-correction NOCS (red) and GT PC (green) and shape")
                print(f"Obj index = {i}")
                obj_label = gt_objs_data[0][i]["label"]
                gt_recons_pc = nocs_ds.unified_objects.distinct_objects["ycbv"][obj_label]["surface_points"]

                visualize_pcs_pyvista(
                    [
                        postcrt_nocs[i, ...].detach(),
                        sampled_gt_nocs[i, ...].detach(),
                        gt_recons_pc.detach(),
                    ],
                    colors=["crimson", "blue", "green"],
                    pt_sizes=[10.0, 10.0, 5.0],
                )

                # shape + nocs
                visualize_sdf_slices_pyvista(
                    postcrt_shape_code[i, ...].unsqueeze(0),
                    pipeline.model.recons_net,
                    cube_scale=0.251,
                    additional_meshes=[gt_recons_pc.detach(), sampled_gt_nocs[i, ...].detach()],
                )

        eval_results = error_metrics(
            gt_objs_data=gt_objs_data,
            depths=depths,
            masks=masks,
            intrinsics=intrinsics,
            objects_info=objects_info,
            object_pc_ds=object_pc_ds,
            payload=payload,
            model=pipeline.model,
            cube_scale=0.251,
            vis_pcs=True,
            use_ICP=False,
            normalized_recons=pipeline.normalized_recons,
            calculate_pred_recons_metrics=True,
            vis_transformed_cad=False,
        )
        all_results.extend(results)

        # save metrics
        np.save(metric_dump_path, all_results)

    return all_results
