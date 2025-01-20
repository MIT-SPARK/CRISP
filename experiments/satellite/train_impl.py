import time
from tqdm import tqdm

import PIL
from PIL import Image
import PIL.ImageOps
from torchvision.transforms import functional as F

# local lib imports
import crisp.models.pipeline
from crisp.models.corrector import *
import crisp.models.loss_functions as losses
from crisp.models.joint import apply_lipschitz_constraint
from crisp.utils.visualization_utils import (
    visualize_sdf_slices_pyvista,
    get_meshes_from_shape_codes,
    generate_video_from_meshes,
    generate_orbiting_video_from_meshes,
)
from crisp.utils.math import se3_inverse_batched_torch, make_se3_batched
from crisp.utils import diff_operators

from experiments.satellite.spe3r_utils import calculate_metrics


def run_pipeline(
    fabric,
    pipeline: crisp.models.pipeline.Pipeline,
    optimizers,
    train_dataloader,
    num_epochs,
    visualize=False,
    save_every_epoch=1,
    model_save_path="./model_ckpts",
    artifacts_save_dir="./artifacts",
    ssl_train=True,
    ssl_type="v1",
    calculate_test_metrics=False,
    calculate_test_recons_metrics=False,
    dump_video=False,
    opt=None,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    on_surface_cutoff = train_dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

    pipeline.train()
    best_val_loss = torch.tensor(np.inf)
    ssl_nocs_iter, ssl_shape_iter = 0, 0
    all_metrics = []
    if dump_video:
        already_exported = {}
    for epoch in range(num_epochs):
        print(f"Training Epoch = {epoch} out of {num_epochs}.")
        start_time = time.time()
        best_cert_percent = 0
        avg_train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            c_iter = epoch * len(train_dataloader) + i
            # nocs related
            rgb_img, segmap, depth, K, model_name = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["depth"],
                batch["cam_intrinsics"].unsqueeze(1),
                batch["model_name"],
            )
            B = rgb_img.shape[0]
            frames_info = [{"model_name": model_name[ii]} for ii in range(rgb_img.shape[0])]

            # one object per image
            objs_info = [[{"bbox": None, "id_in_segm": 1, "label": model_name[ii]}] for ii in range(rgb_img.shape[0])]

            # the mask should be already for the correct object
            mask = segmap.unsqueeze(1)

            # forward pass
            payload = pipeline(
                rgbs=rgb_img, masks=mask, depths=depth, intrinsics=K, objs=objs_info, frames_info=frames_info
            )

            if dump_video:
                assert rgb_img.shape[0] == 1
                frame_info = frames_info[0]
                mname = frame_info["model_name"]

                print("Dump video")
                if mname in already_exported.keys():
                    if already_exported[mname] > 5:
                        continue
                    else:
                        already_exported[mname] += 1
                else:
                    already_exported[mname] = 1

                # generate meshes
                pred_meshes = get_meshes_from_shape_codes(pipeline.model, payload["shape_code"], mesh_recons_scale=1.2)

                # create a folder with frame name
                # generate video
                folder_name = f"video_dump/video_dump_{frame_info['model_name']}_{i:06d}"
                safely_make_folders([folder_name])
                video_path = os.path.join(folder_name, f"mesh_video.mp4")
                # generate_video_from_meshes(pred_meshes, cam_traj_fn, 100, video_path)
                generate_orbiting_video_from_meshes(pred_meshes, video_path, viewup=[0, 0, 1], n_points=50, shift=1)

                # dump rgb
                rgb_img_cpu = rgb_img[0, ...].cpu()
                rgb_img_cpu[0, ...] = rgb_img_cpu[0, ...] * 0.229 + 0.485
                rgb_img_cpu[1, ...] = rgb_img_cpu[1, ...] * 0.224 + 0.456
                rgb_img_cpu[2, ...] = rgb_img_cpu[2, ...] * 0.225 + 0.406
                rgb_pil = F.to_pil_image(rgb_img_cpu)
                rgb_path = os.path.join(folder_name, f"rgb_{frame_info['model_name']}_{i:06d}.jpg")
                rgb_pil.save(rgb_path)

                # dump rgb with mask
                mask_pil = F.to_pil_image(mask[0, ...].int() * 255).convert("1")
                # invert_mask = PIL.ImageOps.invert(mask_pil)
                rgb_masked = Image.blend(mask_pil.convert("RGB"), rgb_pil, 0.5)
                rgb_masked_path = os.path.join(folder_name, f"rgb_masked_{frame_info['model_name']}_{i:06d}.jpg")
                rgb_masked.save(rgb_masked_path)

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

                for iii in range(B):
                    c_metrics[iii]["cert_flag"] = payload["cert_mask"][iii]
                    if pipeline.output_degen_condition_number:
                        c_metrics[iii]["ftf_min_eig"] = payload["sf_degen_conds"][iii]["FTF_min_eig"]
                        c_metrics[iii]["ftf_cond"] = payload["sf_degen_conds"][iii]["FTF_cond"]

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

            cert_percent = torch.sum(payload["cert_mask"]) / payload["cert_mask"].shape[0]
            fabric.log("Train/cert_percent", cert_percent, step=c_iter)
            print(
                f"c_iter={c_iter}, SSL_nocs_iter={ssl_nocs_iter}, SSL_recons_iter={ssl_shape_iter}, cert%={cert_percent}, best cert%={best_cert_percent}"
            )
            if best_cert_percent < cert_percent:
                best_cert_percent = cert_percent

            # self-supervised backprop if we have enough certified samples
            if ssl_train:
                if ssl_type == "v1":
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

                elif ssl_type == "v2":
                    # directly backprop on the corrector loss
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
                    )
                    l2 = losses.sdf_nocs_loss(
                        f_sdf_conditioned=f_sdf_conditioned,
                        nocs=payload["postcrt_nocs"],
                        weights=payload["masks"],
                    )
                    l = opt.ssl_sdf_input_weight * l1 + opt.ssl_sdf_nocs_weight * l2
                    fabric.backward(l)
                    for optim in optimizers.values():
                        optim.step()

                    fabric.log(f"Train/total_loss", l.item(), step=c_iter)
                    fabric.log(f"Train/sdf_input_loss", l1.item(), step=c_iter)
                    fabric.log(f"Train/sdf_nocs_loss", l2.item(), step=c_iter)
                else:
                    raise ValueError(f"Unknown SSL type: {ssl_type}")

        if ssl_train:
            state = {
                "model": pipeline.model,
                "optimizers": optimizers,
                "epoch": epoch,
                "ssl-nocs-iter": ssl_nocs_iter,
                "ssl-shape-iter": ssl_shape_iter,
            }
            fabric.save(os.path.join(model_save_path, f"checkpoint.pth"), state)

            if epoch % save_every_epoch == 0:
                print(f"Saving model at epoch={epoch}.")
                fabric.save(os.path.join(model_save_path, f"checkpoint_epoch={epoch}.pth"), state)

            del all_ssl_loss

        end_time = time.time()
        del payload
        torch.cuda.empty_cache()

    # save the final result
    if ssl_train:
        state = {
            "model": pipeline.model,
            "optimizers": optimizers,
            "epoch": epoch,
            "ssl-nocs-iter": ssl_nocs_iter,
            "ssl-shape-iter": ssl_shape_iter,
        }
        print(f"Saving final model at epoch={epoch}.")
        fabric.save(os.path.join(model_save_path, f"checkpoint.pth"), state)

    return all_metrics


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
    log_every_iter=5,
    model_save_path="./model_ckpts",
    hparams=None,
    opt=None,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    on_surface_cutoff = train_dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

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
            c_iter = epoch * len(train_dataloader) + i
            # nocs related
            rgb_img, segmap, gt_nocs, coords, gt_normals, gt_sdf = (
                batch["rgb"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["coords"],
                batch["normals"],
                batch["sdf"],
            )
            optimizer.zero_grad()

            # the mask should be already for the correct object
            mask = segmap.unsqueeze(1)

            # forward pass
            coords = coords.clone().detach().requires_grad_(True)
            nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)
            gradient = diff_operators.gradient(sdf, coords)
            loss_terms = loss_fn(
                pred_sdf=sdf,
                gradient=gradient,
                gt_normals=gt_normals,
                on_surface_cutoff=on_surface_cutoff,
                global_nonmnfld_start=global_nonmnfld_start,
                global_nonmnfld_count=global_nonmnfld_count,
                pred_nocs=nocs_map,
                gt_nocs=gt_nocs,
                gt_sdf=gt_sdf,
                mask=mask,
                shape_code=shape_code,
                contrastive_same_object_weight=ctr_same_cls_weight,
                contrastive_different_object_weight=ctr_diff_cls_weight,
            )
            loss = loss_terms["total_loss"]

            if i % log_every_iter == 0:
                nocs_l, recons_l, recons_terms = (
                    loss_terms["nocs_loss"],
                    loss_terms["recons_loss"],
                    loss_terms["recons_loss_terms"],
                )
                fabric.log("Train/total_loss", loss.item(), step=c_iter)
                fabric.log("Train/nocs_loss/total", nocs_l.item(), step=c_iter)
                fabric.log("Train/recons_loss/total", recons_l.item(), step=c_iter)
                if "same_cls_loss" in loss_terms.keys():
                    fabric.log("Train/same_cls_loss/total", loss_terms["same_cls_loss"].item(), step=c_iter)
                if "diff_cls_loss" in loss_terms.keys():
                    fabric.log("Train/diff_cls_loss/total", loss_terms["diff_cls_loss"].item(), step=c_iter)

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

            ## log gradients
            # total_norm = 0
            # parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            # for p in parameters:
            #    param_norm = p.grad.detach().data.norm(2)
            #    total_norm += param_norm.item() ** 2
            # total_norm = total_norm**0.5
            # fabric.log("Train/grad_norm", total_norm, step=c_iter)

            fabric.clip_gradients(model, optimizer, clip_val=1.0)

            # total_norm = 0
            # parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            # for p in parameters:
            #    param_norm = p.grad.detach().data.norm(2)
            #    total_norm += param_norm.item() ** 2
            # total_norm = total_norm**0.5
            # fabric.log("Train/grad_norm_after_clip", total_norm, step=c_iter)

            optimizer.step()
            avg_train_loss += loss.item()

            if opt is not None and opt.recons_lipschitz_normalization_type is not None:
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

        del rgb_img, coords, gt_normals, segmap, gt_nocs
        torch.cuda.empty_cache()

        if epoch % validate_per_n_epochs == 0:
            print(f"epoch: {epoch}, loss: {loss}, per epoch time: {(end_time - start_time) / validate_per_n_epochs}")
            val_loss = validate_sl(fabric, model, val_dataloader, loss_fn)
            fabric.log("Val/total_loss", val_loss, step=epoch)
            if val_loss < best_val_loss:
                state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
                print(f"Current val loss = {val_loss} < Best val loss = {best_val_loss}. Saving model.")
                fabric.save(os.path.join(model_save_path, "checkpoint_best_val.pth"), state)
                best_val_loss = val_loss

        # save per epoch model
        state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
        fabric.save(os.path.join(model_save_path, f"checkpoint_epoch={epoch}.pth"), state)
        fabric.save(os.path.join(model_save_path, f"checkpoint.pth"), state)


def train_ssl(
    fabric,
    pipeline,
    optimizers,
    scheduler,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    visualize=False,
    validate_per_n_epochs=50,
    save_every_epoch=1,
    model_save_path="./model_ckpts",
    hparams=None,
    opt=None,
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    run_pipeline(
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


def validate_sl(fabric, model, dataloader, loss_fn):
    torch.cuda.empty_cache()
    on_surface_cutoff = dataloader.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = dataloader.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = dataloader.dataset.unified_objects.sample_global_nonmnfld_points_count

    model.eval()
    total_loss = 0
    fabric.print("Validation...")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # nocs related
        rgb_img, segmap, gt_nocs, coords, gt_normals, gt_sdf = (
            batch["rgb"],
            batch["instance_segmap"],
            batch["nocs"],
            batch["coords"],
            batch["normals"],
            batch["sdf"],
        )

        # only 1 object per image in synthetic training set
        mask = (segmap == 1).unsqueeze(1)

        # forward pass
        coords = coords.clone().detach().requires_grad_(True)
        nocs_map, sdf, shape_code = model(img=rgb_img, mask=mask, coords=coords)
        gradient = diff_operators.gradient(sdf, coords)
        loss_terms = loss_fn(
            pred_sdf=sdf,
            gradient=gradient,
            gt_normals=gt_normals,
            on_surface_cutoff=on_surface_cutoff,
            global_nonmnfld_start=global_nonmnfld_start,
            global_nonmnfld_count=global_nonmnfld_count,
            pred_nocs=nocs_map,
            gt_nocs=gt_nocs,
            gt_sdf=gt_sdf,
            shape_code=shape_code,
            mask=mask,
        )
        loss = loss_terms["total_loss"]

        total_loss += loss.cpu().float().item()
        del loss, loss_terms
        torch.cuda.empty_cache()

    total_loss /= len(dataloader)
    fabric.print(f"Avg. validation loss across {i} batches: {total_loss}")
    return total_loss
