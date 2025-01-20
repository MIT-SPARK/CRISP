import os
import dataclasses
import numpy as np
import torch
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser
import lightning as L
import time
import lightning.fabric as LF
import datetime
from tqdm import tqdm
import torch.utils.data as tchdata
import pyvista as pv
import seaborn

# local lib imports
import crisp.utils as utils
from crisp.datasets.bop_nocs import BOPNOCSDataset
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.registration import align_nocs_to_depth
from crisp.models.loss_functions import implicit_loss_helper, explicit_loss_helper
from crisp.utils.sdf import create_sdf_samples_generic
from crisp.models.loss_functions import nocs_loss, nocs_loss_clamped, siren_udf_loss, siren_sdf_fast_loss, metric_sdf_loss
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.utils.math import se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista
from crisp.utils.evaluation_metrics import rotation_error, translation_error
from crisp.utils import diff_operators
import crisp.utils.sdf

from experiments.unified_model.dataset_checks import single_batch_sanity_test


def train(
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
):
    """Main train loop. Make sure to run setup_dataloaders on dataloaders before passing them."""
    on_surface_cutoff = train_dataloader.dataset.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = train_dataloader.dataset.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = train_dataloader.dataset.dataset.unified_objects.sample_global_nonmnfld_points_count

    model.train()
    best_val_loss = torch.tensor(np.inf)
    for epoch in range(num_epochs):
        print(f"Training Epoch = {epoch} out of {num_epochs}.")
        start_time = time.time()
        avg_train_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
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
            nocs_map, sdf, _ = model(img=rgb_img, mask=mask, coords=coords)
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
            fabric.clip_gradients(model, optimizer, clip_val=0.7)
            optimizer.step()
            avg_train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        end_time = time.time()

        del rgb_img, coords, gt_normals, segmap, gt_nocs
        if epoch % validate_per_n_epochs == 0:
            print(f"epoch: {epoch}, loss: {loss}, per epoch time: {(end_time - start_time) / validate_per_n_epochs}")
            val_loss = validate(fabric, model, val_dataloader, loss_fn)
            fabric.log("Val/total_loss", val_loss, step=epoch)
            if val_loss < best_val_loss:
                state = {"model": model, "optimizer": optimizer, "epoch": epoch, "hparams": hparams}
                print(f"Current val loss = {val_loss} < Best val loss = {best_val_loss}. Saving model.")
                fabric.save(os.path.join(model_save_path, "checkpoint.pth"), state)
                best_val_loss = val_loss


def validate(fabric, model, dataloader, loss_fn):
    torch.cuda.empty_cache()
    on_surface_cutoff = dataloader.dataset.dataset.unified_objects.sample_surface_points_count
    global_nonmnfld_start = dataloader.dataset.dataset.unified_objects.global_nonmnfld_points_start
    global_nonmnfld_count = dataloader.dataset.dataset.unified_objects.sample_global_nonmnfld_points_count

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
        nocs_map, sdf, _ = model(img=rgb_img, mask=mask, coords=coords)
        gradient = diff_operators.gradient(sdf, coords)
        loss, nocs_l, recons_l, recons_terms = loss_fn(
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
        )

        total_loss += loss.cpu().float().item()
        del loss, nocs_l, recons_l, recons_terms
        torch.cuda.empty_cache()

    total_loss /= len(dataloader)
    fabric.print(f"Avg. validation loss across {i} batches: {total_loss}")
    return total_loss


def test(
    fabric,
    model,
    dataloader,
    loss_fn,
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
        for i, batch in tqdm(enumerate(dataloader)):
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
            nocs_map, sdf, _ = model(img=rgb_img, mask=mask, coords=coords)

            if vis_sdf_sample_points:
                visualize_pcs_pyvista(
                    [coords[0, ...], object_pc[0, ...]], colors=["crimson", "lightblue"], pt_sizes=[6.0, 2.0]
                )

            if vis_gt_sanity_test:
                single_batch_sanity_test(
                    mask=mask,
                    gt_nocs=gt_nocs,
                    object_pc=object_pc,
                    depth=depth,
                    cam_intrinsics=cam_intrinsics,
                    gt_world_T_cam=gt_world_T_cam,
                    vis_gt_nocs_registration=True,
                    vis_gt_nocs_and_cad=True,
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
            )
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
                print("Visualizing model predicted NOCS with object points.")
                for j in range(bs):
                    # sanity checks & tests for each instance in batch
                    # retrieve original CAD points
                    # compare the original CAD coords and NOCS points
                    # cad points from NOCS
                    nocs_pts = (nocs_map[j, :3, ...] - 0.5) * 2
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
                    print("Visualize transformed CAD with depth points.")
                    depth_pts, idxs = instance_depth_to_point_cloud_torch(
                        depth[j, ...], cam_intrinsics[j, ...], mask[j, ...].squeeze(0)
                    )
                    # NOCS transformed to depth
                    nocs_pts = (nocs_map[j, :3, idxs[0], idxs[1]] - 0.5) * 2
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
                batched_shape_code = model.forward_shape_code(img=rgb_img)
                for j in range(bs):
                    shape_code = batched_shape_code[0, ...]
                    objkey = str(metadata["dataset_name"][j]) + "-" + str(metadata["obj_name"][j])
                    objcounter[objkey] += 1
                    obj2shpcode[objkey].append(shape_code)

                    if vis_pred_recons or vis_pred_sdf or export_all_pred_recons_mesh:

                        def model_fn(coords):
                            return model.recons_net.forward(shape_code=shape_code.unsqueeze(0), coords=coords)

                        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
                            model_fn=model_fn, N=256, max_batch=64**3, cube_center=np.array([0, 0, 0]), cube_scale=2.5
                        )
                        pred_mesh = utils.sdf.convert_sdf_samples_to_mesh(
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
                                [pred_mesh, mesh_sdf_slices],
                                mesh_args=[{"opacity": 0.2, "color": "white"}, None],
                            )

                        if vis_pred_recons:
                            print("Visualize reconstruction...")
                            print(f"Showing {metadata['dataset_name']}-{metadata['obj_name']}")
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
                pred_mesh = utils.sdf.convert_sdf_samples_to_mesh(
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
    ycbv_data_dir: str = None
    shapenet_data_dir: str = None
    bop_data_dir: str = None
    replicacad_data_dir: str = None
    model_ckpts_save_dir: str = "./model_ckpts"
    preload_to_mem: bool = False
    dataloader_workers: int = 1

    # dataset
    pc_size: int = 60000
    per_batch_sample_surface_points_count: int = 1500
    per_batch_sample_local_nonmnfld_points_count: int = 1500
    per_batch_sample_global_nonmnfld_points_count: int = 4000
    global_nonmnfld_voxel_res: int = 128
    dataset_debug_vis: bool = False
    normalized_recons: bool = True

    # backbone model
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    freeze_pretrained_backbone_weights: bool = True
    log_root_dir: str = "logs"
    backbone_input_res: tuple = (420, 420)

    # implicit recons model
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"
    recons_loss_type: str = "sdf"
    recons_df_loss_mode: str = "metric"

    # nocs model
    nocs_network_type: str = "dpt"
    nocs_lateral_layers_type: str = "spaced"

    # training
    loss_nocs_threshold: float = 0.1
    num_epochs: int = 50
    validate_per_n_epochs: int = 5
    batch_size: int = 5
    lr: float = 3e-4
    nocs_lr: float = 3e-4
    recons_lr: float = 3e-4
    weight_decay: float = 1e-5

    # distributed
    num_devices: int = 1
    num_nodes: int = 1

    # overall loss
    nocs_loss_weight: float = 1.0
    recons_loss_weight: float = 1.0
    nocs_min: float = float("-inf")
    nocs_max: float = float("inf")

    # loss for recons module
    recons_df_weight: float = 3e3
    recons_inter_weight: float = 2e2
    recons_normal_weight: float = 0
    recons_grad_weight: float = 5e1

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

    # automatically populated if missing
    exp_id: str = None


def main(opt: ExpSettings):
    LF.seed_everything(42)
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    artifacts_dump_path = os.path.join(opt.artifacts_save_dir, opt.exp_id)

    # one logger to log root dir, another to the exp folder
    tb_logger_root = LF.loggers.TensorBoardLogger(
        root_dir=os.path.join(opt.log_root_dir, "tensorboard"), name="joint_model_training", flush_secs=10, max_queue=5
    )
    tb_logger_exp = LF.loggers.TensorBoardLogger(root_dir=exp_dump_path, name="log", flush_secs=10, max_queue=5)
    fabric = L.Fabric(
        accelerator="gpu",
        loggers=[tb_logger_root, tb_logger_exp],
        strategy="ddp",
        devices=opt.num_devices,
        num_nodes=opt.num_nodes,
    )
    fabric.launch()

    # dataloaders
    shape_ds = BOPNOCSDataset(
        ds_name="ycbv",
        split="train_real",
        bop_ds_dir=opt.bop_data_dir,
        unified_objects_dataset_path=opt.dataset_dir,
        preload_to_mem=opt.preload_to_mem,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        sample_bounds=(-1.0, 1.0),
        pc_size=opt.pc_size,
        input_H=opt.backbone_input_res[0],
        input_W=opt.backbone_input_res[1],
        normalized_recons=opt.normalized_recons,
        debug_vis=opt.dataset_debug_vis,
    )

    shape_test_ds = BOPNOCSDataset(
        ds_name="ycbv",
        split="test",
        bop_ds_dir=opt.bop_data_dir,
        unified_objects_dataset_path=opt.dataset_dir,
        preload_to_mem=opt.preload_to_mem,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        sample_bounds=(-1.0, 1.0),
        pc_size=opt.pc_size,
        input_H=opt.backbone_input_res[0],
        input_W=opt.backbone_input_res[1],
        normalized_recons=opt.normalized_recons,
        debug_vis=opt.dataset_debug_vis,
    )
    val_dl, train_dl = [
        tchdata.DataLoader(x, shuffle=True, num_workers=opt.dataloader_workers, batch_size=opt.batch_size)
        for x in tchdata.random_split(shape_ds, [0.1, 0.9])
    ]
    test_dl = tchdata.DataLoader(
        shape_test_ds, shuffle=True, num_workers=opt.dataloader_workers, batch_size=opt.batch_size
    )

    test_dl, val_dl, train_dl = fabric.setup_dataloaders(test_dl, val_dl, train_dl)

    # joint nocs & recons model
    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        freeze_pretrained_weights=opt.freeze_pretrained_backbone_weights,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
        lateral_layers_type=opt.nocs_lateral_layers_type,
        backbone_input_res=opt.backbone_input_res,
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

    # loss closure to modify behavior & hyperparams of loss functions
    if opt.recons_loss_type == "sdf":
        recons_loss_fn = siren_sdf_fast_loss
    elif opt.recons_loss_type == "udf":
        recons_loss_fn = siren_udf_loss

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
            scheduler=None,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            loss_fn=loss_fn,
            num_epochs=opt.num_epochs,
            validate_per_n_epochs=opt.validate_per_n_epochs,
            model_save_path=exp_dump_path,
            hparams=hparams,
        )

    test(
        fabric,
        model,
        test_dl,
        loss_fn=None,
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
