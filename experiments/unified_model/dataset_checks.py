from dataclasses import dataclass
from jsonargparse import ArgumentParser
import lightning as L
import lightning.fabric as LF
from tqdm import tqdm
from typing import Optional

import torch
import torch.utils.data as tchdata
import numpy as np
import pyvista as pv
import trimesh
import matplotlib.pyplot as plt

# local lib imports
from crisp.datasets.unified_objects import UnifiedObjects, unified_objects_collate_fn
from crisp.models.nocs import *
from crisp.models.loss_functions import nocs_loss
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.models.registration import align_nocs_to_depth
from crisp.utils.math import se3_inverse_torch, se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.visualization_utils import imgs_show
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista
from crisp.utils.evaluation_metrics import rotation_error, translation_error
from crisp.utils import diff_operators


def single_batch_sanity_test(
    mask,
    gt_nocs,
    sdf_grid,
    object_pc,
    normalized_mesh,
    depth,
    cam_intrinsics,
    gt_world_T_cam,
    voxel_res,
    voxel_origin,
    voxel_size,
    normalized_recons=True,
    vis_gt_nocs_registration=True,
    vis_gt_nocs_and_cad=True,
    vis_sdf_and_mesh=True,
):
    """ """
    bs = mask.shape[0]
    # source: nocs
    # target: depth map point cloud
    # NOCS should be in blender world frame (after recenter)
    # depth is in blender camera frame
    s, cam_R_world, cam_t_cad, cam_Tsim_cad, _, _ = align_nocs_to_depth(
        masks=mask,
        nocs=gt_nocs,
        depth=depth.squeeze(1),
        intrinsics=cam_intrinsics,
        instance_ids=torch.arange(bs),
        img_path=None,
        normalized_nocs=normalized_recons,
        verbose=False,
    )

    # R_det = scale^3 * det(R) = scale^3
    R_det = torch.linalg.det(gt_world_T_cam[:, :3, :3])
    scale = torch.pow(R_det, 1 / 3)
    gt_world_T_cam[:, :3, :3] = gt_world_T_cam[:, :3, :3] / scale.reshape(bs, 1, 1)
    print(f"GT pose scale: {scale}, inv scale: {1 / scale}")

    gt_cam_T_world = se3_inverse_batched_torch(gt_world_T_cam)
    trans_err = torch.mean(translation_error(cam_t_cad.unsqueeze(-1), gt_cam_T_world[:, :3, -1].unsqueeze(-1)))
    rot_err = torch.mean(rotation_error(cam_R_world, gt_cam_T_world[..., :3, :3]))
    print(f"GT NOCS registration avg trans err: {trans_err}")
    print(f"GT NOCS registration avg rotation err: {rot_err}")
    print(f"GT NOCS scale: {s}")

    if vis_gt_nocs_registration:
        # imgs_show(rgb_img)
        for j in range(bs):
            print("Visualize transformed NOCS pts with depth points.")
            depth_pts, idxs = instance_depth_to_point_cloud_torch(depth[j, ...], cam_intrinsics[j, ...], mask[j, ...])
            if normalized_recons:
                nocs_pts = (gt_nocs[j, :3, idxs[0], idxs[1]] - 0.5) * 2
            else:
                nocs_pts = gt_nocs[j, :3, idxs[0], idxs[1]]

            cad_p = nocs_pts.reshape(3, -1)
            cam_p = cam_Tsim_cad[j, :3, :3] @ cad_p + cam_Tsim_cad[j, :3, 3].reshape(3, 1)
            pl = pv.Plotter(shape=(1, 1))
            transformed_cad_pc = pv.PolyData((cam_p.T).cpu().numpy())
            depth_pc = pv.PolyData((depth_pts.T).cpu().numpy())
            pl.add_mesh(transformed_cad_pc, color="lightblue", point_size=5.0, render_points_as_spheres=True)
            pl.add_mesh(depth_pc, color="crimson", point_size=5.0, render_points_as_spheres=True)
            pl.show_grid()
            pl.show_axes()
            pl.show()

    if vis_gt_nocs_and_cad:
        print(
            "Visualize GT NOCS points (blue) transformed back to normalized CAD frame,"
            " normalized object CAD points, and normalized mesh (red)"
        )
        for j in range(bs):
            # sanity checks & tests for each instance in batch
            # retrieve original CAD points
            # compare the original CAD coords and NOCS points
            # cad points from NOCS
            if normalized_recons:
                nocs_pts = (gt_nocs[j, :3, ...] - 0.5) * 2
            else:
                nocs_pts = gt_nocs[j, :3, ...]
            nocs_pts = nocs_pts.reshape(3, -1).float()

            # create the trimesh mesh
            mesh = None
            if normalized_mesh is not None:
                if isinstance(normalized_mesh, list):
                    mesh = trimesh.Trimesh(
                        vertices=normalized_mesh[j][0].numpy(force=True),
                        faces=normalized_mesh[j][1].numpy(force=True),
                    )
                else:
                    mesh = normalized_mesh

            # visualize NOCS and object mesh points
            pl = pv.Plotter(shape=(1, 1))
            nocs_pc = pv.PolyData((nocs_pts.T).cpu().numpy())
            og_coords_pc = pv.PolyData((object_pc[j, ...]).cpu().numpy())
            pl.add_mesh(nocs_pc, color="lightblue", point_size=10.0, render_points_as_spheres=True)
            pl.add_mesh(og_coords_pc, color="black", point_size=5.0, render_points_as_spheres=True)
            if mesh is not None:
                pl.add_mesh(mesh, color="crimson", opacity=0.4)
            pl.show_grid()
            pl.show_axes()
            pl.show()

    if vis_sdf_and_mesh:
        print("Visualize SDF and normalized mesh.")
        for j in range(bs):
            # create the trimesh mesh
            mesh = trimesh.Trimesh(
                vertices=normalized_mesh[j][0].numpy(force=True),
                faces=normalized_mesh[j][1].numpy(force=True),
            )

            instance_sdf = sdf_grid[j, ...].numpy(force=True)
            grid_sdf_values = instance_sdf.reshape(
                voxel_res,
                voxel_res,
                voxel_res,
            )
            mesh_sdf_slices = gen_pyvista_voxel_slices(grid_sdf_values, voxel_origin, (voxel_size,) * 3)

            visualize_meshes_pyvista(
                [mesh, mesh_sdf_slices],
                mesh_args=[{"opacity": 0.2, "color": "white"}, None],
            )

    return


def sanity_test(
    fabric, dataloader, vis_gt_nocs_and_cad=True, vis_gt_nocs_registration=True, vis_sdf_and_mesh=True, vis_rgb=True
):
    fabric.print("Testing...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            (
                rgb_img,
                depth,
                segmap,
                gt_nocs,
                object_pc,
                sdf_grid,
                normalized_mesh,
                cam_intrinsics,
                gt_world_T_cam,
                metadata,
            ) = (
                batch["rgb"],
                batch["depth"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["object_pc"],
                batch["sdf_grid"],
                batch["normalized_mesh"],
                batch["cam_intrinsics"],
                batch["cam_pose"],
                batch["metadata"],
            )
            mask = segmap == 1

            voxel_res = dataloader.dataset.global_nonmnfld_points_voxel_res
            cube_scale = dataloader.dataset.sample_bounds[1] - dataloader.dataset.sample_bounds[0]
            voxel_origin = np.array([0, 0, 0]) - cube_scale / 2
            voxel_size = cube_scale / (voxel_res - 1)

            if vis_rgb:
                print("Visualize RGB image.")
                imgs_show(rgb_img)

            single_batch_sanity_test(
                mask=mask,
                gt_nocs=gt_nocs,
                object_pc=object_pc,
                sdf_grid=sdf_grid,
                voxel_res=voxel_res,
                voxel_size=voxel_size,
                voxel_origin=voxel_origin,
                normalized_mesh=normalized_mesh,
                depth=depth,
                cam_intrinsics=cam_intrinsics,
                gt_world_T_cam=gt_world_T_cam,
                vis_gt_nocs_registration=vis_gt_nocs_registration,
                vis_gt_nocs_and_cad=vis_gt_nocs_and_cad,
                vis_sdf_and_mesh=vis_sdf_and_mesh,
                normalized_recons=dataloader.dataset.normalized_recons,
            )

            torch.cuda.empty_cache()


@dataclass
class ExpSettings:
    dataset_dir: str
    shapenet_data_dir: Optional[str] = None
    bop_data_dir: Optional[str] = None
    replicacad_data_dir: Optional[str] = None
    spe3r_data_dir: Optional[str] = None
    preload_to_mem: bool = False

    # dataset
    pc_size: int = 60000
    per_batch_sample_surface_points_count: int = 1500
    per_batch_sample_local_nonmnfld_points_count: int = 1500
    per_batch_sample_global_nonmnfld_points_count: int = 4000
    global_nonmnfld_voxel_res: int = 128
    dataset_debug_vis: bool = False
    normalized_recons: bool = True

    force_recompute_sdf: bool = False
    num_sdf_compute_workers: int = 1
    vis_gt_nocs_registration: bool = True
    vis_gt_nocs_and_cad: bool = True
    vis_sdf_and_mesh: bool = True
    vis_rgb: bool = False

    # automatically populated if missing
    exp_id: str = None


def main(opt: ExpSettings):
    LF.seed_everything(42)
    fabric = L.Fabric(accelerator="cuda", strategy="ddp", devices=[0])
    fabric.launch()

    # dataloaders
    shape_ds = UnifiedObjects(
        folder_path=opt.dataset_dir,
        shapenet_dataset_path=opt.shapenet_data_dir,
        bop_dataset_path=opt.bop_data_dir,
        replicacad_dataset_path=opt.replicacad_data_dir,
        spe3r_dataset_path=opt.spe3r_data_dir,
        preload_to_mem=opt.preload_to_mem,
        pc_size=opt.pc_size,
        sample_bounds=(-1.0, 1.0),
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        debug_vis=opt.dataset_debug_vis,
        force_recompute_sdf=opt.force_recompute_sdf,
        num_workers=opt.num_sdf_compute_workers,
        normalized_recons=opt.normalized_recons,
        data_to_output=[
            "rgb",
            "nocs",
            "depth",
            "coords",
            "sdf_grid",
            "normalized_mesh",
            "instance_segmap",
            "cam_intrinsics",
            "cam_pose",
            "object_pc",
            "metadata",
        ],
    )
    test_dl = tchdata.DataLoader(
        shape_ds, shuffle=False, num_workers=0, batch_size=5, collate_fn=unified_objects_collate_fn
    )
    test_dl = fabric.setup_dataloaders(test_dl)

    sanity_test(
        fabric,
        test_dl,
        vis_gt_nocs_and_cad=opt.vis_gt_nocs_and_cad,
        vis_gt_nocs_registration=opt.vis_gt_nocs_registration,
        vis_sdf_and_mesh=opt.vis_sdf_and_mesh,
        vis_rgb=opt.vis_rgb,
    )


if __name__ == "__main__":
    """Sanity checks for the generated unified dataset"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    main(ExpSettings(**opt))
