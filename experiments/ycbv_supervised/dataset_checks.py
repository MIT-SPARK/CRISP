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
from experiments.unified_model.dataset_checks import single_batch_sanity_test
from crisp.datasets.bop_nocs import BOPNOCSDataset
from crisp.datasets.unified_objects import unified_objects_collate_fn
from crisp.models.registration import align_nocs_to_depth
from crisp.utils.math import se3_inverse_torch, se3_inverse_batched_torch, instance_depth_to_point_cloud_torch
from crisp.utils.visualization_utils import imgs_show
from crisp.utils.visualization_utils import visualize_pcs_pyvista, gen_pyvista_voxel_slices, visualize_meshes_pyvista
from crisp.utils.evaluation_metrics import rotation_error, translation_error
from crisp.utils import diff_operators
import crisp.utils.sdf


def single_batch_sanity_test_bk(
    mask,
    gt_nocs,
    sdf,
    object_pc,
    normalized_mesh,
    depth,
    cam_intrinsics,
    gt_world_T_cam,
    voxel_res,
    voxel_origin,
    voxel_size,
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
        depth=depth,
        intrinsics=cam_intrinsics,
        instance_ids=torch.arange(bs),
        img_path=None,
        verbose=False,
    )

    gt_cam_T_world = se3_inverse_batched_torch(gt_world_T_cam)
    trans_err = torch.mean(translation_error(cam_t_cad.unsqueeze(-1), gt_cam_T_world[:, :3, -1].unsqueeze(-1)))
    rot_err = torch.mean(rotation_error(cam_R_world, gt_world_T_cam[..., :3, :3].permute((0, 2, 1))))
    print(f"GT NOCS registration avg trans err: {trans_err}")
    print(f"GT NOCS registration avg rotation err: {rot_err}")
    print(f"GT NOCS scale: {s}")

    if vis_gt_nocs_registration:
        for j in range(bs):
            print("Visualize transformed NOCS pts (lightblue) with depth points (red).")
            depth_pts, idxs = instance_depth_to_point_cloud_torch(depth[j, ...], cam_intrinsics[j, ...], mask[j, ...])
            nocs_pts = (gt_nocs[j, :3, idxs[0], idxs[1]] - 0.5) * 2
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
            "Visualize GT NOCS points transformed back to normalized CAD frame,"
            " normalized object CAD points, and normalized mesh"
        )
        for j in range(bs):
            # sanity checks & tests for each instance in batch
            # retrieve original CAD points
            # compare the original CAD coords and NOCS points
            # cad points from NOCS
            nocs_pts = (gt_nocs[j, :3, ...] - 0.5) * 2
            nocs_pts = nocs_pts.reshape(3, -1)

            # create the trimesh mesh
            mesh = trimesh.Trimesh(
                vertices=normalized_mesh[0][j, ...].numpy(force=True),
                faces=normalized_mesh[1][j, ...].numpy(force=True),
            )

            # visualize NOCS and object mesh points
            pl = pv.Plotter(shape=(1, 1))
            nocs_pc = pv.PolyData((nocs_pts.T).cpu().numpy())
            og_coords_pc = pv.PolyData((object_pc[j, ...]).cpu().numpy())
            pl.add_mesh(nocs_pc, color="lightblue", point_size=10.0, render_points_as_spheres=True)
            pl.add_mesh(og_coords_pc, color="black", point_size=5.0, render_points_as_spheres=True)
            pl.add_mesh(mesh, color="crimson", opacity=0.4)
            pl.show_grid()
            pl.show_axes()
            pl.show()

    if vis_sdf_and_mesh:
        print("Visualize SDF and normalized mesh.")
        for j in range(bs):
            # create the trimesh mesh
            mesh = trimesh.Trimesh(
                vertices=normalized_mesh[0][j, ...].numpy(force=True),
                faces=normalized_mesh[1][j, ...].numpy(force=True),
            )

            instance_sdf = sdf[j, ...].numpy(force=True)
            non_grid_sdf_points_cnt = object_pc[j, ...].shape[0]

            grid_sdf_values = instance_sdf[non_grid_sdf_points_cnt * 2 :].reshape(
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
    fabric,
    dataloader,
    normalized_recons=True,
    vis_gt_nocs_and_cad=True,
    vis_gt_nocs_registration=True,
    vis_sdf_and_mesh=True,
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
                sdf,
                normalized_mesh,
                coords,
                cam_intrinsics,
                gt_world_T_cam,
            ) = (
                batch["rgb"],
                batch["depth"],
                batch["instance_segmap"],
                batch["nocs"],
                batch["object_pc"],
                batch["sdf"],
                batch["normalized_mesh"],
                batch["coords"],
                batch["cam_intrinsics"],
                batch["cam_pose"],
            )
            mask = segmap == 1

            if "unified_objects" in dataloader.dataset.__dict__.keys():
                voxel_res = dataloader.dataset.unified_objects.global_nonmnfld_points_voxel_res
                cube_scale = (
                    dataloader.dataset.unified_objects.sample_bounds[1]
                    - dataloader.dataset.unified_objects.sample_bounds[0]
                )
            else:
                voxel_res = dataloader.dataset.global_nonmnfld_points_voxel_res
                cube_scale = dataloader.dataset.sample_bounds[1] - dataloader.dataset.sample_bounds[0]

            voxel_origin = np.array([0, 0, 0]) - cube_scale / 2
            voxel_size = cube_scale / (voxel_res - 1)

            single_batch_sanity_test(
                mask=mask,
                gt_nocs=gt_nocs,
                object_pc=object_pc,
                sdf_grid=batch["sdf_grid"],
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
                normalized_recons=normalized_recons,
            )

            torch.cuda.empty_cache()


@dataclass
class ExpSettings:
    dataset_dir: str
    shapenet_data_dir: Optional[str] = None
    bop_data_dir: Optional[str] = None
    replicacad_data_dir: Optional[str] = None
    preload_to_mem: bool = False

    # dataset
    pc_size: int = 60000
    per_batch_sample_surface_points_count: int = 1500
    per_batch_sample_local_nonmnfld_points_count: int = 1500
    per_batch_sample_global_nonmnfld_points_count: int = 4000
    global_nonmnfld_voxel_res: int = 128
    normalized_recons: bool = True
    dataset_debug_vis: bool = False
    vis_gt_nocs_registration: bool = True
    vis_gt_nocs_and_cad: bool = True
    vis_sdf_and_mesh: bool = True

    # automatically populated if missing
    exp_id: str = None


def main(opt: ExpSettings):
    LF.seed_everything(42)
    fabric = L.Fabric(accelerator="cuda", strategy="ddp", devices=[0])
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
        input_H=224,
        input_W=224,
        normalized_recons=opt.normalized_recons,
        debug_vis=opt.dataset_debug_vis,
        data_to_output=[
            "rgb",
            "depth",
            "nocs",
            "coords",
            "normals",
            "sdf",
            "cam_intrinsics",
            "cam_pose",
            "instance_segmap",
            "normalized_mesh",
            "object_pc",
            "sdf_grid",
        ],
    )
    test_dl = tchdata.DataLoader(
        shape_ds, shuffle=False, num_workers=0, batch_size=1, collate_fn=unified_objects_collate_fn
    )
    test_dl = fabric.setup_dataloaders(test_dl)

    sanity_test(
        fabric,
        test_dl,
        normalized_recons=opt.normalized_recons,
        vis_gt_nocs_and_cad=opt.vis_gt_nocs_and_cad,
        vis_gt_nocs_registration=opt.vis_gt_nocs_registration,
        vis_sdf_and_mesh=opt.vis_sdf_and_mesh,
    )


if __name__ == "__main__":
    """Sanity checks for the generated unified dataset"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    main(ExpSettings(**opt))
