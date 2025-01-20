import os
import dataclasses
from tqdm import tqdm
import numpy as np
import torch
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from jsonargparse import ArgumentParser
import utils.geometry
import pyvista as pv
import seaborn as sns
import pandas as pd

# local lib imports
from crisp.datasets.unified_objects import (
    UnifiedObjects,
)
from experiments.unified_model.train import ExpSettings


def plot_chamfer_heatmap(df, title):
    fig, ax = plt.subplots()
    sns.heatmap(df, cmap="YlGnBu", cbar=True, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def gt_shape_sdf_test(shape_ds):
    chamfer_table = {}
    sdf_w_reg_table = {}
    for k, v in shape_ds.distinct_objects["ycbv"].items():
        chamfer_table[k] = {kk: 0 for kk in shape_ds.distinct_objects["ycbv"].keys()}
        sdf_w_reg_table[k] = {kk: 0 for kk in shape_ds.distinct_objects["ycbv"].keys()}

    global_nonmnfld_points_voxel_res = 64
    cube_scale = 0.251
    off_surface_coords_global, voxel_size, voxel_origin = utils.geometry.voxelize_cube(
        N=global_nonmnfld_points_voxel_res, cube_center=np.array([0, 0, 0]), cube_scale=cube_scale
    )

    records = []
    for k in tqdm(chamfer_table.keys()):
        shape_k_points = shape_ds.distinct_objects["ycbv"][k]["surface_points"].numpy()

        for kk in chamfer_table[k].keys():
            kk_mesh = shape_ds.distinct_objects["ycbv"][kk]["normalized_mesh"]
            # use the sdf function
            chamfer_dist = utils.geometry.query_sdf_from_mesh(shape_k_points, kk_mesh)
            cf = np.mean(np.abs(chamfer_dist))

            # w/ free space reg
            nonmnfld_pred = utils.geometry.query_sdf_from_mesh(off_surface_coords_global, kk_mesh)
            inter_cost = np.exp(-100 * np.abs(nonmnfld_pred)).mean()

            records.append(
                {
                    "Query": k,
                    "SDF": kk,
                    "manifold": cf,
                    "inter": 100 * inter_cost,
                    "manifold_w_inter": 3e3 * cf + 100 * inter_cost,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


if __name__ == "__main__":
    """
    For all objects, use CAD model to calculate SDF
    Test GT NOCS / GT object point clouds with all the latent code conditioned networks
    Check whether the correct pair has the lowest loss
    Loss is manifold points SDF values + free space SDF non zero term
    """
    print("Experimental code for shape loss")
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    shape_ds = UnifiedObjects(
        folder_path=opt.dataset_dir,
        shapenet_dataset_path=opt.shapenet_data_dir,
        bop_dataset_path=opt.bop_data_dir,
        spe3r_dataset_path=opt.spe3r_data_dir,
        replicacad_dataset_path=opt.replicacad_data_dir,
        preload_to_mem=opt.preload_to_mem,
        pc_size=opt.pc_size,
        sample_surface_points_count=opt.per_batch_sample_surface_points_count,
        sample_local_nonmnfld_points_count=opt.per_batch_sample_local_nonmnfld_points_count,
        sample_global_nonmnfld_points_count=opt.per_batch_sample_global_nonmnfld_points_count,
        global_nonmnfld_points_voxel_res=opt.global_nonmnfld_voxel_res,
        sample_bounds=(-1.0, 1.0),
        debug_vis=opt.dataset_debug_vis,
        normalized_recons=opt.normalized_recons,
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

    ctable = gt_shape_sdf_test(shape_ds)
    plot_chamfer_heatmap(
        ctable.pivot(index="Query", columns="SDF", values="manifold"),
        title="SDF on Manifold",
    )
    plot_chamfer_heatmap(
        ctable.pivot(index="Query", columns="SDF", values="inter"),
        title="Free Space Regularization",
    )
    plot_chamfer_heatmap(
        ctable.pivot(index="Query", columns="SDF", values="manifold_w_inter"),
        title="SDF on Manifold + Free Space Regularization",
    )
