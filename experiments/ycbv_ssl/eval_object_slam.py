import os
import numpy as np
from jsonargparse import ArgumentParser
from dataclasses import dataclass

# project local imports
from crisp.backend import bop_to_g2o
from crisp.backend.object_pgo import rpgo
from crisp.datasets import bop
from crisp.utils.visualization_utils import plot_obj_pose_graphs, plot_obj_pose_graph


@dataclass
class Settings:
    ds_name: str = "ycbv"
    bop_ds_dir: str = "../../data/bop/bop_datasets"
    split: str = "train_real"
    seq_name: str = "ycbv_train_real_0"
    data_dir: str = "./joint_model_bopg2o/"
    output_dir: str = "./joint_model_bopg2o_output/pgo_output.g2o"
    plot_folder: str = "./plots"

    # SLAM parameters
    odom_as_inliers: bool = True
    obj_meas_as_outliers: bool = False
    inlier_cost_thres: float = 0.05


if __name__ == "__main__":
    # evaluating one object SLAM sequence, given some .g2o file
    parser = ArgumentParser()
    parser.add_class_arguments(Settings)
    opt = parser.parse_args()
    opt = Settings(**opt)

    # Obtaining BOPG2O Datasets
    seq_name = opt.seq_name  # change this
    source_file = os.path.join(opt.data_dir, seq_name + ".g2o")
    gt_source_file = os.path.join(opt.data_dir, seq_name + "_gt.g2o")
    object_index_file = os.path.join(opt.data_dir, seq_name + ".object_index")
    gt_pose_graph = rpgo.read_g2o(gt_source_file)

    # Setting up
    pgo_backend = rpgo.ObjectPGO(
        in_file=source_file,
        out_file=opt.output_dir,
        object_index_file=object_index_file,
        plot_folder=opt.plot_folder,
        plot=False,
        init=True,
        gnc=True,
        obj_pose_meas_variance=np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]),
        odom_as_inliers=opt.odom_as_inliers,
        obj_meas_as_outliers=opt.obj_meas_as_outliers,
        first_odom_node_prior=rpgo.g2o_pose_to_gtsam_pose(gt_pose_graph[0][0][1]),
        inlier_cost_thres=opt.inlier_cost_thres,
    )

    # Solving
    pgo_backend.solve()

    # load reconstructed objects using latent shape code

    # Plot Object SLAM
    object_index = bop_to_g2o.read_object_index(object_index_file)
    labels = list(object_index.keys())
    gt_shape_models = bop.get_bop_objects(labels, opt.ds_name)

    print("Visualize GT pose graph")
    input_pose_graph = rpgo.read_g2o(pgo_backend.in_file)
    output_pose_graph = rpgo.read_g2o(pgo_backend.out_file)

    pl = plot_obj_pose_graph(gt_pose_graph, object_index, gt_shape_models, color="blue")
    #pl = plot_obj_pose_graph(input_pose_graph, object_index, gt_shape_models, plotter=pl, color="yellow")
    pl = plot_obj_pose_graph(output_pose_graph, object_index, gt_shape_models, plotter=pl, color="green")

    pl.show_grid()
    pl.show_axes()
    pl.show()
