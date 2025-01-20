import numpy as np
from crisp.utils.visualization_utils import visualize_pcs_pyvista

if __name__ == "__main__":
    print("Debug script to investigate the difference between the batch corrector and instance corrector")

    #instance_depths = np.load("./instance_corrector_logs/instance_depth_pts_1.npy")
    #batch_depths = np.load("./batch_corrector_logs/batch_depth_pts_1.npy")

    #visualize_pcs_pyvista([batch_depths, instance_depths], colors=["crimson", "blue"], pt_sizes=[5.0, 5.0], bg_color="grey")

    #instance_precorr_nocs = np.load("./instance_corrector_logs/instance_precorr_nocs_1.npy")
    #batch_precorr_nocs = np.load("./batch_corrector_logs/batch_precorr_nocs_1.npy")

    #visualize_pcs_pyvista([batch_precorr_nocs, instance_precorr_nocs], colors=["crimson", "blue"], pt_sizes=[5.0, 5.0], bg_color="grey")

    instance_precorr_nocs = np.load("./instance_corrector_logs/instance_precorr_nocs_1.npy")
    instance_postcorr_nocs = np.load("./instance_corrector_logs/instance_postcorr_nocs_1.npy")
    visualize_pcs_pyvista([instance_precorr_nocs, instance_postcorr_nocs], colors=["crimson", "blue"], pt_sizes=[5.0, 5.0], bg_color="grey")
