import numpy as np
import _pickle as cPickle

my_gt_path = "/mnt/jnshi_data/datasets/NOCS/gt_results/real/test/gt_results.pkl"
nocs_gt_dir = "/mnt/jnshi_data/datasets/NOCS/gts/real_test"

if __name__ == "__main__":
    print("Compare my GT and NOCS GT")

    # load my gt
    with open(my_gt_path, "rb") as f:
        my_gt = cPickle.load(f)

    s1img0 = my_gt["gt_results"]["scene_1/0000"]

    # load nocs gt results
    nocs_gt_path = nocs_gt_dir + "/" + "results_real_test_scene_1_0000.pkl"
    with open(nocs_gt_path, "rb") as f:
        nocs_gt = cPickle.load(f)

    print("yolo")
    nocs_gt_RTs = nocs_gt["gt_RTs"]

    # reformat
    my_gt_RTs = []
    for entry in s1img0:
        T = np.eye(4)
        T[:3, :3] = entry["gt_R"]
        T[:3, :3] *= entry["gt_s"]
        T[:3, 3] = entry["gt_t"]

        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1

        my_gt_RTs.append(z_180_RT @ T)

    for i in range(len(my_gt_RTs)):
        print(f"my GT:  {my_gt_RTs[i]}")
        print(f"NOCS GT: {nocs_gt_RTs[i]}")
