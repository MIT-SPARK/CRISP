import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import os


def make_obj_histogram(obj_name, obj_scales, path_to_save):
    outlier_thres = np.percentile(obj_scales, 99)

    # Create histogram
    plt.hist(obj_scales[obj_scales < outlier_thres], bins=100, alpha=0.5)

    # Set title and labels
    plt.title(f"Histogram of scales for {obj_name}")
    plt.xlabel("Scale")
    plt.ylabel("Frequency")

    # Save the histogram
    file_name = f"{obj_name}_histogram.png"
    plt.savefig(os.path.join(path_to_save, file_name))  # change your saving path

    # Clear the current plot
    plt.close()



if __name__ == '__main__':
    print("Investigate object distribution")
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    # histogram of object scales
    # for each object, plot scale histogram
    #file_path = "/mnt/jnshi_data/datasets/NOCS/gt_results/real/test/gt_results.pkl"
    #histogram_save_path = script_dir.parent / "plots" / "histogram" / "real_test"
    file_path = "/mnt/jnshi_data/datasets/NOCS/gt_results/camera/train/gt_results.pkl"
    histogram_save_path = script_dir.parent / "plots" / "histogram" / "camera_train"

    with open(file_path, "rb") as f:
        output_data = pickle.load(f)

    obj_scale_index = output_data["object_scale_index"]

    i = 0
    for obj_name, obj_scales in obj_scale_index.items():
        make_obj_histogram(obj_name, np.array(obj_scales), path_to_save=histogram_save_path)
        i += 1
        if i > 100:
            break

