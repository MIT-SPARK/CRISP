from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == "__main__":
    print("Generate object scales data")

    # load data, calculate median scale, save back to file
    file_path = "/mnt/jnshi_data/datasets/NOCS/gt_results/camera/train/gt_results.pkl"
    script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

    with open(file_path, "rb") as f:
        output_data = pickle.load(f)

    obj_scale_index = output_data["object_scale_index"]
    obj_median_scales, obj_avg_scales = {}, {}

    for obj_name, obj_scales in obj_scale_index.items():
        obj_median_scales[obj_name] = np.nanmedian(obj_scales)
        obj_avg_scales[obj_name] = np.nanmean(obj_scales)

    all_obj_names = set(obj_scale_index.keys())
    all_categories = set([x.split("_")[0] for x in all_obj_names])
    print(f"All shapenet categories present: {sorted(all_categories)}")

    output_data["object_median_scales"] = obj_median_scales
    output_data["object_avg_scales"] = obj_avg_scales

    with open(file_path, "wb") as f:
        pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)
