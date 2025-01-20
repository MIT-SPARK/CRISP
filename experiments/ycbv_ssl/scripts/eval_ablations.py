import trimesh
import numpy as np
import os
from jsonargparse import ArgumentParser
from dataclasses import dataclass
from tqdm import tqdm
import random
import string
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import pathlib
from crisp.utils.file_utils import safely_make_folders
from scipy.ndimage.filters import gaussian_filter1d

from eval_utils import *


script_dir = pathlib.Path(__file__).parent.resolve().absolute()

runs_folder = script_dir.parent / "exp_results" / "20241111_ablations"

SAVE_PLOTS = True

sns.set_palette("tab10")

rand_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))


def calculate_scores(dfs_to_plot_list):
    df = pd.concat(dfs_to_plot_list, ignore_index=True)
    methods = df["method"].unique()
    for method in methods:
        df_method = df[df["method"] == method]

        adds = torch.tensor(df_method["adds_mean"].to_numpy()).unsqueeze(1)
        auc_adds_1 = add_s_error_with_cd(adds, threshold=0.01)
        auc_adds_2 = add_s_error_with_cd(adds, threshold=0.02)
        auc_adds_3 = add_s_error_with_cd(adds, threshold=0.03)

        cdist = torch.tensor(df_method["shape_chamfer"].to_numpy()).unsqueeze(1)
        auc_shape_2 = add_s_error_with_cd(cdist, threshold=0.03)
        auc_shape_5 = add_s_error_with_cd(cdist, threshold=0.05)
        auc_shape_10 = add_s_error_with_cd(cdist, threshold=0.1)

        print(f"Method: {method}")
        print(f"Mean ADD-S: {df_method['adds_mean'].mean()}")
        print(f"Mean Shape Chamfer: {df_method['shape_chamfer'].mean()}")
        print(f"AUC (1 cm): {auc_adds_1[1].item()}")
        print(f"AUC (2 cm): {auc_adds_2[1].item()}")
        print(f"AUC (3 cm): {auc_adds_3[1].item()}")
        print(f"Shape AUC (3 cm): {auc_shape_2[1].item()}")
        print(f"Shape AUC (5 cm): {auc_shape_5[1].item()}")
        print(f"Shape AUC (10 cm): {auc_shape_10[1].item()}")
    return


if __name__ == "__main__":
    print("Calculating ADD-S and ADD-S AUC scores")

    pose_dfs_to_plot_list, shape_chamfer_dfs_to_plot_list = [], []
    acc_nocs = []
    for folder in sorted(runs_folder.iterdir()):
        if folder.is_dir():
            try:
                df = load_traj_data(folder, name=folder.name)

                pose_dfs_to_plot_list.append(
                    pd.DataFrame(
                        {
                            "adds_mean": df["adds_mean"],
                            "shape_chamfer": df["shape_chamfer_distance"],
                            "method": folder.name,
                        }
                    )
                )
            except FileNotFoundError:
                print(f"Missing data in {folder}")
            except Exception as e:
                print(f"Error loading data from {folder}: {e}")

    calculate_scores(pose_dfs_to_plot_list)
