import numpy as np
import os
from jsonargparse import ArgumentParser
from dataclasses import dataclass
import random
import string
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import pathlib
from scipy.ndimage.filters import gaussian_filter1d

from eval_utils import *

script_dir = pathlib.Path(__file__).parent.resolve().absolute()
runs_folder = script_dir.parent / "exp_results" / "20241113_runtime"

folders_to_skip = ["after_sslv2", "after_ssl"]
folders_to_use = ["sup", "before_ssl", "after_ssl", "202408041531_2RN8V", "202408042232_3I5IV"]
FILE_EXTENSION = "pdf"
SAVE_PLOTS = True
PLOT_BASELINES = False

sns.set_palette("tab10")

RAND_STRING = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

# plotting settings
plt.rcParams.update({"font.size": 15})


def make_runtime_table(df):
    """Helper function to generate the runtime table"""
    df = pd.concat(runtime_dfs_to_plot_list, ignore_index=True)
    methods = df["method"].unique()
    final_data = []
    for method in methods:
        df_method = df[df["method"] == method]
        if "lsq" in method:
            filtered_runtime = df_method["mf_corrector_time"][df_method["mf_corrector_time"] > 1]
        elif "pgd" in method:
            filtered_runtime = df_method["mf_corrector_time"][df_method["mf_corrector_time"] > 1]
        elif "model" in method:
            filtered_runtime = df_method["model_inference_time"][df_method["model_inference_time"] > 1]
        else:
            raise ValueError(f"Unknown method: {method}")

        runtime_mean = filtered_runtime.mean()
        runtime_median = filtered_runtime.median()
        runtime_std = filtered_runtime.std()
        final_data.append(
            {
                "method": method,
                "mean_runtime": runtime_mean,
                "median_runtime": runtime_median,
                "std_runtime": runtime_std,
            }
        )

    df = pd.DataFrame(final_data)

    return df


if __name__ == "__main__":
    print("Plotting for YCBV Test")

    ## In SSL metrics: percent cert
    # during_ssl_df = load_traj_data(script_dir.parent / "exp_results" / "20240131/during_ssl/", name="ssl_process")
    # plot_cert_percent(during_ssl_df, script_dir.parent / "plots")

    # Before SSL
    runtime_dfs_to_plot_list = []
    nd_oc_stats = {}
    if "sota" in runs_folder.name:
        PLOT_BASELINES = True
    for folder in sorted(runs_folder.iterdir()):
        if folder.is_dir():
            try:
                df = load_traj_data(folder, name=folder.name)
                total_length = df.shape[0]
                sf_corrector_time = 0
                mf_corrector_time = 0
                model_inference_time = 0
                if "sf_corrector_time" in df.keys():
                    sf_corrector_time = df["sf_corrector_time"]
                if "mf_corrector_time" in df.keys():
                    mf_corrector_time = df["mf_corrector_time"]
                if "model_inference_time" in df.keys():
                    model_inference_time = df["model_inference_time"]

                runtime_dfs_to_plot_list.append(
                    pd.DataFrame(
                        {
                            "mf_corrector_time": mf_corrector_time,
                            "sf_corrector_time": sf_corrector_time,
                            "model_inference_time": model_inference_time,
                            "method": folder.name,
                        }
                    )
                )
            except FileNotFoundError:
                print(f"Missing data in {folder}")
            except Exception as e:
                print(f"Error loading data from {folder}: {e}")

    final_runtime_df = make_runtime_table(runtime_dfs_to_plot_list)
    final_runtime_df.to_csv("runtime.csv")
