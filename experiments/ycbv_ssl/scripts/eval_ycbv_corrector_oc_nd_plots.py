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
# runs_folder = script_dir.parent / "exp_results" / "20241029_degen_cert_ssl"
# runs_folder = script_dir.parent / "exp_results" / "20241024_proj_gd_lip_contra_cert_schedule"
# runs_folder = script_dir.parent / "exp_results" / "20241101_oc_nd_plot"
runs_folder = script_dir.parent / "exp_results" / "20241101_corrector_plot"

folders_to_skip = ["after_sslv2", "after_ssl"]
folders_to_use = ["sup", "before_ssl", "after_ssl", "202408041531_2RN8V", "202408042232_3I5IV"]
SAVE_PLOTS = True
PLOT_BASELINES = False

# apply degen cert
# keys are method names, values are thresholds on min eigenvalue (of FtF)
sns.set_palette("tab10")

RAND_STRING = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

# plotting settings
plt.rcParams.update({"font.size": 15})
degen_thres = 0.005

FILE_EXTENSION = "pdf"
name2label = {
    "Synthetic": "None",
    "Synthetic + Corrector (LSQ)": "LSQ",
    "Synthetic + Corrector (PGD)": "BCD",
}


def plot_adds_boxplot(dfs, column="adds_to_plot", dump_folder=pathlib.Path("./")):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=df, y=column, x="method", ax=ax, palette=name2color, legend=False, log_scale=True)
    fig.tight_layout()
    ax.set_ylabel(ADDS_LABEL)
    ax.set_axisbelow(True)
    ax.grid(True)
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"ycbv_test_adds_boxplot_{RAND_STRING}.pdf", dpi=200)
    plt.savefig(dump_folder / f"ycbv_test_adds_boxplot_{RAND_STRING}.png", dpi=200)
    return


def plot_shape_chamfer_boxplot(dfs_to_plot_list, column="shape_chamfer", dump_folder=pathlib.Path("./")):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs_to_plot_list, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=df, y=column, x="method", ax=ax, palette=name2color, legend=False)
    fig.tight_layout()
    ax.set_ylabel(SHAPE_CHAMFER_LABEL)
    ax.set_axisbelow(True)
    ax.grid(True)
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"ycbv_test_shape_chamfer_boxplot_{RAND_STRING}.pdf", dpi=200)
    plt.savefig(dump_folder / f"ycbv_test_shape_chamfer_boxplot_{RAND_STRING}.png", dpi=200)

    return


def plot_oc_nd_catplots(df, dump_folder=pathlib.Path("./")):
    methods = df["method"]
    x = np.arange(len(methods))
    flags = ["oc", "oc + nd"]
    width = 0.33
    multiplier = 0

    sns.set_palette("Set2")
    fig, ax = plt.subplots(1, 1)
    for flag in flags:
        offset = width * multiplier
        measurement = list(df[flag] * 100)
        rects = ax.bar(x + offset, measurement, width, label=flag)
        ax.bar_label(rects, padding=3, fmt="%.1f%%")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage of Total Detections (%)")
    ax.set_xticks(x + width / 2, methods)
    ax.set_ylim(0, 20)
    ax.legend(ncols=2, loc="upper left")
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"ycbv_test_corrector_comp_oc_nd_{RAND_STRING}.{FILE_EXTENSION}", dpi=200)

    return


def plot_oc_only_catplots(df, dump_folder=pathlib.Path("./")):
    plt.rcParams.update({"font.size": 25})
    methods = df["method"]
    x = np.arange(len(methods))
    flags = ["oc"]
    width = 0.4
    multiplier = 0

    sns.set_palette("Set2")
    fig, ax = plt.subplots(1, 1)
    for flag in flags:
        offset = width * multiplier
        measurement = list(df[flag] * 100)
        rects = ax.bar(x + offset, measurement, width, label=flag)
        ax.bar_label(rects, padding=3, fmt="%.1f%%")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Percentage of Detections (%)", fontsize=20)
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 20)
    ax.legend(ncols=2, loc="upper left")
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"ycbv_test_corrector_comp_oc_only_{RAND_STRING}.{FILE_EXTENSION}", dpi=200)

    return


if __name__ == "__main__":
    print("Plotting for YCBV Test")

    nd_oc_stats = []
    for folder in sorted(runs_folder.iterdir()):
        if folder.is_dir():
            try:
                df = load_traj_data(folder, name=folder.name)
                total_length = df.shape[0]
                df["nd_flag"] = df["ftf_min_eig"] > degen_thres
                oc_passed_objects = df["cert_flag"].sum()
                nd_passed_objects = df[df["ftf_min_eig"] > degen_thres].shape[0]
                oc_nd_passed_objects = df[df["cert_flag"] & (df["ftf_min_eig"] > degen_thres)].shape[0]

                nd_oc_stats.append(
                    {
                        "method": name2label[folder.name],
                        "oc": oc_passed_objects / total_length,
                        "nd": nd_passed_objects / total_length,
                        "oc + nd": oc_nd_passed_objects / total_length,
                    }
                )

            except FileNotFoundError:
                print(f"Missing data in {folder}")
            except Exception as e:
                print(f"Error loading data from {folder}: {e}")

    print(f"ND/OC stats: ")
    for v in nd_oc_stats:
        print(f"{v}")

    df = pd.DataFrame(nd_oc_stats)
    plot_oc_nd_catplots(df, dump_folder=script_dir.parent / "plots")
    plot_oc_only_catplots(df, dump_folder=script_dir.parent / "plots")
