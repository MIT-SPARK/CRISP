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
runs_folder = script_dir.parent / "exp_results" / "20241101_oc_nd_plot"

folders_to_skip = ["after_sslv2", "after_ssl"]
folders_to_use = ["sup", "before_ssl", "after_ssl", "202408041531_2RN8V", "202408042232_3I5IV"]
FILE_EXTENSION = "pdf"
SAVE_PLOTS = True
PLOT_BASELINES = False

# apply degen cert
# keys are method names, values are thresholds on min eigenvalue (of FtF)
degen_thres = 0.005
degen_certs = {
    "nd_supervised_real": 0.005,
    "oc_nd_supervised_real": 0.005,
    "nd_synthetic": 0.005,
    "oc_nd_synthetic": 0.005,
    "nd_synthetic_corrector": 0.005,
    "oc_nd_synthetic_corrector": 0.005,
    "Real (oc + nd)": 0.005,
    "Synthetic (oc + nd)": 0.005,
}
# apply observably correct cert
oc_certs = [
    "oc_supervised_real",
    "oc_nd_supervised_real",
    "oc_synthetic",
    "oc_nd_synthetic",
    "oc_synthetic_corrector",
    "oc_nd_synthetic_corrector",
    "Real (oc)",
    "Real (oc + nd)",
    "Synthetic (oc)",
    "Synthetic (oc + nd)",
]

methods2plot = ["Real", "Real (oc)", "Real (oc + nd)"]

hue_order_certs = [
    "Real",
    "Real (oc)",
    "Real (oc + nd)",
    # "Synthetic",
    # "Synthetic (oc)",
    # "Synthetic (oc + nd)",
]

sns.set_palette("tab10")

RAND_STRING = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

# plotting settings
plt.rcParams.update({"font.size": 15})


def load_cosypose_data(data_path, name="cosypose"):
    """Load Cosypose data"""
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

    df = pd.DataFrame({"adds_to_plot": data})
    df["method"] = name
    return df


def load_bundlesdf_data(data_path, name="bundleSDF"):
    """Load Cosypose data"""
    with open(data_path, "rb") as data_file:
        data = np.load(data_file)

    df = pd.DataFrame({"adds_to_plot": data})
    df["method"] = name
    return df


def load_shape_data(data_path, name="Shap-E"):
    """Load ShapeE data"""
    with open(data_path, "rb") as data_file:
        data = np.load(data_file)

    df = pd.DataFrame({"shape_chamfer": data.flatten()})
    df["method"] = name
    return df


def load_gdrnet_data(data_path, name="GDRNet"):
    """Load GDRNet data"""
    with open(data_path, "rb") as data_file:
        data = pickle.load(data_file)

    df = pd.DataFrame({"adds_to_plot": data})
    df["method"] = name
    return df


def load_proposed_data(exp_results_dir, name="method"):
    """Load evaluation results of proposed pipeline"""
    df = load_traj_data(exp_results_dir, name=name)
    return df


def plot_adds_cdf_per_obj(dfs, column="adds_to_plot", dump_folder=pathlib.Path("./"), xlim_max=0.06):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs, ignore_index=True)

    unique_obj_labels = sorted(list(set(df["obj_label"])))

    for obj_label in unique_obj_labels:
        fig, ax = plt.subplots(1, 1)
        sns.ecdfplot(
            data=df[df["obj_label"] == obj_label],
            x=column,
            hue="method",
            ax=ax,
            palette=name2color,
            hue_order=hue_order_certs,
        )

        # update lines
        lss = []
        for i in range(len(ax.legend_.texts)):
            ctxt = ax.legend_.texts[i]._text
            lss.append(name2lines[ctxt])
        lss = lss[::-1]

        handles = ax.legend_.legendHandles[::-1]
        for line, ls, handle in zip(ax.lines, lss, handles):
            line.set_linestyle(ls)
            handle.set_ls(ls)

        sns.move_legend(
            ax,
            "lower right",
            ncol=1,
            title=None,
        )
        ax.set_ylabel(ADDS_CDF_YLABEL)
        ax.set_xlabel(ADDS_LABEL)
        ax.set_xlim([0, xlim_max])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True)
        fig.tight_layout()
        if not SAVE_PLOTS:
            plt.show()

        plt.savefig(dump_folder / f"ycbv_{obj_label}_test_adds_cdf.{FILE_EXTENSION}", dpi=200)

    return


def plot_shape_chamfer_cdf_per_obj(
    dfs_to_plot_list, column="shape_chamfer", dump_folder=pathlib.Path("./"), xlim_max=0.3
):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs_to_plot_list, ignore_index=True)
    unique_obj_labels = sorted(list(set(df["obj_label"])))

    for obj_label in unique_obj_labels:
        fig, ax = plt.subplots(1, 1)
        sns.ecdfplot(
            data=df[df["obj_label"] == obj_label],
            x=column,
            hue="method",
            ax=ax,
            palette=name2color,
            hue_order=hue_order_certs,
        )
        # update lines
        lss = []
        for i in range(len(ax.legend_.texts)):
            ctxt = ax.legend_.texts[i]._text
            lss.append(name2lines[ctxt])
        lss = lss[::-1]

        handles = ax.legend_.legendHandles[::-1]
        for line, ls, handle in zip(ax.lines, lss, handles):
            line.set_linestyle(ls)
            handle.set_ls(ls)

        ax.set_ylabel(SHAPE_CHAMFER_CDF_YLABEL, fontsize=12)
        ax.set_xlabel(SHAPE_CHAMFER_LABEL)
        ax.set_xlim([0, xlim_max])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()
        sns.move_legend(
            ax,
            "lower right",
            ncol=1,
            title=None,
        )
        ax.set_axisbelow(True)
        ax.grid(True)

        if not SAVE_PLOTS:
            plt.show()

        plt.savefig(dump_folder / f"ycbv_{obj_label}_test_shape_chamfer_cdf.{FILE_EXTENSION}", dpi=200)

    return


if __name__ == "__main__":
    print("Plotting for YCBV Test")

    ## In SSL metrics: percent cert
    # during_ssl_df = load_traj_data(script_dir.parent / "exp_results" / "20240131/during_ssl/", name="ssl_process")
    # plot_cert_percent(during_ssl_df, script_dir.parent / "plots")

    # Before SSL
    pose_dfs_to_plot_list, shape_chamfer_dfs_to_plot_list = [], []
    nd_oc_stats = {}
    if "sota" in runs_folder.name:
        PLOT_BASELINES = True
    for folder in sorted(runs_folder.iterdir()):
        if folder.is_dir():
            try:
                if folder.name not in methods2plot:
                    continue
                df = load_traj_data(folder, name=folder.name)
                total_length = df.shape[0]
                if folder.name in oc_certs:
                    df = df[df["cert_flag"]]
                if folder.name in degen_certs.keys():
                    df = df[df["ftf_min_eig"] > degen_certs[folder.name]]
                pose_dfs_to_plot_list.append(
                    pd.DataFrame({"adds_to_plot": df["adds_mean"], "method": folder.name, "obj_label": df["obj_label"]})
                )
                shape_chamfer_dfs_to_plot_list.append(
                    pd.DataFrame(
                        {
                            "shape_chamfer": df["shape_chamfer_distance"],
                            "method": folder.name,
                            "obj_label": df["obj_label"],
                        }
                    )
                )
            except FileNotFoundError:
                print(f"Missing data in {folder}")
            except Exception as e:
                print(f"Error loading data from {folder}: {e}")

    # plot_adds_cdf(
    #    [bssl_precrt_df, bssl_postcrt_df, pssl_precrt_df, pssl_postcrt_df, sup_precrt_df, sup_postcrt_df, csy_df],
    #    dump_folder=script_dir.parent / "plots",
    # )
    # cdf plots
    plot_adds_cdf_per_obj(
        pose_dfs_to_plot_list,
        dump_folder=script_dir.parent / "plots" / "per_obj",
    )
    plot_shape_chamfer_cdf_per_obj(shape_chamfer_dfs_to_plot_list, dump_folder=script_dir.parent / "plots" / "per_obj")
