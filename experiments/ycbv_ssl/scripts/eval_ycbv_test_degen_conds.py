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
# runs_folder = script_dir.parent / "exp_results" / "20241028_degen_cond_normalized_synthetic"
# USE THE BELOW FOR VISUALIZING min eigen vs track length
# runs_folder = script_dir.parent / "exp_results" / "20241105_degen_track_vis"

runs_folder = script_dir.parent / "exp_results" / "20241106_degen_oc_nonlip_synth"

SAVE_PLOTS = True

sns.set_palette("tab10")

rand_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

# plotting settings
plt.rcParams.update({"font.size": 13})


def plot_degen_conds(df, dump_folder):
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(df, x="track_len", y="FTF_min_eig", hue="obj_id", palette="tab10")
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"ycbv_test_degen_cond_{rand_string}.pdf", dpi=200)


def plot_adds_vs_degen_cond(dfs, column="adds_to_plot", dump_folder=pathlib.Path("./")):
    df = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(df, x="ftf_min_eig", y=column, palette="tab10", ax=ax)

    ax.set_xlabel(MIN_EIG_LABEL)
    ax.set_ylabel(ADDS_LABEL)
    ax.grid(True)
    ax.set_yscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"adds_vs_degen_cond_{rand_string}.pdf", dpi=200)
    plt.savefig(dump_folder / f"adds_vs_degen_cond_{rand_string}.png", dpi=200)
    return


def plot_adds_vs_oc(dfs, column="adds_to_plot", dump_folder=pathlib.Path("./")):
    df = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(df, x="oc_score", y=column, palette="tab10", ax=ax)

    ax.grid(True)
    ax.set_xlabel(OC_SCORE_LABEL)
    ax.set_ylabel(ADDS_LABEL)
    ax.set_yscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"adds_vs_oc_score_{rand_string}.pdf", dpi=200)
    return


def plot_adds_vs_degen_cond_per_obj(dfs, column="adds_to_plot", dump_folder=pathlib.Path("./")):
    df = pd.concat(dfs, ignore_index=True)
    all_obj_labels = sorted(list(set(df["obj_label"])))
    adds_plots_dump_folder = dump_folder / f"adds_vs_degen_cond_{rand_string}"
    safely_make_folders([adds_plots_dump_folder])

    for obj_label in tqdm(all_obj_labels):
        fig, ax = plt.subplots(1, 1)
        df_obj = df[df["obj_label"] == obj_label]
        ax.set_title(obj_label)
        sns.scatterplot(df_obj, x="ftf_min_eig", y=column, palette="tab10", ax=ax)

        fig.tight_layout()
        if not SAVE_PLOTS:
            plt.show()

        plt.savefig(adds_plots_dump_folder / f"adds_vs_degen_cond_{obj_label}.pdf", dpi=200)
    return


def plot_shape_chamfer_vs_degen_cond(dfs_to_plot_list, column="shape_chamfer", dump_folder=pathlib.Path("./")):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs_to_plot_list, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(data=df, x="ftf_min_eig", y=column, ax=ax)
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_xlabel(MIN_EIG_LABEL)
    ax.set_ylabel(SHAPE_CHAMFER_LABEL)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"shape_chamfer_vs_degen_cond_{rand_string}.pdf", dpi=200)

    return


def plot_shape_chamfer_vs_oc(dfs_to_plot_list, column="shape_chamfer", dump_folder=pathlib.Path("./")):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs_to_plot_list, ignore_index=True)

    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    sns.scatterplot(data=df, x="oc_score", y=column, ax=ax)
    ax.set_xlabel(OC_SCORE_LABEL)
    ax.set_ylabel(SHAPE_CHAMFER_LABEL)
    ax.set_yscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    if not SAVE_PLOTS:
        plt.show()

    plt.savefig(dump_folder / f"shape_chamfer_vs_oc_score_{rand_string}.pdf", dpi=200)
    return


def plot_shape_chamfer_vs_degen_cond_per_obj(dfs_to_plot_list, column="shape_chamfer", dump_folder=pathlib.Path("./")):
    """Plot ADDS scores CDF plots"""
    df = pd.concat(dfs_to_plot_list, ignore_index=True)
    all_obj_labels = sorted(list(set(df["obj_label"])))
    shape_chamfer_plots_dump_folder = dump_folder / f"shape_chamfer_vs_degen_cond_{rand_string}"
    safely_make_folders([shape_chamfer_plots_dump_folder])

    for obj_label in tqdm(all_obj_labels):
        fig, ax = plt.subplots(1, 1)
        df_obj = df[df["obj_label"] == obj_label]
        ax.set_title(obj_label)
        sns.scatterplot(df_obj, x="ftf_min_eig", y=column, palette="tab10", ax=ax)

        fig.tight_layout()
        if not SAVE_PLOTS:
            plt.show()

        plt.savefig(shape_chamfer_plots_dump_folder / f"shape_chamfer_vs_degen_cond_{obj_label}.pdf", dpi=200)
        plt.savefig(shape_chamfer_plots_dump_folder / f"shape_chamfer_vs_degen_cond_{obj_label}.png", dpi=200)

    return


def plot_degen_conds_per_trajectory(df, dump_folder):
    plt.rcParams["figure.figsize"] = (6, 3)
    all_traj_ids = sorted(list(set(df["traj_id"])))
    traj_plots_dump_folder = dump_folder / f"ycbv_test_degen_cond_{rand_string}"
    safely_make_folders([traj_plots_dump_folder])
    for traj_id in tqdm(all_traj_ids):
        fig, ax = plt.subplots(1, 1)
        df_traj = df[df["traj_id"] == traj_id]

        # get final FTF_min_eig range
        final_track_len = df_traj["track_len"].max()
        final_ftf_min_eig = df_traj[df_traj["track_len"] == final_track_len]["FTF_min_eig"]
        final_ftf_min_eig_range = final_ftf_min_eig.min(), final_ftf_min_eig.max()
        final_ftf_min_eig_diff = final_ftf_min_eig_range[1] - final_ftf_min_eig_range[0]
        print(f"Final FTF_min_eig range for trajectory {traj_id}: {final_ftf_min_eig_diff}")

        # ax.set(yscale="log")
        ax.grid(True, which="both")
        # FTF_min_eig
        # df_traj = df_traj[(df_traj["obj_id"] == "obj_000001") | (df_traj["obj_id"] == "obj_000014")]
        df_traj = df_traj[df_traj["obj_id"] == "obj_000014"]
        sns.scatterplot(df_traj, x="track_len", y="FTF_min_eig", hue="obj_id", palette="tab10")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("N")
        ax.set_ylabel(r"$\lambda_{min}$")
        ax.set_yscale("log")
        sns.move_legend(
            ax,
            "lower right",
            ncol=1,
            title="Object Label",
        )
        fig.tight_layout()
        if not SAVE_PLOTS:
            plt.show()

        plt.savefig(traj_plots_dump_folder / f"mf_degen_cond_{traj_id}.png", dpi=200)
        plt.savefig(traj_plots_dump_folder / f"mf_degen_cond_{traj_id}.pdf", dpi=200)
        plt.savefig(traj_plots_dump_folder / f"mf_degen_cond_{traj_id}.svg", dpi=200)
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def save_acc_nocs_meshes(acc_nocs_data, dump_folder, idx_to_save=None):
    if idx_to_save is None:
        idx_to_save = list(range(75))
    for traj_id, traj_data in acc_nocs_data.items():
        for obj_id, acc_nocs_data in traj_data.items():
            for i, acc_nocs in tqdm(enumerate(acc_nocs_data), total=len(acc_nocs_data)):
                if i not in idx_to_save:
                    continue
                # use trimesh to save point cloud
                print(f"acc_nocs size: {acc_nocs.shape}")
                # filter out out of ran
                acc_nocs = acc_nocs[:, acc_nocs[0, :] > -0.09]
                acc_nocs = acc_nocs[:, acc_nocs[0, :] < 0.09]
                acc_nocs = acc_nocs[:, acc_nocs[1, :] > -0.1]
                acc_nocs = acc_nocs[:, acc_nocs[1, :] < 0.1]
                acc_nocs = acc_nocs[:, acc_nocs[2, :] > -0.1]
                acc_nocs = acc_nocs[:, acc_nocs[2, :] < 0.1]
                pc = trimesh.PointCloud(acc_nocs.T, process=False)
                pc.export(dump_folder / f"acc_nocs_{traj_id}_{obj_id}_{i}.ply")
                # scene = pc.scene()
                # data = scene.save_image(resolution=(1080, 1080))
                # with open(dump_folder / f"acc_nocs_{traj_id}_{obj_id}_{i}.png", "wb") as f:
                #    f.write(data)
    return


if __name__ == "__main__":
    print("Plotting for YCBV Test Degen Condition")

    pose_dfs_to_plot_list, shape_chamfer_dfs_to_plot_list = [], []
    acc_nocs = []
    for folder in sorted(runs_folder.iterdir()):
        if folder.is_dir():
            try:
                df_degen_conds = load_traj_degen_cond_data(folder, name=folder.name)
                df = load_traj_data(folder, name=folder.name)

                oc_scores = None
                if "oc_score" in df.keys():
                    oc_scores = df["oc_score"]

                pose_dfs_to_plot_list.append(
                    pd.DataFrame(
                        {
                            "adds_to_plot": df["adds_mean"],
                            "ftf_min_eig": df["ftf_min_eig"],
                            "obj_label": df["obj_label"],
                            "oc_score": oc_scores,
                            "method": folder.name,
                        }
                    )
                )
                shape_chamfer_dfs_to_plot_list.append(
                    pd.DataFrame(
                        {
                            "shape_chamfer": df["shape_chamfer_distance"],
                            "ftf_min_eig": df["ftf_min_eig"],
                            "obj_label": df["obj_label"],
                            "oc_score": oc_scores,
                            "method": folder.name,
                        }
                    )
                )
                ## plot_degen_conds(df, dump_folder=script_dir.parent / "plots")
                # plot_degen_conds_per_trajectory(df_degen_conds, dump_folder=script_dir.parent / "plots" / "degen_plots")

                # acc_nocs_traj_data = load_traj_acc_nocs_data(folder)
                # save_acc_nocs_meshes(
                #    acc_nocs_traj_data,
                #    dump_folder=script_dir.parent / "plots" / "acc_nocs",
                #    idx_to_save=[0, 20, 40, 60, 70, 74],
                # )
            except FileNotFoundError:
                print(f"Missing data in {folder}")
            except Exception as e:
                print(f"Error loading data from {folder}: {e}")

    # plot error per degen cond
    plot_adds_vs_degen_cond(
        pose_dfs_to_plot_list,
        dump_folder=script_dir.parent / "plots" / "degen_plots",
    )
    plot_adds_vs_oc(
        pose_dfs_to_plot_list,
        dump_folder=script_dir.parent / "plots" / "degen_plots",
    )
    # plot_adds_vs_degen_cond_per_obj(
    #    pose_dfs_to_plot_list,
    #    dump_folder=script_dir.parent / "plots" / "degen_plots",
    # )
    plot_shape_chamfer_vs_degen_cond(
        shape_chamfer_dfs_to_plot_list,
        dump_folder=script_dir.parent / "plots" / "degen_plots",
    )
    plot_shape_chamfer_vs_oc(
        shape_chamfer_dfs_to_plot_list,
        dump_folder=script_dir.parent / "plots" / "degen_plots",
    )
    # plot_shape_chamfer_vs_degen_cond_per_obj(
    #    shape_chamfer_dfs_to_plot_list,
    #    dump_folder=script_dir.parent / "plots" / "degen_plots",
    # )
