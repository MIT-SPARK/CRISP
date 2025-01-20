import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns

from crisp.utils.evaluation_metrics import add_s_error_with_cd

CDF_LW = 3
ADDS_LABEL = "ADD-S Scores"
ADDS_CDF_YLABEL = "Cumulative Distribution of ADD-S Scores"
ADDS_CDF_SHORT_YLABEL = "CDF of ADD-S Scores"
SHAPE_CHAMFER_CDF_YLABEL = "Cumulative Distribution of Chamfer Distances"
SHAPE_CHAMFER_CDF_SHORT_YLABEL = "CDF of Chamfer Distances"
SHAPE_CHAMFER_LABEL = "Chamfer Distances"
MIN_EIG_LABEL = r"$\lambda_{min}$"
OC_SCORE_LABEL = "oc Scores"

name2color = {
    "CRISP-Real": sns.color_palette("tab10")[1],
    "Real": sns.color_palette("tab10")[1],
    "Real (oc)": sns.color_palette("tab10")[1],
    "CRISP-Real (oc)": sns.color_palette("tab10")[1],
    "Real (oc + nd)": sns.color_palette("tab10")[1],
    "CRISP-Real (oc + nd)": sns.color_palette("tab10")[1],
    "CRISP-Syn": sns.color_palette("tab10")[7],
    "Synthetic": sns.color_palette("tab10")[7],
    "Synthetic (oc)": sns.color_palette("tab10")[7],
    "Synthetic (oc + nd)": sns.color_palette("tab10")[7],
    "CRISP-Syn (oc)": sns.color_palette("tab10")[7],
    "CRISP-Syn (oc + nd)": sns.color_palette("tab10")[7],
    "Synthetic + Corrector (LSQ)": sns.color_palette("tab10")[2],
    "Synthetic + Corrector (PGD)": sns.color_palette("tab10")[3],
    # "Synthetic + SSL (LSQ)": sns.color_palette("tab10")[4],
    # "Synthetic + SSL (PGD)": sns.color_palette("tab10")[5],
    "CRISP-Syn-ST (LSQ)": sns.color_palette("tab10")[4],
    "Synthetic + ST (LSQ)": sns.color_palette("tab10")[4],
    "CRISP-Syn-ST (PGD)": sns.color_palette("tab10")[5],
    "Synthetic + ST (PGD)": sns.color_palette("tab10")[5],
    # baselines
    "GDRNet++": sns.color_palette("tab10")[0],
    "Shap-E": sns.color_palette("tab10")[6],
    "BundleSDF": sns.color_palette("tab10")[8],
    "CosyPose": sns.color_palette("tab10")[9],
}

name2lines = {
    "Real": "-",
    "Real (oc)": "--",
    "Real (oc + nd)": ":",
    "CRISP-Real": "-",
    "CRISP-Real (oc)": "--",
    "CRISP-Real (oc + nd)": ":",
    "Synthetic": "-",
    "Synthetic (oc)": "--",
    "Synthetic (oc + nd)": ":",
    "CRISP-Syn": "-",
    "CRISP-Syn (oc)": "--",
    "CRISP-Syn (oc + nd)": ":",
    "CRISP-Syn + LSQ": "-",
    "Synthetic + Corrector (LSQ)": "-",
    "CRISP-Syn + PGD": "-",
    "Synthetic + Corrector (PGD)": "-",
    # "Synthetic + SSL (LSQ)": "-",
    # "Synthetic + SSL (PGD)": "-",
    "Synthetic + ST (LSQ)": "-",
    "Synthetic + ST (PGD)": "-",
    "CRISP-Syn-ST (LSQ)": "-",
    "CRISP-Syn-ST (PGD)": "-",
    # baselines
    "GDRNet++": "-",
    "Shap-E": "-",
    "BundleSDF": "-",
    "CosyPose": "-",
}


def load_traj_data(exp_results_dir, name="method"):
    """Load evaluation results"""
    traj_data_dir = os.path.join(exp_results_dir, "traj_data")
    all_folders = [
        f for f in os.listdir(traj_data_dir) if os.path.isdir(os.path.join(traj_data_dir, f)) and "traj_" in f
    ]

    data = []
    if len(all_folders) != 0:
        all_traj_data_paths = sorted([os.path.join(traj_data_dir, f, "traj_results.npy") for f in all_folders])
    else:
        all_traj_data_paths = [os.path.join(traj_data_dir, "traj_results.npy")]
    print(f"Loading trajectory data from {all_traj_data_paths}")

    for fpath in all_traj_data_paths:
        try:
            traj_id = int(fpath.split("/")[-2].split("_")[-1])
        except ValueError:
            traj_id = float("nan")
        traj_data = list(np.load(fpath, allow_pickle=True))
        traj_data = [{"traj_id": traj_id, **d} for d in traj_data]

        for entry in traj_data:
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    entry[k] = v.detach().cpu().item()

        data.extend(traj_data)

    df = pd.DataFrame(data)
    df["method"] = name
    return df


def load_traj_degen_cond_data(exp_results_dir, name="method"):
    """Load evaluation results"""
    traj_data_dir = os.path.join(exp_results_dir, "traj_data")
    all_folders = [
        f for f in os.listdir(traj_data_dir) if os.path.isdir(os.path.join(traj_data_dir, f)) and "traj_" in f
    ]

    all_traj_data = []
    if len(all_folders) != 0:
        all_traj_data_paths = sorted([os.path.join(traj_data_dir, f, "mf_degen_conds.npy") for f in all_folders])
    else:
        all_traj_data_paths = [os.path.join(traj_data_dir, "mf_degen_conds.npy")]
    print(f"Loading trajectory data from {all_traj_data_paths}")

    for fpath in all_traj_data_paths:
        try:
            traj_id = int(fpath.split("/")[-2].split("_")[-1])
        except ValueError:
            traj_id = float("nan")
        traj_data = np.load(fpath, allow_pickle=True).item()

        ctraj_entries = []
        for k, v in traj_data.items():
            for ii, d in enumerate(v):
                ctraj_entries.append({"traj_id": traj_id, "obj_id": k, "track_len": ii, **d})

        all_traj_data.extend(ctraj_entries)

    df = pd.DataFrame(all_traj_data)
    df["method"] = name
    return df


def load_traj_acc_nocs_data(exp_results_dir, name="method"):
    """Load evaluation results"""
    traj_data_dir = os.path.join(exp_results_dir, "traj_data")
    all_folders = [
        f for f in os.listdir(traj_data_dir) if os.path.isdir(os.path.join(traj_data_dir, f)) and "traj_" in f
    ]

    all_traj_data = []
    if len(all_folders) != 0:
        all_traj_data_paths = sorted([os.path.join(traj_data_dir, f, "mf_acc_nocs.npy") for f in all_folders])
    else:
        all_traj_data_paths = [os.path.join(traj_data_dir, "mf_acc_nocs.npy")]
    print(f"Loading acc nocs data from {all_traj_data_paths}")

    all_data = {}
    for fpath in all_traj_data_paths:
        try:
            traj_id = int(fpath.split("/")[-2].split("_")[-1])
        except ValueError:
            traj_id = float("nan")
        traj_data = np.load(fpath, allow_pickle=True).item()

        # do the accumulation
        acc_traj_data = {}
        for obj_id, acc_nocs_data in traj_data.items():
            sumed_nocs = [acc_nocs_data[0]]
            for i in range(1, len(acc_nocs_data)):
                sumed_nocs.append(np.concatenate((acc_nocs_data[i], sumed_nocs[-1]), axis=1))
            acc_traj_data[obj_id] = sumed_nocs

        all_data[traj_id] = acc_traj_data

    return all_data


def make_pose_add_s_table(df, adds_column="adds_to_plot"):
    """Helper function to generate the pose table"""
    # statistics to generate:
    # for each method, we need to compute:
    # 1. mean ADD-S
    # 2. median ADD-S
    # 3. AUC (2 cm)
    # 4. AUC (5 cm)
    # 5. AUC (10 cm)
    final_table = []
    methods = df["method"].unique()
    for method in methods:
        df_method = df[df["method"] == method]
        adds = torch.tensor(df_method[adds_column].to_numpy()).unsqueeze(1)
        auc_2 = add_s_error_with_cd(adds, threshold=0.01)
        auc_5 = add_s_error_with_cd(adds, threshold=0.02)
        auc_10 = add_s_error_with_cd(adds, threshold=0.03)

        # add to table
        final_table.append(
            {
                "method": method,
                "mean_adds": adds.mean().item(),
                "median_adds": adds.median().item(),
                "adds_auc_1cm": auc_2[1].item(),
                "adds_auc_2cm": auc_5[1].item(),
                "adds_auc_3cm": auc_10[1].item(),
            }
        )

    metrics_df = pd.DataFrame(final_table)
    return metrics_df


def make_shape_table(df, shp_column="shape_chamfer"):
    """Helper function to generate the pose table"""
    # statistics to generate:
    # for each method, we need to compute:
    # 1. mean ADD-S
    # 2. median ADD-S
    # 3. AUC (2 cm)
    # 4. AUC (5 cm)
    # 5. AUC (10 cm)
    final_table = []
    methods = df["method"].unique()
    for method in methods:
        df_method = df[df["method"] == method]
        cdist = torch.tensor(df_method[shp_column].to_numpy()).unsqueeze(1)
        auc_2 = add_s_error_with_cd(cdist, threshold=0.03)
        auc_5 = add_s_error_with_cd(cdist, threshold=0.05)
        auc_10 = add_s_error_with_cd(cdist, threshold=0.1)

        # add to table
        final_table.append(
            {
                "method": method,
                "mean_shape_chamfer": cdist.mean().item(),
                "median_shape_chamfer": cdist.median().item(),
                "shape_chamfer_auc_3cm": auc_2[1].item(),
                "shape_chamfer_auc_5cm": auc_5[1].item(),
                "shape_chamfer_auc_10cm": auc_10[1].item(),
            }
        )

    metrics_df = pd.DataFrame(final_table)
    return metrics_df
