from dataclasses import dataclass
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from eval_utils import load_traj_data


@dataclass
class ExpSettings:
    exp_results_dir: str = "./model_ckpts/202311202317_OPPLM"


def load_precrt_traj_data(exp_results_dir, name="no-corrector"):
    df = load_traj_data(exp_results_dir, name=name)
    df["cert_percent"] = df["precrt_cert_percent"]
    return df


def report_metrics(df, method_col="method"):
    """Plot the metrics"""
    # histograms of:
    # rotation err
    # translation err
    # scale err
    plt.tight_layout()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex=False)
    sns.ecdfplot(data=df, x="R_err", hue=method_col, ax=ax1)
    sns.ecdfplot(data=df, x="t_err", hue=method_col, ax=ax2)
    sns.ecdfplot(data=df, x="s_err", hue=method_col, ax=ax3)
    f.savefig("./plots/pose_errors.pdf", dpi=150)

    # histograms of add-s median std & mean
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), sharex=False)
    sns.ecdfplot(data=df, x="adds_mean", hue=method_col, ax=ax1, log_scale=True)
    sns.ecdfplot(data=df, x="adds_median", hue=method_col, ax=ax2, log_scale=True)
    sns.ecdfplot(data=df, x="adds_std", hue=method_col, ax=ax3, log_scale=True)
    ax1.set_xlim([0, 0.5])
    ax1.set_xlim([0, 0.5])
    ax1.set_xlim([0, 0.5])
    f.savefig("./plots/adds_errors.pdf", dpi=150)

    return


def plot_cert(df, method_col="method"):
    # average certified %
    f, ax1 = plt.subplots(1, 1, figsize=(7, 5), sharex=False)
    sns.histplot(data=df, x="cert_percent", hue=method_col, ax=ax1, bins=20)
    f.savefig("./plots/percent_certified.pdf", dpi=150)

    f, ax1 = plt.subplots(1, 1, figsize=(7, 5), sharex=False)
    sns.ecdfplot(data=df, x="cert_percent", hue=method_col, ax=ax1)
    ax1.set_xlim([0, 0.5])
    f.savefig("./plots/percent_certified_cdf.pdf", dpi=150)

    return


if __name__ == "__main__":
    sdf_nocs_dir = "./model_ckpts/202311202317_OPPLM"
    sdf_input_dir = "./model_ckpts/202311212336_KHYTY"

    # sdf-nocs
    sdf_nocs_df = load_traj_data(sdf_nocs_dir, "sdf-nocs")

    # sdf-input
    sdf_input_df = load_traj_data(sdf_input_dir, "sdf-input")

    # no corrector
    no_corrector_cert_df = load_precrt_traj_data(sdf_input_dir, "no-corrector")

    df = pd.concat([sdf_nocs_df, sdf_input_df], ignore_index=True)
    report_metrics(df)

    df_w_no_crt = pd.concat([df, no_corrector_cert_df], ignore_index=True)
    plot_cert(df_w_no_crt)
