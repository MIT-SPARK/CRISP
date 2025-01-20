import os
import pandas as pd
import numpy as np
import scipy.stats


def load_proposed_method_data(directory_path):
    subfolders = [
        os.path.join(directory_path, o)
        for o in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, o))
    ]
    subfolders.sort()
    if len(subfolders) == 0:
        return []
    latest_subfolder = subfolders[-1]
    metrics = np.load(os.path.join(latest_subfolder, "metrics.npy"), allow_pickle=True)
    return metrics


def load_method(method_name: str, data_dir):
    our_path = os.path.join(data_dir, method_name)
    test_data_folder = os.path.join(our_path, "test")
    our_test_data = load_proposed_method_data(test_data_folder)
    our_test_data = [{"split": "test", "method": method_name, **x} for x in our_test_data]

    val_data_folder = os.path.join(our_path, "validation")
    our_val_data = load_proposed_method_data(val_data_folder)
    our_val_data = [{"split": "val", "method": method_name, **x} for x in our_val_data]
    return our_val_data, our_test_data


def load_data(data_dir="./data"):
    all_data = []
    # load sat_sq_recon data
    sat_sq_recon_path = os.path.join(data_dir, "sat_sq_recon/")
    sat_sq_recon_test_data = np.load(os.path.join(sat_sq_recon_path, "test", "metrics.npy"), allow_pickle=True)
    sat_sq_recon_test_data = [{"split": "test", "method": "sat_sq_recon", **x} for x in sat_sq_recon_test_data]
    sat_sq_recon_val_data = np.load(os.path.join(sat_sq_recon_path, "validation", "metrics.npy"), allow_pickle=True)
    sat_sq_recon_val_data = [{"split": "val", "method": "sat_sq_recon", **x} for x in sat_sq_recon_val_data]
    all_data.extend(sat_sq_recon_test_data)
    all_data.extend(sat_sq_recon_val_data)

    # load our data (supervised)
    try:
        sup_val_data, sup_test_data = load_method("sup", data_dir)
        all_data.extend(sup_test_data)
        all_data.extend(sup_val_data)
    except FileNotFoundError:
        print("No data found for supervised.")

    # load supervised w/ corrector lsq
    try:
        sup_cor_val_lsq_data, sup_cor_test_lsq_data = load_method("sup_corrector_lsq", data_dir)
        all_data.extend(sup_cor_val_lsq_data)
        all_data.extend(sup_cor_test_lsq_data)
    except FileNotFoundError:
        print("No data found for supervised corrector lsq.")

    # load supervised w/ corrector proj gd
    try:
        sup_cor_val_proj_gd_data, sup_cor_test_proj_gd_data = load_method("sup_corrector_proj_gd", data_dir)
        all_data.extend(sup_cor_val_proj_gd_data)
        all_data.extend(sup_cor_test_proj_gd_data)
    except FileNotFoundError:
        print("No data found for supervised corrector proj gd.")

    ## load our data (ssl)
    #try:
    #    ssl_val_data, ssl_test_data = load_method("sup_ssl", data_dir)
    #    all_data.extend(ssl_test_data)
    #    all_data.extend(ssl_val_data)
    #except FileNotFoundError:
    #    print("No data found for supervised + SSL.")

    # load our data (ssl lsq)
    try:
        ssl_lsq_val_data, ssl_lsq_test_data = load_method("sup_ssl_lsq", data_dir)
        all_data.extend(ssl_lsq_test_data)
        all_data.extend(ssl_lsq_val_data)
    except FileNotFoundError:
        print("No data found for supervised + SSL (LSQ).")

    # load our data (ssl proj gd)
    try:
        ssl_proj_gd_val_data, ssl_proj_gd_test_data = load_method("sup_ssl_proj_gd", data_dir)
        all_data.extend(ssl_proj_gd_test_data)
        all_data.extend(ssl_proj_gd_val_data)
    except FileNotFoundError:
        print("No data found for supervised + SSL (Proj GD).")

    # sup ssl w/ corrector
    try:
        ssl_corrector_val_data, ssl_corrector_test_data = load_method("sup_ssl_corrector", data_dir)
        all_data.extend(ssl_corrector_test_data)
        all_data.extend(ssl_corrector_val_data)
    except FileNotFoundError:
        print("No data found for supervised + SSL w/ corrector.")

    # synth
    try:
        synth_val_data, synth_test_data = load_method("synth", data_dir)
        all_data.extend(synth_test_data)
        all_data.extend(synth_val_data)
    except FileNotFoundError:
        print("No data found for synth.")

    # synth w/ corrector lsq
    try:
        synth_corrector_lsq_val_data, synth_corrector_lsq_test_data = load_method("synth_corrector_lsq", data_dir)
        all_data.extend(synth_corrector_lsq_test_data)
        all_data.extend(synth_corrector_lsq_val_data)
    except FileNotFoundError:
        print("No data found for synth w/ corrector lsq.")

    # synth w/ corrector proj gd
    try:
        synth_corrector_proj_gd_val_data, synth_corrector_proj_gd_test_data = load_method("synth_corrector_proj_gd", data_dir)
        all_data.extend(synth_corrector_proj_gd_test_data)
        all_data.extend(synth_corrector_proj_gd_val_data)
    except FileNotFoundError:
        print("No data found for synth w/ corrector proj gd.")

    ## synth ssl
    #try:
    #    ssl_synth_val_data, ssl_synth_test_data = load_method("synth_ssl", data_dir)
    #    all_data.extend(ssl_synth_test_data)
    #    all_data.extend(ssl_synth_val_data)
    #except FileNotFoundError:
    #    print("No data found for synth + SSL.")

    # synth ssl (LSQ)
    try:
        ssl_lsq_synth_val_data, ssl_lsq_synth_test_data = load_method("synth_ssl_lsq", data_dir)
        all_data.extend(ssl_lsq_synth_test_data)
        all_data.extend(ssl_lsq_synth_val_data)
    except FileNotFoundError:
        print("No data found for synth + SSL (LSQ).")

    # synth ssl (Proj GD)
    try:
        ssl_proj_gd_synth_val_data, ssl_proj_gd_synth_test_data = load_method("synth_ssl_proj_gd", data_dir)
        all_data.extend(ssl_proj_gd_synth_test_data)
        all_data.extend(ssl_proj_gd_synth_val_data)
    except FileNotFoundError:
        print("No data found for synth + SSL (Proj GD).")

    # synth ssl w/ corrector
    try:
        ssl_synth_corrector_val_data, ssl_synth_corrector_test_data = load_method("synth_ssl_corrector", data_dir)
        all_data.extend(ssl_synth_corrector_test_data)
        all_data.extend(ssl_synth_corrector_val_data)
    except FileNotFoundError:
        print("No data found for synth + SSL w/ corrector.")

    df = pd.DataFrame.from_records(all_data)
    df = df.astype(
        {
            "eR": float,
            "eT": float,
            "chamfer_l1": float,
            "chamfer_l2": float,
            "pose_chamfer_l1": float,
            "pose_chamfer_l2": float,
        }
    )
    return df


def get_stats(df, method_name, split, reduction="mean"):
    metrics_names = ["eR", "eT", "chamfer_l1", "chamfer_l2", "pose_chamfer_l1", "pose_chamfer_l2"]
    # sat_sq_recon
    if reduction == "mean":
        stats = {x: df.loc[(df["method"] == method_name) & (df["split"] == split)][x].mean() for x in metrics_names}
        #stats = {x: scipy.stats.trim_mean(df.loc[(df["method"] == method_name) & (df["split"] == split)][x], 0.1) for x in metrics_names}
    elif reduction == "median":
        stats = {x: df.loc[(df["method"] == method_name) & (df["split"] == split)][x].median() for x in metrics_names}
    else:
        raise NotImplementedError
    return stats


def summarize_method_stats(df, method_name):
    summary_table_entries = []
    sat_sq_recon_val_mean = get_stats(df, method_name, "val")
    sat_sq_recon_test_mean = get_stats(df, method_name, "test")
    summary_table_entries.append({"method": method_name, "split": "val", "type": "mean", **sat_sq_recon_val_mean})
    summary_table_entries.append({"method": method_name, "split": "test", "type": "mean", **sat_sq_recon_test_mean})

    sat_sq_recon_val_median = get_stats(df, method_name, "val", reduction="median")
    sat_sq_recon_test_median = get_stats(df, method_name, "test", reduction="median")
    summary_table_entries.append(
        {"method": method_name, "split": "val", "type": "median", **sat_sq_recon_val_median}
    )
    summary_table_entries.append(
        {"method": method_name, "split": "test", "type": "median", **sat_sq_recon_test_median}
    )
    return summary_table_entries


if __name__ == "__main__":
    print("Evaluation script for satellite dataset.")

    df = load_data(data_dir="data")

    summary_table_entries = []

    # table data: mean, median of
    # trans_err, rot_err, l1 chamfer, l2 chamfer
    summary_table_entries.extend(summarize_method_stats(df, "sat_sq_recon"))

    # sup
    summary_table_entries.extend(summarize_method_stats(df, "sup"))

    # sup corrector
    summary_table_entries.extend(summarize_method_stats(df, "sup_corrector_lsq"))
    summary_table_entries.extend(summarize_method_stats(df, "sup_corrector_proj_gd"))

    # sup ssl
    summary_table_entries.extend(summarize_method_stats(df, "sup_ssl_proj_gd"))
    summary_table_entries.extend(summarize_method_stats(df, "sup_ssl_lsq"))

    # sup ssl w/ corrector
    #summary_table_entries.extend(summarize_method_stats(df, "sup_ssl_corrector"))

    # synth
    summary_table_entries.extend(summarize_method_stats(df, "synth"))

    # synth corrector
    summary_table_entries.extend(summarize_method_stats(df, "synth_corrector_lsq"))
    summary_table_entries.extend(summarize_method_stats(df, "synth_corrector_proj_gd"))

    # synth ssl
    summary_table_entries.extend(summarize_method_stats(df, "synth_ssl_proj_gd"))
    summary_table_entries.extend(summarize_method_stats(df, "synth_ssl_lsq"))

    # synth ssl w/ corrector
    #summary_table_entries.extend(summarize_method_stats(df, "synth_ssl_corrector"))

    summary_df = pd.DataFrame.from_records(summary_table_entries)
    summary_df = summary_df.sort_values(by=["split", "method"])
    print(summary_df.to_string())
