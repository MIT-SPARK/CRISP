import os
import numpy as np
import pandas as pd

metrics_path = "/home/jnshi/code/Hydra-Objects/experiments/nocs/artifacts/202411092051_2QXT0/metrics.npy"

if __name__ == "__main__":
    print("Eval shape recons")
    metrics = list(np.load(metrics_path, allow_pickle=True))
    shape_chamfer_metrics = [{"class_name": x["gt_class_name"], "cf": x["cf_l2"]} for x in metrics]
    df = pd.DataFrame(shape_chamfer_metrics)
    print(f'Per cat avg: {df.groupby("class_name").mean() * 1000}')
    print(f'Avg: {df["cf"].mean() * 1000}')
