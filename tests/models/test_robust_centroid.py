import matplotlib.pyplot as plt
import pytest
import numpy as np
import torch


@pytest.mark.parametrize("B, N, outlier_ratio", [(1, 5000, 0.05), (20, 5000, 0.05)])
def test_robust_centroid_gnc_2d(B, N, outlier_ratio):
    from crisp.models.robust_centroid import robust_centroid_gnc

    # [0, 1] x [0, 1]
    X = torch.tensor(np.random.rand(B, 2, N)).float()
    payload = robust_centroid_gnc(X, cost_type="gnc-tls", clamp_thres=1)
    a_outlierfree = payload["robust_centroid"]

    # add outliers
    for bid in range(B):
        for i in range(int(N * outlier_ratio)):
            X[bid, :, i] += np.random.rand(2) * 100

    payload = robust_centroid_gnc(X, cost_type="gnc-tls", clamp_thres=1)
    a_outlier = payload["robust_centroid"]

    for bid in range(B):
        assert np.linalg.norm(a_outlier[bid, ...].numpy(force=True) - a_outlierfree[bid, ...].numpy(force=True)) < 5e-2