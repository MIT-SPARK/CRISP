import pytest
import numpy as np
import random
import math


@pytest.mark.parametrize("N", [(500), (1000), (2000)])
def test_geometric_median_1d(N):
    from crisp.utils.geometry import geometric_median

    X = np.random.rand(N, 1)
    a = geometric_median(X, eps=1e-7)

    assert np.linalg.norm(a - np.median(X, axis=0)) < 1e-3


@pytest.mark.parametrize("N", [(5000), (10000)])
def test_geometric_median_2d(N):
    from crisp.utils.geometry import geometric_median

    # [0, 1] x [0, 1]
    X = np.random.rand(N, 2)
    a = geometric_median(X, eps=1e-7)

    # geometric median should be close to 0.5, 0.5
    assert np.linalg.norm(a - np.array([0.5, 0.5])) < 1e-2


@pytest.mark.parametrize("N, outlier_ratio", [(5000, 0.05), (10000, 0.05)])
def test_geometric_median_2d_outliers(N, outlier_ratio):
    from crisp.utils.geometry import geometric_median

    # [0, 1] x [0, 1]
    X = np.random.rand(N, 2)
    a_outlierfree = geometric_median(X, eps=1e-7)

    # add outliers
    for i in range(int(N * outlier_ratio)):
        X[i, ...] += np.random.rand(2) * 100

    a_outlier = geometric_median(X, eps=1e-7)
    assert np.linalg.norm(a_outlierfree - a_outlier) < 1e-1


@pytest.mark.parametrize("N, D", [(5, 10), (20, 500)])
def test_geometric_median_simple(N, D):
    from crisp.utils.geometry import geometric_median

    X = np.random.rand(N, D)
    a = geometric_median(X)
    assert a.shape == (D,)

    # equivariance check
    t = np.random.rand(1, D)
    X_translated = X + t
    a_2 = geometric_median(X_translated)
    assert np.linalg.norm(a + t - a_2) < 1e-5
