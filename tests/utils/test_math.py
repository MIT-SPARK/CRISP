import pytest
import torch


@pytest.mark.parametrize("B,D,N", [(1, 1, 1), (20, 5, 100), (100, 20, 100)])
def test_batch_var(B, D, N):
    from crisp.utils.math import batch_var

    points = torch.rand((B, D, N))
    weights = torch.ones((B, N))
    var1 = torch.var(points, dim=2, correction=0)
    var2 = batch_var(points=points, weights=weights, correction=0)

    assert torch.all(torch.isclose(var1, var2))


@pytest.mark.parametrize("B,D,N", [(1, 5, 5), (20, 5, 100), (100, 20, 100)])
def test_batch_var_weighted(B, D, N):
    from crisp.utils.math import batch_var

    points = torch.rand((B, D, N))
    weights = torch.rand((B, N))
    var1 = batch_var(points=points, weights=weights, correction=0)
    for bid in range(B):
        var2 = torch.cov(points[bid, ...], correction=0, aweights=weights[bid, ...])
        assert torch.all(torch.isclose(var1[bid, ...].flatten(), var2.diag().flatten()))


@pytest.mark.parametrize("B", [(1), (20), (100)])
def test_make_scaled_se3_inverse_batched(B):
    from crisp.utils.math import project_SO3
    from crisp.utils.math import make_scaled_se3_inverse_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device, dtype=torch.float64)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device, dtype=torch.float64) * 10
    t = torch.rand((B, 3, 1), device=device, dtype=torch.float64) * 10

    pc_src = torch.rand((B, 3, 5), device=device, dtype=torch.float64) * 10
    pc_tgt = s * (R @ pc_src) + t
    T_inv = make_scaled_se3_inverse_batched(s, R, t)
    pc_src_2 = T_inv[:, :3, :3] @ pc_tgt + T_inv[:, :3, -1].reshape((B, 3, 1))
    assert torch.allclose(pc_src, pc_src_2)


def test_projection_simplex():
    from crisp.utils.math import project_simplex
    import numpy as np

    rng = np.random.RandomState(0)
    V = rng.rand(100, 10)
    result = project_simplex(V, z=1)
    assert np.allclose(np.sum(result, axis=1), 1)
    assert np.all(result >= 0)