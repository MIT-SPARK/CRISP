import logging
import torch


def weighted_mean(weights, X):
    """Compute weighted mean of provided K-dimensional point clouds

    Args:
        weights: (B, 1, N)
        X: (B, K, N)

    Returns:
        weighted_mean: (B, K, 1)
    """
    # B, K, N
    weighted_points = weights * X
    # B, K, 1
    x = torch.sum(weighted_points, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True)
    return x


def weighted_objective(p, weights, X):
    """Compute the residuals and objective values of the weighted mean problem

    Args:
        p:
        weights:
        X:

    Returns:
        cost: (B, 1) weighted and normalized sum of each of the point cloud in the batch
        residuals:
        sq_residuals:
    """
    # residuals: B, 1, N
    sq_residuals = torch.sum(torch.square(X - p), dim=1, keepdim=True)
    residuals = torch.sqrt(sq_residuals)

    # cost: B, 1
    # batch-wise dot product
    B, _, N = residuals.shape
    cost = torch.bmm(residuals.view(B, 1, N), weights.view(B, N, 1)).squeeze(-1)
    return cost, residuals, sq_residuals


def update_gnc_tls_weights(residuals, clamp_thres, mu):
    """Update TLS weights following GNC"""

    _thL = (mu / (1 + mu)) * (clamp_thres**2)
    _thU = ((mu + 1) / mu) * (clamp_thres**2)

    _ones = torch.ones_like(residuals)
    weights = torch.zeros_like(residuals)

    # setting weights = 1 if sq_residuals < thL
    # setting weights = 0 if sq_residuals > thU
    # otherwise, weights = thres / residuals * sqrt(mu * (mu - 1)) - mu
    _function = -mu + torch.sqrt(mu * (mu + 1)) * clamp_thres / residuals
    sq_residuals = torch.square(residuals)
    weights = _ones * (torch.le(sq_residuals, _thL)) + _function * (
        torch.gt(sq_residuals, _thL) * torch.le(sq_residuals, _thU)
    )
    return weights


def update_gnc_gm_weights(residuals, clamp_thres, mu):
    """Update GM weights following GNC"""
    const = mu * (clamp_thres**2)
    weights = torch.square(const / (torch.square(residuals) + const))
    return weights


def robust_centroid_gnc(
    X,
    cost_type="gnc-tls",
    clamp_thres=0.1,
    max_iterations=100,
    cost_diff_stop_th=1e-5,
    cost_abs_stop_th=1e-5,
):
    """Solve robust centroid problem with GNC

    Args:
        X: (B, K, N)
        cost_type: type of the cost function to use in GNC; gnc-tls or gnc-gm
        clamp_thres:
        max_iterations: maximum iterations to use for the GNC algorithm
        cost_diff_stop_th:
        cost_abs_stop_th: If absolute cost is lower than this value, stop optimization. Note that too large of a value
                          the precision will be affected.
    """
    B, _, N = X.shape
    device = X.device
    weights = torch.ones(B, 1, N).to(device=device)
    iter = 0

    # initial solution
    # x: (B, K, 1)
    x = weighted_mean(weights, X)
    cost, residuals, sq_residuals = weighted_objective(x, weights, X)
    best_x = x.clone()
    best_weights = weights.clone()
    # save the cost for the next iteration
    prev_cost = cost

    # initialize mu
    sq_thres = clamp_thres**2
    r_sq_max = torch.max(sq_residuals, -1)[0].unsqueeze(-1)
    if cost_type == "gnc-gm":
        mu = 2 * r_sq_max / sq_thres
        mu_update_fun = lambda x: x / 1.4
    elif cost_type == "gnc-tls":
        mu = sq_thres / (2 * r_sq_max)
        mu_update_fun = lambda x: x * 1.4
    else:
        raise NotImplementedError

    # weight update function
    if cost_type == "gnc-gm":
        weight_f = update_gnc_gm_weights
    elif cost_type == "gnc-tls":
        weight_f = update_gnc_tls_weights
    else:
        raise NotImplementedError

    # update weights with the first solution
    weights = weight_f(residuals=residuals, clamp_thres=clamp_thres, mu=mu)

    # individual flag for each point cloud in the batch
    # flags = 1 if we still need to optimize
    # flags = 0 if we already reached stopping threshold
    batch_optim_flags = torch.ones(B, 1, device=device)
    # threshold for each pc in the batch (for comparison with the cost diff)
    while iter < max_iterations:
        iter += 1

        # solve with current weights
        x = weighted_mean(weights, X)

        # current objective
        cost, residuals, sq_residuals = weighted_objective(x, weights, X)

        # update weights
        weights = weight_f(residuals=residuals, clamp_thres=clamp_thres, mu=mu)

        # update cost diff: B, 1, N
        cost_diff = torch.abs(cost - prev_cost)
        prev_cost = cost

        # flags shape: B, 1. check for absolute cost and cost difference
        # true if we want to continue running the optimization
        batch_optim_flags = torch.logical_and(batch_optim_flags, cost > cost_abs_stop_th)
        batch_optim_flags = torch.logical_and(batch_optim_flags, cost_diff > cost_diff_stop_th)

        if not torch.any(batch_optim_flags):
            break

        # save x only for the K-dimensional point clouds  that we are still not terminated
        best_x[batch_optim_flags[:, 0], ...] = x[batch_optim_flags[:, 0], ...]
        best_weights[batch_optim_flags[:, 0], ...] = weights[batch_optim_flags[:, 0], ...]

        # calculate mu
        mu = mu_update_fun(mu)

    if iter == max_iterations:
        logging.warning("Hitting maximum iterations in GNC for robust centroid.")

    payload = {
        "method_name": f"gnc_{cost_type}",
        "residuals": residuals,
        "cost": cost,
        "robust_centroid": best_x,
        "weights": weights,
        "iterations": iter,
    }

    return payload
