import logging
import numpy as np
import random
import torch
import torch.nn.functional
from torch import _VF
from torch.profiler import record_function

try:
    from pytorch3d import ops
except Exception:
    print(f"Missing pytorch3d. Some functions will not work.")


def masked_varul_mean(data, mask):
    """Return data mean and variance lower and upper bounds, using masked data
    inputs:
    data    : torch.tensor of shape (B, n)
    mask    : torch.tensor of shape (B, n). dtype=torch.bool

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """
    device_ = data.device
    batch_size = data.shape[0]

    var = torch.zeros(batch_size, 2).to(device_)
    mean = torch.zeros(batch_size).to(device_)

    for batch, (d, m) in enumerate(zip(data, mask)):
        dm = torch.masked_select(d, m)

        dm_mean = dm.mean()
        dm_centered = dm - dm_mean
        dm_centered_up = dm_centered * (dm_centered >= 0)
        dm_centered_lo = dm_centered * (dm_centered < 0)
        len = dm_centered.shape[0]

        dm_var_up = torch.sum(dm_centered_up**2) / (len + 0.001)
        dm_var_lo = torch.sum(dm_centered_lo**2) / (len + 0.001)

        mean[batch] = dm_mean
        var[batch, 0] = dm_var_lo
        var[batch, 1] = dm_var_up

    return var, mean


def varul_mean(data):
    """Return data mean and variance lower and upper bounds
    inputs:
    data    : torch.tensor of shape (B, n)

    outputs:
    var     : torch.tensor of shape (B, 2)  :
        var[:, 0] = lower variance
        var[:, 1] = upper variance

    mean    : torch.tensor of shape (B,)

    """
    mean = data.mean(dim=1).unsqueeze(-1)

    data_centered = data - mean
    data_pos = data_centered * (data_centered >= 0)
    data_neg = data_centered * (data_centered < 0)
    len = data_centered.shape[1]

    var_up = torch.sum(data_pos**2, dim=1) / (len + 0.001)
    var_low = torch.sum(data_neg**2, dim=1) / (len + 0.001)

    var_up = var_up.unsqueeze(-1)
    var_low = var_low.unsqueeze(-1)
    var = torch.cat([var_low, var_up], dim=1)

    return var, mean.squeeze(-1)


def set_all_random_seeds(rng_seed):
    """Helper function to set all relevant libraries random seed"""
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    return


def grid_trilinear_interp(
    point: np.ndarray, origin: np.ndarray, xyz_res: np.ndarray, dims: np.ndarray, data_grid: np.ndarray
):
    """Perform trilinear interpolation, assume regular grid

    Args:
        point: 3D (x,y,z) coordinate
        xyz_res: Resolution of the grid in 3D: (x res, y res, z res)
        origin: Origin of the grid (minimal (x,y,z) value of all the coordinates)
        data_grid: 3D array of scalar values

    """
    # ensure point lies in extant
    assert point[0] >= origin[0] and point[1] >= origin[1] and point[2] >= origin[2]
    max_x = origin[0] + (dims[0] - 1) * xyz_res[0]
    max_y = origin[1] + (dims[1] - 1) * xyz_res[1]
    max_z = origin[2] + (dims[2] - 1) * xyz_res[2]
    assert point[0] <= max_x and point[1] <= max_y and point[2] <= max_z

    # calculate x,y,z indices
    ind_000_arr = np.floor_divide(point - np.asarray(origin), xyz_res).astype(int)
    ind_000 = tuple(ind_000_arr)
    if np.allclose(origin + ind_000_arr * xyz_res, point):
        # short circuit for points on vertices
        return data_grid[ind_000]
    ind_001 = tuple(ind_000_arr + [0, 0, 1])
    ind_010 = tuple(ind_000_arr + [0, 1, 0])
    ind_011 = tuple(ind_000_arr + [0, 1, 1])
    ind_100 = tuple(ind_000_arr + [1, 0, 0])
    ind_101 = tuple(ind_000_arr + [1, 0, 1])
    ind_110 = tuple(ind_000_arr + [1, 1, 0])
    ind_111 = tuple(ind_000_arr + [1, 1, 1])

    c_000 = data_grid[ind_000]
    c_001 = data_grid[ind_001]
    c_010 = data_grid[ind_010]
    c_011 = data_grid[ind_011]
    c_100 = data_grid[ind_100]
    c_101 = data_grid[ind_101]
    c_110 = data_grid[ind_110]
    c_111 = data_grid[ind_111]

    x_0 = ind_000[0] * xyz_res[0] + origin[0]
    y_0 = ind_000[1] * xyz_res[1] + origin[1]
    z_0 = ind_000[2] * xyz_res[2] + origin[2]

    x_d = (point[0] - x_0) / xyz_res[0]
    y_d = (point[1] - y_0) / xyz_res[1]
    z_d = (point[2] - z_0) / xyz_res[2]

    # x interp
    c_00 = c_000 * (1 - x_d) + c_100 * x_d
    c_01 = c_001 * (1 - x_d) + c_101 * x_d
    c_10 = c_010 * (1 - x_d) + c_110 * x_d
    c_11 = c_011 * (1 - x_d) + c_111 * x_d

    # y interp
    c_0 = c_00 * (1 - y_d) + c_10 * y_d
    c_1 = c_01 * (1 - y_d) + c_11 * y_d

    # z interp and final value
    c = c_0 * (1 - z_d) + c_1 * z_d

    return c


def grid_trilinear_interp_torch_single_point(point, origin, xyz_res, dims, data_grid, device="cuda"):
    """Perform trilinear interpolation, assume regular grid

    Args:
        point: 3D (x,y,z) coordinate
        xyz_res: Resolution of the grid in 3D: (x res, y res, z res)
        origin: Origin of the grid (minimal (x,y,z) value of all the coordinates)
        data_grid: 3D array of scalar values

    """
    # ensure point lies in extant
    assert point[0] >= origin[0] and point[1] >= origin[1] and point[2] >= origin[2]
    max_x = origin[0] + (dims[0] - 1) * xyz_res[0]
    max_y = origin[1] + (dims[1] - 1) * xyz_res[1]
    max_z = origin[2] + (dims[2] - 1) * xyz_res[2]
    assert point[0] <= max_x and point[1] <= max_y and point[2] <= max_z

    # calculate x,y,z indices
    ind_000_arr = torch.div(point - origin, xyz_res, rounding_mode="floor").long()
    ind_000 = tuple(ind_000_arr.tolist())
    if torch.allclose(origin + ind_000_arr * xyz_res, point.float()):
        # short circuit for points on vertices
        return data_grid[ind_000]
    ind_001 = tuple((ind_000_arr + torch.tensor([0, 0, 1], device=device)).tolist())
    ind_010 = tuple((ind_000_arr + torch.tensor([0, 1, 0], device=device)).tolist())
    ind_011 = tuple((ind_000_arr + torch.tensor([0, 1, 1], device=device)).tolist())
    ind_100 = tuple((ind_000_arr + torch.tensor([1, 0, 0], device=device)).tolist())
    ind_101 = tuple((ind_000_arr + torch.tensor([1, 0, 1], device=device)).tolist())
    ind_110 = tuple((ind_000_arr + torch.tensor([1, 1, 0], device=device)).tolist())
    ind_111 = tuple((ind_000_arr + torch.tensor([1, 1, 1], device=device)).tolist())

    c_000 = data_grid[ind_000]
    c_001 = data_grid[ind_001]
    c_010 = data_grid[ind_010]
    c_011 = data_grid[ind_011]
    c_100 = data_grid[ind_100]
    c_101 = data_grid[ind_101]
    c_110 = data_grid[ind_110]
    c_111 = data_grid[ind_111]

    x_0 = ind_000[0] * xyz_res[0] + origin[0]
    y_0 = ind_000[1] * xyz_res[1] + origin[1]
    z_0 = ind_000[2] * xyz_res[2] + origin[2]

    x_d = (point[0] - x_0) / xyz_res[0]
    y_d = (point[1] - y_0) / xyz_res[1]
    z_d = (point[2] - z_0) / xyz_res[2]

    # x interp
    c_00 = c_000 * (1 - x_d) + c_100 * x_d
    c_01 = c_001 * (1 - x_d) + c_101 * x_d
    c_10 = c_010 * (1 - x_d) + c_110 * x_d
    c_11 = c_011 * (1 - x_d) + c_111 * x_d

    # y interp
    c_0 = c_00 * (1 - y_d) + c_10 * y_d
    c_1 = c_01 * (1 - y_d) + c_11 * y_d

    # z interp and final value
    c = c_0 * (1 - z_d) + c_1 * z_d

    return c


def grid_trilinear_interp_torch_slow(
    points: torch.Tensor,
    origin: torch.Tensor,
    xyz_res: torch.Tensor,
    dims: torch.Tensor,
    data_grid: torch.Tensor,
    device="cuda",
):
    """Perform trilinear interpolation, assume regular grid, and point is a torch tensor.

    Args:
        point: A batch of points (B,3,m), where B is batch size, m is the number of points
        origin: Origin of the grid (minimal (x,y,z) value of all the coordinates)
        xyz_res: Resolution of the grid in 3D: (x res, y res, z res)
        dims: Dimension of the data grid
        data_grid: 3D array of scalar values

    """
    # current implementation simply calls the single point version above
    # TODO: Faster torch-based batch implementation (floor divide)
    B = points.size(dim=0)
    m = points.size(dim=2)
    output = torch.zeros(B, 1, m, device=device)
    for batch_idx in range(B):
        for i in range(m):
            point = points[batch_idx, :, i]
            sdf_v = grid_trilinear_interp_torch_single_point(point, origin, xyz_res, dims, data_grid, device=device)
            output[batch_idx, 0, i] = sdf_v

    return output


def grid_trilinear_interp_exact_torch(
    points: torch.Tensor,
    origin: torch.Tensor,
    xyz_res: torch.Tensor,
    dims: torch.Tensor,
    data_grid: torch.Tensor,
    outside_value=0,
):
    """Perform trilinear interpolation, assume regular grid, and point is a torch tensor.
    This function performs explicit tests to ensure that points passed as vertices return
    exactly as the input data grid. For points outside the bounds, they are filled with
    the provided outside value.

    Args:
        point: A batch of points (B,3,m), where B is batch size, m is the number of points
        origin: Origin of the grid (minimal (x,y,z) value of all the coordinates)
        xyz_res: Resolution of the grid in 3D: (x res, y res, z res)
        dims: Dimension of the data grid
        data_grid: 3D array of scalar values
        outside_value: a scalar value to return if the points' coordinates are out of bounds
        device: device on which the tensors live

    """
    # center the points
    centered_points = points - origin.view((3, 1))

    # calculate the indices of the bottom left points of the cells in which the query points lie in
    ind_000 = torch.div(centered_points, xyz_res.view((3, 1)), rounding_mode="floor").long()

    # get mask of points that are queries on vertices
    ind = torch.div(centered_points, xyz_res.view((3, 1)))
    vt_mask = torch.all(torch.eq(ind, ind_000), dim=1)
    vt_values = data_grid[
        torch.masked_select(ind[:, 0, :], vt_mask).long(),
        torch.masked_select(ind[:, 1, :], vt_mask).long(),
        torch.masked_select(ind[:, 2, :], vt_mask).long(),
    ]

    # out of bounds mask: points outside the SDF volume
    # OOB defined as: any index < 0, any index > dim(x/y/z)
    # oob_mask will be (B, 1, m)
    # oob_mask will be used to replace out of bound values
    oob_mask = torch.logical_or(ind_000[:, 0, :] < 0, ind_000[:, 0, :] >= dims[0] - 1)
    oob_mask.logical_or_(ind_000[:, 1, :] < 0).logical_or_(ind_000[:, 1, :] >= dims[1] - 1)
    oob_mask.logical_or_(ind_000[:, 2, :] < 0).logical_or_(ind_000[:, 2, :] >= dims[2] - 1)

    # replace out of bounds indices with 0 to ensure no errors when accessing data grid
    # the values will be replaced anyways so the actual indices don't matter
    ind_000.masked_fill_(oob_mask.unsqueeze(1), 0)

    # get corner values
    c_000 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :], ind_000[:, 2, :]]
    c_001 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :], ind_000[:, 2, :] + 1]
    c_010 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :] + 1, ind_000[:, 2, :]]
    c_011 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :] + 1, ind_000[:, 2, :] + 1]
    c_100 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :], ind_000[:, 2, :]]
    c_101 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :], ind_000[:, 2, :] + 1]
    c_110 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :] + 1, ind_000[:, 2, :]]
    c_111 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :] + 1, ind_000[:, 2, :] + 1]

    # fractional distance within the cell
    xyz_d = (centered_points - ind_000 * xyz_res.view((3, 1))).div_(xyz_res.view((3, 1)))

    # compute interpolations
    interp_v = (
        c_000 * (1 - xyz_d[:, 0, :]) * (1 - xyz_d[:, 1, :]) * (1 - xyz_d[:, 2, :])
        + c_100 * xyz_d[:, 0, :] * (1 - xyz_d[:, 1, :]) * (1 - xyz_d[:, 2, :])
        + c_010 * (1 - xyz_d[:, 0, :]) * xyz_d[:, 1, :] * (1 - xyz_d[:, 2, :])
        + c_001 * (1 - xyz_d[:, 0, :]) * (1 - xyz_d[:, 1, :]) * xyz_d[:, 2, :]
        + c_101 * xyz_d[:, 0, :] * (1 - xyz_d[:, 1, :]) * xyz_d[:, 2, :]
        + c_011 * (1 - xyz_d[:, 0, :]) * xyz_d[:, 1, :] * xyz_d[:, 2, :]
        + c_110 * xyz_d[:, 0, :] * xyz_d[:, 1, :] * (1 - xyz_d[:, 2, :])
        + c_111 * xyz_d[:, 0, :] * xyz_d[:, 1, :] * xyz_d[:, 2, :]
    )

    # replace with mask
    interp_v.masked_fill_(oob_mask, outside_value)

    # replace vertices with vertex values
    interp_v.masked_scatter_(vt_mask, vt_values)

    return interp_v.unsqueeze_(1)


@torch.jit.script
def grid_trilinear_interp_torch(
    points: torch.Tensor,
    origin: torch.Tensor,
    xyz_res: torch.Tensor,
    dims: torch.Tensor,
    data_grid: torch.Tensor,
    outside_value: float = 0,
):
    """Perform trilinear interpolation, assume regular grid, and point is a torch tensor.
    This function does not perform explicit tests to ensure that points passed as vertices return
    exactly as the input data grid. For points >= the bounds, they are filled with
    the provided outside value.

    Args:
        point: A batch of points (B,3,m), where B is batch size, m is the number of points
        origin: Origin of the grid (minimal (x,y,z) value of all the coordinates)
        xyz_res: Resolution of the grid in 3D: (x res, y res, z res)
        dims: Dimension of the data grid
        data_grid: 3D array of scalar values
        outside_value: a scalar value to return if the points' coordinates are out of bounds
        device: device on which the tensors live

    """
    # center the points
    centered_points = points - origin.view((3, 1))

    # calculate the indices of the bottom left points of the cells in which the query points lie in
    ind_000 = torch.div(centered_points, xyz_res.view((3, 1)), rounding_mode="floor").long()

    # out of bounds mask: points outside the SDF volume
    # OOB defined as: any index < 0, any index > dim(x/y/z)
    # oob_mask will be (B, 1, m)
    # oob_mask will be used to replace out of bound values
    with record_function("oob_masking"):
        oob_mask = torch.logical_or(ind_000[:, 0, :] < 0, ind_000[:, 0, :] >= dims[0] - 1)
        oob_mask.logical_or_(ind_000[:, 1, :] < 0).logical_or_(ind_000[:, 1, :] >= dims[1] - 1)
        oob_mask.logical_or_(ind_000[:, 2, :] < 0).logical_or_(ind_000[:, 2, :] >= dims[2] - 1)

    # replace out of bounds indices with 0 to ensure no errors when accessing data grid
    # the values will be replaced anyways so the actual indices don't matter
    ind_000.masked_fill_(oob_mask.unsqueeze(1), 0)

    # get corner values
    with record_function("grid_indexing"):
        c_000 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :], ind_000[:, 2, :]]
        c_001 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :], ind_000[:, 2, :] + 1]
        c_010 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :] + 1, ind_000[:, 2, :]]
        c_011 = data_grid[ind_000[:, 0, :], ind_000[:, 1, :] + 1, ind_000[:, 2, :] + 1]
        c_100 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :], ind_000[:, 2, :]]
        c_101 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :], ind_000[:, 2, :] + 1]
        c_110 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :] + 1, ind_000[:, 2, :]]
        c_111 = data_grid[ind_000[:, 0, :] + 1, ind_000[:, 1, :] + 1, ind_000[:, 2, :] + 1]

    # fractional distance within the cell
    with record_function("frac_dists"):
        xyz_d = (centered_points - ind_000 * xyz_res.view((3, 1))).div_(xyz_res.view((3, 1)))

    # compute interpolations
    with record_function("tri_interp"):
        interp_v = (
            c_000 * (1 - xyz_d[:, 0, :]) * (1 - xyz_d[:, 1, :]) * (1 - xyz_d[:, 2, :])
            + c_100 * xyz_d[:, 0, :] * (1 - xyz_d[:, 1, :]) * (1 - xyz_d[:, 2, :])
            + c_010 * (1 - xyz_d[:, 0, :]) * xyz_d[:, 1, :] * (1 - xyz_d[:, 2, :])
            + c_001 * (1 - xyz_d[:, 0, :]) * (1 - xyz_d[:, 1, :]) * xyz_d[:, 2, :]
            + c_101 * xyz_d[:, 0, :] * (1 - xyz_d[:, 1, :]) * xyz_d[:, 2, :]
            + c_011 * (1 - xyz_d[:, 0, :]) * xyz_d[:, 1, :] * xyz_d[:, 2, :]
            + c_110 * xyz_d[:, 0, :] * xyz_d[:, 1, :] * (1 - xyz_d[:, 2, :])
            + c_111 * xyz_d[:, 0, :] * xyz_d[:, 1, :] * xyz_d[:, 2, :]
        )

    # replace with mask
    interp_v.masked_fill_(oob_mask, outside_value)

    return interp_v.unsqueeze_(1)


def grid_trilinear_interp_cuda(
    points: torch.Tensor,
    origin: torch.Tensor,
    xyz_res: torch.Tensor,
    dims: torch.Tensor,
    data_grid: torch.Tensor,
    outside_value: float = 0,
):
    """Trilinear interpolation with CUDA"""
    raise NotImplementedError


def grid_trilinear_interp_cuda_texture(
    points: torch.Tensor,
    origin: torch.Tensor,
    xyz_res: torch.Tensor,
    dims: torch.Tensor,
    data_grid: torch.Tensor,
    outside_value: float = 0,
):
    """Trilinear interpolation with CUDA texture 3D"""
    raise NotImplementedError


def line_search_wolfe(x0, f0, grad0, d, f_func, grad_func, c1=0, c2=0.5, flag=None):
    """Line search enforcing weak Wolfe conditions.

    Args:
         x0: current point
         f0: current obj function value
         grad0: gradient of the obj function at current point
         d: descent direction
         f_func:
         grad_func:
         c1: Wolfe parameter for the sufficient decrease condition
             f(x0 + t d) ** < ** f0 + c1*t*grad0'*d (default=0)
         c2: Wolfe parameter for the WEAK condition on directional derivative
             (grad f)(x0 + t d)'*d ** > ** c2*grad0'*d  (default=0.5)
             where 0 <= c1 <= c2 <= 1.

    Returns:
         alpha:   steplength satisfying weak Wolfe conditions if one was found,
             otherwise left end point of interval bracketing such a point
             (possibly 0)
         xalpha:  x0 + alpha*d
         falpha:  f(x0 + alpha d)
         gradalpha:(grad f)(x0 + alpha d)
         fail:    0 if both Wolfe conditions satisfied, or falpha < fvalquit
                  1 if one or both Wolfe conditions not satisfied but an
                     interval was found bracketing a point where both satisfied
                 -1 if no such interval was found, function may be unbounded below
         beta:    same as alpha if it satisfies weak Wolfe conditions,
                   otherwise right end point of interval bracketing such a point
                   (inf if no such finite interval found)
         fevalrec:  record of function evaluations

    """
    device = x0.device
    B = x0.shape[0]
    n = x0.shape[2]
    n_all = 3 * n

    if flag is None:
        flag = torch.ones((B, 1), device=device)

    # lower bound on step length
    alpha = torch.zeros((B, 1), device=device)
    # upper bound on step length satisfying weak Wolfe conditions
    beta = torch.tensor(float("inf"), device=device).repeat(B, 1)

    xalpha = x0.clone()
    falpha = f0.clone()
    gradalpha = grad0.clone()

    # check direction validity
    g0 = torch.bmm(grad0.view(B, 1, n_all), d.view(B, n_all, 1)).view(B, 1)
    if torch.any(g0 >= 0):
        logging.warning("At least one batch's d is not a descent direction.")
    dnorm = torch.linalg.norm(d, dim=(1, 2)).view(B, 1)
    if not torch.all(dnorm):
        logging.error("d is zero for at least a batch.")

    # step length; try step length = 1 first (for all batches)
    t = torch.ones((B, 1), device=device)

    # bisect and expansion variables
    nbisect = torch.zeros((B, 1), device=device)
    nexpand = torch.zeros((B, 1), device=device)
    # nbisectmax: size (B, 1)
    nbisectmax = torch.maximum(torch.tensor(30.0, device=device), torch.round(torch.log2(1e5 * dnorm)))
    # nexpandmax: size (B, 1)
    nexpandmax = torch.maximum(torch.tensor(10.0, device=device), torch.round(torch.log2(1e5 / dnorm)))

    nfeval = 0
    done = torch.zeros((B, 1), dtype=torch.bool, device=device)
    done_flag = False
    fail = torch.ones((B, 1), dtype=torch.bool, device=device)
    while not done_flag:
        x = x0 + t.view((B, 1, 1)) * flag.view((B, 1, 1)) * d
        nfeval += 1
        f = f_func(x)
        grad = grad_func(x)
        gtd = torch.bmm(grad.view(B, 1, n_all), d.view(B, n_all, 1)).view(B, 1)

        # for batches where the first condition is not satisfied (sufficient decrease)
        # step length too large
        d_mask = torch.logical_or(f >= f0 + c1 * t * g0, torch.isnan(f))
        beta[d_mask] = t[d_mask]

        # for batches where the second condition is not satisfied (curvature)
        # step length not far enough
        c_mask = torch.logical_or(gtd <= c2 * g0, torch.isnan(f))
        alpha[c_mask] = t[c_mask]
        xalpha[c_mask.flatten(), :, :] = x[c_mask.flatten(), :, :]
        falpha[c_mask] = f[c_mask]
        gradalpha[c_mask.flatten(), :, :] = grad[c_mask.flatten(), :, :]

        # for batches where both conditions are satisfied
        # use de morgan's law on and(not(d_mask), not(c_mask))
        sat_mask = torch.logical_not(torch.logical_or(d_mask, c_mask))
        fail[sat_mask] = False
        alpha[sat_mask] = t[sat_mask]
        xalpha[sat_mask.flatten()] = x[sat_mask.flatten()]
        falpha[sat_mask] = f[sat_mask]
        gradalpha[sat_mask.flatten()] = grad[sat_mask.flatten()]
        beta[sat_mask] = t[sat_mask]

        # all batches satisfy both conditions
        # will break if all batches have fail = False
        if not torch.any(fail):
            return alpha, xalpha, falpha, gradalpha, fail, beta

        # at least one condition is not satisfied
        # so we need to set up step length for next iteration
        # 1. check beta: if beta is not infinity, that means there
        # is a finite upper bound on step length, so we bisect towards it.
        beta_mask = beta < torch.tensor(float("inf"), device=device)
        nbisect_mask = nbisect < nbisectmax

        # beta check satisfied & not reached maximum bisect limit yet
        to_bisect_mask = torch.logical_and(beta_mask, nbisect_mask)
        nbisect[to_bisect_mask] += 1
        t[to_bisect_mask] = (alpha[to_bisect_mask] + beta[to_bisect_mask]) / 2

        # beta check satisfied & reached maximum bisect limit
        not_to_bisect_mask = torch.logical_and(beta_mask, torch.logical_not(nbisect_mask))
        done[not_to_bisect_mask] = True

        # beta check not satisfied & not reached maximum expansion limit
        nexpand_mask = nexpand < nexpandmax
        to_expand_mask = torch.logical_and(torch.logical_not(beta_mask), nexpand_mask)
        nexpand[to_expand_mask] += 1
        t[to_expand_mask] = 2 * alpha[to_expand_mask]

        # beta check not satisfied & reached maximum expansion limit
        not_to_expand_mask = torch.logical_and(torch.logical_not(beta_mask), torch.logical_not(nexpand_mask))
        done[not_to_expand_mask] = True

        # check for whether all batches are done
        done_flag = torch.all(done)

    fail[torch.isinf(beta)] = True
    # if torch.any(torch.isinf(beta)):
    #    logging.error(
    #        "For at least one batch: line search failed to bracket point satisfying weak Wolfe condition."
    #        "Function may be unbounded below."
    #    )

    # if torch.any(fail):
    #    logging.error(
    #        "For at least one batch: line search failed to satisfy weak Wolfe conditions, "
    #        "although point satisfying conditions was bracketed."
    #    )

    return alpha, xalpha, falpha, gradalpha, fail, beta


def depth_to_point_cloud(depth, K):
    """Convert depth to point clouds"""
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    pts_im = np.vstack([us, vs]).T
    return pts, pts_im


def depth_to_point_cloud_map_batched(depth, intrinsics, grid_x, grid_y):
    """Return point cloud in (B, 3, H, W)"""
    B = intrinsics.shape[0]
    # calculate the x, y (lots of broadcasting)
    x = (
        (grid_x[None, None, ...] - intrinsics[:, 0, 0, 2].reshape((B, 1, 1, 1)))
        * depth
        / intrinsics[:, 0, 0, 0].reshape(B, 1, 1, 1)
    )
    y = (
        (grid_y[None, None, ...] - intrinsics[:, 0, 1, 2].reshape((B, 1, 1, 1)))
        * depth
        / intrinsics[:, 0, 1, 1].reshape(B, 1, 1, 1)
    )
    pts = torch.cat((x, y, depth), dim=1)
    return pts


def instance_depth_to_point_cloud_torch(depth, intrinsics, instance_mask=None):
    """Backproject depth map to point cloud given an instance mask. All valid depth pixels in the mask
    will be converted into point cloud.
    The final selected mask will also be returned.
    The returned point cloud dimension is (3, N).
    """
    assert instance_mask.dim() == depth.dim()
    if instance_mask is not None:
        final_instance_mask = torch.logical_and(instance_mask, depth > 0)
    else:
        final_instance_mask = depth > 0
    idxs = torch.nonzero(final_instance_mask, as_tuple=True)

    z = depth[idxs[0], idxs[1]]
    x = (idxs[1] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (idxs[0] - intrinsics[1, 2]) * z / intrinsics[1, 1]
    pts = torch.stack((x, y, z), dim=0)

    return pts, idxs


def depth_to_point_cloud_torch(depth, K, mask=None, x_index=1, y_index=0, device="cpu", pc_size=5000):
    """Non-batched version of converting depth map to point cloud, with specified pc size"""
    if mask is None:
        masked_depth = depth
    else:
        # elementwise multiplication; all non-masked pixels become zero
        # note that invalid depth measurements will also have depth equals to zero
        masked_depth = depth * mask.int().float()

    nonzeros = masked_depth.nonzero(as_tuple=True)
    if nonzeros[0].size(0) == 0:
        logging.warning("No nonzero depth values.")
        return None
    zs = depth[nonzeros[0], nonzeros[1]]
    us = nonzeros[x_index]
    vs = nonzeros[y_index]
    xs = ((us - K[0, 2]) * zs) / K[0, 0]
    ys = ((vs - K[1, 2]) * zs) / K[1, 1]
    pts = torch.stack([xs, ys, zs]).to(device=device)
    if pts.shape[-1] > pc_size:
        indices = torch.randperm(pts.shape[-1])[:pc_size]
        pts = pts[:, indices]
    else:
        # pad zeros to required size
        pts = torch.nn.functional.pad(pts, pad=(0, pc_size - pts.shape[-1], 0, 0), value=0.0)

    return pts


def depth_to_point_cloud_with_rgb_torch(depth, rgb, K, mask=None, x_index=1, y_index=0, device="cpu", pc_size=5000):
    """Extracting point cloud with corresponding RGB values

    Args:
        depth: (H, W) depth map
        rgb: (H, W, 3) RGB image
        K:
        mask:
        x_index:
        y_index:
        device:
        pc_size:
    """
    assert depth.shape[0] == rgb.shape[0]
    assert depth.shape[1] == rgb.shape[1]
    assert rgb.shape[-1] == 3

    if mask is None:
        masked_depth = depth
    else:
        # elementwise multiplication; all non-masked pixels become zero
        # note that invalid depth measurements will also have depth equals to zero
        masked_depth = depth * mask.int().float()

    nonzeros = masked_depth.nonzero(as_tuple=True)
    if nonzeros[0].size(0) == 0:
        logging.warning("No nonzero depth values.")
        return None
    # normalization based on image net
    rgb_temp = rgb.float() / 255.0
    rgb_temp[:, :, 0] = (rgb_temp[:, :, 0] - 0.485) / 0.229
    rgb_temp[:, :, 1] = (rgb_temp[:, :, 1] - 0.456) / 0.224
    rgb_temp[:, :, 2] = (rgb_temp[:, :, 2] - 0.406) / 0.225
    rgbs = rgb_temp[nonzeros[0], nonzeros[1], :]
    zs = depth[nonzeros[0], nonzeros[1]]
    us = nonzeros[x_index]
    vs = nonzeros[y_index]
    xs = ((us - K[0, 2]) * zs) / K[0, 0]
    ys = ((vs - K[1, 2]) * zs) / K[1, 1]
    pts = torch.stack([xs, ys, zs, rgbs[:, 0], rgbs[:, 1], rgbs[:, 2]]).to(device=device)
    if pts.shape[-1] > pc_size:
        indices = torch.randperm(pts.shape[-1])[:pc_size]
        pts = pts[:, indices]
    else:
        # pad zeros to required size
        pts = torch.nn.functional.pad(pts, pad=(0, pc_size - pts.shape[-1], 0, 0), value=0.0)

    return pts


def depth_to_point_cloud_batched(depth, K, mask=None, x_index=2, y_index=1, device="cpu", pc_size=5000):
    """Convert depth to point clouds, assuming batched input.
    Because each batch's depth point cloud may contain different numbers of non-zero points,
    we need to pad the output points

    Args:
        depth:
        K:
        mask: a batched binary torch tensor representing the pixels to select to generate point clouds
        x_index: the index at which the x-coordinate of the image frame changes
        y_index: the index at which the y-coordinate of the image frame changes
        device:
        pc_size: size of the point cloud, if the number of valid points is lower, patch zeros; if the number of valid
                 points is higher, sample
    """
    assert len(depth.shape) == 3
    B = depth.shape[0]

    if mask is None:
        masked_depth = depth
    else:
        # elementwise multiplication; all non-masked pixels become zero
        masked_depth = depth * mask

    pts = torch.zeros((B, 3, pc_size))
    for bid in range(B):
        pts[bid, :, :] = depth_to_point_cloud_torch(
            masked_depth[bid, :, :],
            K[bid, :, :],
            x_index=x_index - 1,
            y_index=y_index - 1,
            device=device,
            pc_size=pc_size,
        )

    return pts


def sq_half_chamfer_dists(pc, pc_):
    """Return squared half chamfer distances. All zero points will be ignored

    Inputs:
        pc  : torch.tensor of shape (B, 3, n)
        pc_ : torch.tensor of shape (B, 3, m)

    Output:
        loss    : (B, 1)
            returns max_loss if max_loss is true
    """
    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y
    return sq_dist.squeeze(-1)


def sq_half_chamfer_dists_clamped(pc, pc_, thres):
    """Clamped squared half chamfer distances"""
    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y
    sq_dist.clamp_(max=thres**2)
    return sq_dist.squeeze(-1)


def kNN_torch(query, dataset, k):
    """
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B, N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
    """
    # assert query.is_cuda and dataset.is_cuda, "Input tensors should be gpu tensors."                                  #Change: 22 Dec 2021: Commented this line out.
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert query.shape[0] == dataset.shape[0], "Input tensors should have same batch size."
    assert query.shape[2] == dataset.shape[2], "Input tensors should have same dimension."

    dists = square_distance(query, dataset)  # dists: [B, N0, N1]
    neighbors = dists.argsort()[:, :, :k]  # neighbors: [B, N0, k]
    return neighbors


def kNN_torch_fast(query, dataset, k):
    """use cdist in VF for faster knn
    inputs
        query: (B, N0, D) shaped torch gpu Tensor.
        dataset: (B, N1, D) shaped torch gpu Tensor.
        k: int
    outputs
        neighbors: (B, N0, k) shaped torch Tensor.
                   Each row is the indices of a neighboring points.
    """
    assert query.dim() == 3 and dataset.dim() == 3, "Input tensors should be 3D."
    assert query.shape[0] == dataset.shape[0], "Input tensors should have same batch size."
    assert query.shape[2] == dataset.shape[2], "Input tensors should have same dimension."

    dists = _VF.cdist(query, dataset, 2.0, 1)
    neighbors = torch.topk(dists, k, dim=-1, largest=False, sorted=True)  # neighbors: [B, N0, k]
    return neighbors[1]


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def o3d_uniformly_sample_then_average(o3d_mesh, total_number_of_points=50000):
    """Sample an Open3D mesh then use those points to compute the centroid"""
    pcd = o3d_mesh.sample_points_uniformly(number_of_points=total_number_of_points)
    mean = np.average(np.array(pcd.points), axis=0)
    return mean


def make_se3(R, t):
    """Turn R, t  to 4x4, nonbatched"""
    x = torch.zeros(4, 4).to(R.device)
    x[-1, -1] = 1.0
    x[:3, :3] = R
    x[:3, -1] = t.flatten()
    return x


def make_se3_batched(R, t):
    """Turn R, t  to 4x4"""
    x = torch.zeros(R.shape[0], 4, 4).to(R.device)
    x[:, -1, -1] = 1.0
    x[:, :3, :3] = R
    x[:, :3, -1] = t.squeeze(-1)
    return x


def make_scaled_se3_batched(s, R, t):
    """Turn s, R, t to 4x4"""
    x = torch.zeros(R.shape[0], 4, 4).to(R.device)
    x[:, -1, -1] = 1.0
    x[:, :3, :3] = s.reshape((R.shape[0], 1, 1)) * R
    x[:, :3, -1] = t.squeeze(-1)
    return x


def make_scaled_se3_inverse_batched(s: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
    """Return the 4x4 matrix that is the inverse of the (s, R, t) transform.
    That is,
    Given B = s * R * A + t,
    We have A = 1/s * R^T B - 1/s * R^T t
    And the returned T will be:
    T = [1/s * R^T, -1/s R^T; 0, 1]
    """
    assert R.shape[1] == R.shape[2] == 3 and s.shape[0] == R.shape[0] == t.shape[0]
    bs = R.shape[0]
    R_T = R.permute((0, 2, 1))
    inv_T = torch.zeros(bs, 4, 4).to(R.device).to(R.dtype)
    inv_T[:, -1, -1] = 1.0
    inv_T[:, :3, :3] = 1 / s.reshape((R.shape[0], 1, 1)) * R_T
    inv_T[:, :3, -1] = (-1 / s.reshape((R.shape[0], 1, 1)) * R_T @ t).squeeze(2)
    return inv_T


def se3_inverse_torch(T):
    """Invert an SE(3) matrix"""
    assert T.shape[0] == T.shape[1] == 4
    inv_T = torch.zeros(4, 4).to(T.device)
    inv_T[-1, -1] = 1
    # R' = R^T
    inv_T[:3, :3] = T[:3, :3].T
    # t' = -R^T t
    inv_T[:3, -1] = -inv_T[:3, :3] @ T[:3, -1]
    return inv_T


def se3_inverse_batched_torch(T):
    """Invert a batch of SE(3) matrix"""
    assert T.shape[1] == T.shape[2] == 4
    bs = T.shape[0]
    inv_T = torch.zeros(bs, 4, 4).to(T.device)
    inv_T[:, -1, -1] = 1
    # R' = R^T
    inv_T[:, :3, :3] = T[:, :3, :3].permute((0, 2, 1))
    # t' = -R^T t
    inv_T[:, :3, -1] = torch.bmm(-inv_T[:, :3, :3], T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return inv_T


def se3_inverse_numpy(T):
    assert T.shape[0] == T.shape[1] == 4
    R = T[:3, :3]
    t = T[:3, 3:]
    Rinv = R.transpose()
    tinv = -Rinv @ t

    Tinv = np.zeros_like(T)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3:] = tinv
    return Tinv


def get_max_min_intra_pts_dists(pts):
    """Get the maximum and minimum intra points distances"""
    intra_kp_dists = torch.cdist(torch.transpose(pts, -1, -2), torch.transpose(pts, -1, -2), p=2)
    half_dists = intra_kp_dists[torch.triu(intra_kp_dists, diagonal=1) != 0]
    max_dist = torch.max(half_dists).item()
    min_dist = torch.min(half_dists).item()
    return max_dist, min_dist


def project_SO3(A):
    """Project a batched matrix to SO(3)"""
    U, S, Vh = torch.linalg.svd(A)
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    temp = U.clone()
    temp[:, :, -1] *= d[:, None]
    R = temp @ Vh
    return R, U, S, Vh, d


def SO3_from_axis_angle(unit_axis: torch.Tensor, angle):
    """Generate a rotation matrix given a unit axis and an angle"""
    if angle == 0:
        return torch.eye(3).to(unit_axis.device)
    else:
        so3 = torch.tensor(
            [[0, -unit_axis[2], unit_axis[1]], [unit_axis[2], 0, -unit_axis[0]], [-unit_axis[1], unit_axis[0], 0]]
        ).to(unit_axis.device)
        R = torch.eye(3) + so3 * torch.sin(angle) + (so3 @ so3) * (1 - torch.cos(angle))
        return R


@torch.jit.script
def batch_var(points: torch.Tensor, weights: torch.Tensor, correction: float = 0):
    """Batched variance with weights. PyTorch unfortunately does not have a weighted variance function.

    Parameters
    ----------
    points: (B, D, N)
    weights: (B, N)
    """
    B, D, N = points.shape
    # (B, D)
    weighted_mean = torch.sum(points * weights.view(B, 1, N), dim=2) / weights.sum(dim=1, keepdim=True)
    # (B, D, N)
    diffs = points - weighted_mean.view(B, D, 1)
    # (B, D, N)
    prods = diffs * weights.view(B, 1, N) * diffs
    # (B, D)
    numerator = torch.sum(prods, dim=2)
    denom = torch.sum(weights, dim=1, keepdim=True) - correction
    return numerator / denom


def sample_within_nonzero_masks(samples, masks: torch.Tensor, padding=False):
    """Sample nonzero indices given some binary masks.

    Parameters
    ----------
    samples: number of points to sample (the same across entire batch)
    masks: (B, N) binary masks
    padding: if True, if samples > all unique nonzeros, padding values of 0 will be added. Otherwise, the returned
    indices will contain repeated entries.

    Returns
    -------
    Tensor with size (B, count) containing indices,
    Tensor with size (B, count) representing the mask of valid indices (will contain zeros only when padding=True)
    Tensor with size (B, ) representing the number of valid sample indices in the returned indices
    """
    assert masks.ndim == 2
    B, N = masks.shape[0], masks.shape[1]
    # if we sort the mask, the 1s (nonzero) will be at the front, and we can simply sample from them by generating
    # random integers
    sorted_nonzero_indices = masks.sort(descending=True)[1]
    nonzeros = torch.count_nonzero(masks, dim=1)  # (B,)

    final_idx = torch.zeros(size=(B, samples), device=masks.device).long()
    valid_indices_mask = torch.ones(size=(B, samples), device=masks.device).bool()
    valid_indices_cnt = torch.zeros(size=(B,), device=masks.device).int()

    for bid in range(B):
        nz_cnt = nonzeros[bid].cpu().item()
        if padding == False:
            # we are okay with the repeated values in the returned indices
            if nz_cnt == 0:
                # if there's nonzero, return None as an error
                return None
            rand_idx = torch.randint(low=0, high=nz_cnt, size=(samples,)).to(masks.device)
            final_idx[bid, ...] = torch.gather(sorted_nonzero_indices[bid, ...], 0, rand_idx.long())
            valid_indices_cnt[bid, ...] = samples
        else:
            # we are not okay with the repeated values in the returned indices
            if nz_cnt != 0:
                if nz_cnt >= samples:
                    # TODO: Use multinomial here
                    rand_idx = torch.randint(low=0, high=nz_cnt, size=(samples,)).to(masks.device)
                    final_idx[bid, ...] = torch.gather(sorted_nonzero_indices[bid, ...], 0, rand_idx.long())
                    valid_indices_cnt[bid, ...] = samples
                else:
                    # all indices are used as sample; missing indices are zero
                    final_idx[bid, :nz_cnt] = torch.gather(
                        sorted_nonzero_indices[bid, ...], 0, torch.arange(nz_cnt).to(masks.device).long()
                    )
                    valid_indices_mask[bid, nz_cnt:] = 0
                    valid_indices_cnt[bid, ...] = nz_cnt

    return final_idx, valid_indices_mask, valid_indices_cnt


def H(n):
    """Returns an approximate value of n-th harmonic number.
       http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def project_simplex(V, z=1):
    """
    V: (num_samples, feature_dim)
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V

    Credit: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Paper: https://www.jmlr.org/papers/volume7/shalev-shwartz06a/shalev-shwartz06a.pdf
    """
    n_features = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    z = np.ones(len(V)) * z
    cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
    ind = np.arange(n_features) + 1
    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V)), rho - 1] / rho
    return np.maximum(V - theta[:, np.newaxis], 0)
