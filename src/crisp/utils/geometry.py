import logging
import igl
import trimesh
import numpy as np
from scipy.spatial.distance import cdist, euclidean


def sample_pts_and_normals(mesh, count, interp=True):
    """Sample points with normals. Credit: https://github.com/mikedh/trimesh/issues/1285

    Parameters
    ----------
    mesh
    count
    interp : Set True to interpolate the normals with barycentric interpolation. Otherwise just take the face normals.

    """
    samples, fid = mesh.sample(count, return_index=True)
    if not interp:
        return samples, mesh.face_normals[fid]
    else:
        # compute the barycentric coordinates of each sample
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=samples)
        # interpolate vertex normals from barycentric coordinates
        interp = trimesh.unitize(
            (mesh.vertex_normals[mesh.faces[fid]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1)
        )
        return samples, interp


def sample_pts(mesh, count):
    """Sample points with normals. Credit: https://github.com/mikedh/trimesh/issues/1285

    Parameters
    ----------
    mesh
    count
    interp : Set True to interpolate the normals with barycentric interpolation. Otherwise just take the face normals.

    """
    samples = mesh.sample(count, return_index=False)
    return samples


def normalize_points_to_cube(coords, centroid=None, keep_aspect_ratio=True, center_at=None, cube_scale=1):
    """Normalize coordinates into a (unit) cube.

    Parameters
    ----------
    coords : (N, 3)
    keep_aspect_ratio : Set True to keep the 3D aspect ratio of the point clouds.
    center_at : Location of the center of the point cloud. By default center at origin.
    cube_scale : Scale of the cube. Default to unit cube (each side has length one).
    """
    if center_at is None:
        center_at = np.array([0, 0, 0])

    # zero the coordinates
    if centroid is None:
        og_center = np.mean(coords, axis=0, keepdims=True)
    else:
        og_center = centroid
    coords -= og_center

    coord_max = np.amax(coords, axis=0, keepdims=True)
    coord_min = np.amin(coords, axis=0, keepdims=True)
    max_dist = np.abs(coord_max - coord_min)
    if keep_aspect_ratio:
        max_dist = np.amax(max_dist)

    # to unit or scaled cube center at origin
    scale = max_dist / cube_scale
    coords = coords / scale

    # offset the center if necessary
    coords += center_at

    return coords, {"final_offset": center_at, "og_scale": scale, "og_center": og_center}


def geometric_median(X, eps=1e-5, max_iter=100):
    """
    Compute the geometric median of points in R^N.

    Based on: https://stackoverflow.com/a/30305181

    Credit:
    Vardi, Yehuda, and Cun-Hui Zhang. "The multivariate L 1-median and associated data depth."
    Proceedings of the National Academy of Sciences 97.4 (2000): 1423-1426.

    Parameters
    ----------
    X: (N, D) array
    eps
    max_iter
    """
    y = np.mean(X, 0)

    for ii in range(max_iter):
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) / np.linalg.norm(y) < eps:
            return y1

        y = y1

    logging.warning("Geometric median did not converge.")
    return y


def query_sdf_from_mesh(query: np.ndarray, mesh: trimesh.Trimesh):
    """Given a set of query points, return their corresponding SDF values.
    Uses the fast winding number method described in:

    Barill, Gavin, et al. "Fast winding numbers for soups and clouds."
    ACM Transactions on Graphics (TOG) 37.4 (2018): 1-12.

    Parameters
    ----------
    query
    mesh
    """
    assert query.shape[1] == 3
    V, F = mesh.vertices, mesh.faces
    S, I, C = igl.signed_distance(
        query, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER, return_normals=False
    )
    return S


def voxelize_cube(N, cube_center: np.ndarray, cube_scale=1.0):
    """
    Parameters
    ----------
    N: number of voxels per side
    cube_center
    cube_scale

    Returns
    -------
    (n, 3) array of voxel coordinates
    """
    assert len(cube_center) == 3

    voxel_origin = cube_center - cube_scale / 2
    voxel_size = cube_scale / (N - 1)

    # use longlong to avoid overflow
    overall_index = np.arange(0, N**3, 1, dtype=np.longlong)
    # first 3 columns: coordinates; last column: SDF values
    samples = np.zeros((N**3, 3))

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index / N) % N
    samples[:, 0] = ((overall_index / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples, voxel_size, voxel_origin


def sample_within_cube(N, cube_center: np.ndarray, cube_scale=1.0):
    """Sample N points uniformly within a cube"""
    assert len(cube_center) == 3

    voxel_origin = cube_center - cube_scale / 2
    low_bound, high_bound = voxel_origin, voxel_origin + cube_scale

    samples = np.zeros((N, 3))
    samples[:, 0] = np.random.uniform(low_bound[0], high_bound[0], N)
    samples[:, 1] = np.random.uniform(low_bound[1], high_bound[1], N)
    samples[:, 2] = np.random.uniform(low_bound[2], high_bound[2], N)

    return samples
