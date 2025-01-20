"""
Thanks to Jingnan and Henk for this piece of code. We have used it only for verifying the correctness of our PACE implementation.
"""


import cvxpy as cp
import numpy as np
from itertools import combinations, product
import networkx as nx
import scipy as scipy
import os
import sys

sys.path.append("../")
from learning_objects.utils.sdp_data import get_rotation_relaxation_constraints, get_vectorization_permutation
# import robin_py

ORDER_NOT_COVISIBLE = -1
ORDER_IJK = 1
ORDER_IKJ = 2
ORDER_BOTH = 3
ORDER_UNKNOWN = 4


def compute_lam(lam_base, N, K):
    return lam_base * np.sqrt(float(K) / float(N))


def inlier_stats(A, B):
    '''
    A contains the indices of ground-truth inliers
    B contains the indices of robin estimated inliers
    '''
    common = np.intersect1d(A, B)
    if B.shape[0] > 0:
        B_inlier_rate = float(common.shape[0]) / float(B.shape[0])
    else:
        B_inlier_rate = 0

    num_A_and_B = common.shape[0]
    num_A_not_in_B = A.shape[0] - common.shape[0]
    num_B_not_in_A = B.shape[0] - common.shape[0]

    return np.array([B_inlier_rate, num_A_and_B, num_A_not_in_B, num_B_not_in_A])


def rotation_error(R0, R1):
    return np.abs(
        np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2.0, -0.999999,
                          0.999999))) / np.pi * 180


def translation_error(t0, t1):
    return np.linalg.norm(t0 - t1)


def project_to_SO3(A):
    U, S, Vh = np.linalg.svd(A)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vh

    # RRtran = R @ R.T
    # assert(
    #     np.linalg.norm(RRtran-np.identity(3),ord='fro')<1e-12,'Projection to SO3 failed')
    return R


def decompose_nonrigid_model(A):
    '''
    If A is the model for a nonrigid registration problem, then decompose A into
    9 basis shapes such that the affine transformation applied on A is equivalent
    to a linear combination of the 9 basis shapes
    '''
    N = A.shape[1]
    zero = np.zeros((1, N))
    row1 = [A[0, :]]
    row2 = [A[1, :]]
    row3 = [A[2, :]]
    A1 = np.concatenate((
        row1, zero, zero), axis=0)
    A2 = np.concatenate((
        zero, row1, zero), axis=0)
    A3 = np.concatenate((
        zero, zero, row1), axis=0)
    A4 = np.concatenate((
        row2, zero, zero), axis=0)
    A5 = np.concatenate((
        zero, row2, zero), axis=0)
    A6 = np.concatenate((
        zero, zero, row2), axis=0)
    A7 = np.concatenate((
        row3, zero, zero), axis=0)
    A8 = np.concatenate((
        zero, row3, zero), axis=0)
    A9 = np.concatenate((
        zero, zero, row3), axis=0)

    A_basis = np.asarray([A1, A2, A3, A4, A5, A6, A7, A8, A9])

    return A_basis


def basis_to_cads(A_basis):
    K = A_basis.shape[0]
    cad_db = []
    for i in range(K):
        cad = {'kpts': np.squeeze(A_basis[i, :, :])}
        cad_db.append(cad)
    return cad_db


def minimum_distance_to_convex_hull(A):
    '''
    A is shape 3 by K, compute the minimum distance from the origin to the convex hull of A
    '''
    K = A.shape[1]
    P = A.T @ A
    one = np.ones((K, 1))
    # Use CVXPY to solve
    x = cp.Variable(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)),
                      [x >= 0,
                       one.T @ x == 1])
    prob.solve(solver='ECOS', verbose=False)
    x_val = x.value
    min_distance = np.linalg.norm(A @ x_val)
    return min_distance


def compute_min_max_distances(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij


def compute_min_max_distances_with_idx_maps(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij, si, sj


def test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db):
    """Calculate the 2D winding order and compare with the order database
    Assume triplet_points are 2-by-3,
    """
    sort_index = np.argsort(list(triplet_semantic_ids))
    sorted_triplet_semantic_ids = tuple(np.array(triplet_semantic_ids)[sort_index])
    sorted_triplet_points = triplet_points[:, sort_index]
    expected_order = order_db[sorted_triplet_semantic_ids]
    if expected_order == ORDER_BOTH:
        return True
    if expected_order == ORDER_NOT_COVISIBLE:
        return False
    v1 = sorted_triplet_points[:, 1] - sorted_triplet_points[:, 0]
    v2 = sorted_triplet_points[:, 2] - sorted_triplet_points[:, 0]
    mat = np.vstack((v1, v2)).T
    # note: the opencv coordinate system is left handed
    # ijk: determinant negative
    # ikj: determinant positive
    det = np.linalg.det(mat)
    measured_order = 0
    if det < 0:
        measured_order = ORDER_IJK
    elif det > 0:
        measured_order = ORDER_IKJ
    else:
        # det == 0
        # Either: collinear points, or some points share the same coordinates
        # in this case, the order can be either, and we always add edges (hence always return True)
        return True
    return measured_order == expected_order


def robin_prune_outliers_triplet_order_invariant(tgt, tgt_semantic_id_map, order_db, method='maxclique'):
    assert (tgt.shape[0] == 2)
    assert (tgt.shape[1] == len(tgt_semantic_id_map))
    num_keypoints = tgt.shape[1]
    all_triplets = combinations(list(range(num_keypoints)), 3)

    # creating a Graph in robin
    g = robin_py.AdjListGraph()
    # also create a graph in networkx
    g_nx = nx.Graph()
    for i in range(num_keypoints):
        g.AddVertex(i)
        g_nx.add_node(i)

    # build the compatibility graph
    for triplet in all_triplets:
        # convert triplet indices to semantic ids
        triplet_semantic_ids = (
            tgt_semantic_id_map[triplet[0]], tgt_semantic_id_map[triplet[1]], tgt_semantic_id_map[triplet[2]])
        triplet_points = np.vstack((tgt[:, triplet[0]], tgt[:, triplet[1]], tgt[:, triplet[2]])).T
        consistent = test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db)
        if consistent:
            # add edges to robin graph
            g.AddEdge(triplet[0], triplet[1])
            g.AddEdge(triplet[0], triplet[2])
            g.AddEdge(triplet[1], triplet[2])
            # add edges to nx graph
            g_nx.add_edge(triplet[0], triplet[1])
            g_nx.add_edge(triplet[0], triplet[2])
            g_nx.add_edge(triplet[1], triplet[2])

    # run robin
    if method == "maxclique":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CLIQUE)
    elif method == "maxcore":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CORE)
    else:
        raise RuntimeError('Prune outliers only support maxclique and maxcore')
    inlier_indices.sort()
    return [tgt_semantic_id_map[i] for i in inlier_indices], inlier_indices, g_nx


def robin_prune_outliers(tgt, cad_dist_min, cad_dist_max, noise_bound, method='maxclique'):
    '''
    First form a compatibility graph and then
    Use robin to select inliers
    '''
    N = tgt.shape[1]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    # distances || tgt_j - tgt_i ||
    tgt_dist_ij = np.linalg.norm(
        tgt[:, sj] - tgt[:, si], axis=0)  # shape (n-1)_tri

    allEdges = np.arange(si.shape[0])
    check1 = tgt_dist_ij >= (cad_dist_min - 2 * noise_bound)
    check2 = tgt_dist_ij <= (cad_dist_max + 2 * noise_bound)
    mask_compatible = check1 & check2
    validEdges = allEdges[mask_compatible]
    sdata = np.zeros_like(si)
    sdata[mask_compatible] = 1

    comp_mat = np.zeros((N, N))
    comp_mat[si, sj] = sdata

    # creating a Graph in robin
    g = robin_py.AdjListGraph()
    for i in range(N):
        g.AddVertex(i)

    for edge_idx in validEdges:
        # print(f'Add edge between {si[edge_idx]} and {sj[edge_idx]}.')
        g.AddEdge(si[edge_idx], sj[edge_idx])

    if method == "maxclique":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CLIQUE)
    elif method == "maxcore":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CORE)
    else:
        raise RuntimeError('Prune outliers only support maxclique and maxcore')

    # adj_mat = g.GetAdjMat()

    return inlier_indices, comp_mat


def solve_wahba(A, B):
    '''
    solve the Wahba problem using SVD
    '''
    M = B @ A.T
    R = project_to_SO3(M)
    return R


def solve_3dcat_with_altern(tgt, cad_kpts, lam=0.0, weights=None, enforce_csum=True, print_info=False,
                            normalize_lam=False):
    '''
    Solver weighted outlier-free category registration using alternating minimization
    '''
    N = tgt.shape[1]
    K = cad_kpts.shape[0]
    if weights is None:
        weights = np.ones(N)

    if normalize_lam:
        lam_old = lam
        lam = lam_old * float(N) / float(K)
        print(f'Altern: normalize LAM: {lam_old} --> {lam}.')

    if K > 3 * N and lam == 0.0:
        raise RuntimeError('If K is larger than 3N, then lambda has to be strictly positive.')

    wsum = np.sum(weights)
    wsqrt = np.sqrt(weights)

    # compute weighted centers of tgt and cad_kpts
    # tgt has shape 3 by N
    # cad_kpts has shape K by 3 by N
    y_w = tgt @ weights / wsum  # y_w shape (3,)
    b_w = np.sum(
        cad_kpts * weights[np.newaxis, np.newaxis, :], axis=2) / wsum  # b_w shape (K,3)
    # compute relative positions
    Ybar = (tgt - y_w[:, np.newaxis]) * wsqrt[np.newaxis, :]
    ybar = np.reshape(Ybar, (3 * N,), order='F')  # vectorize
    B = (cad_kpts - b_w[:, :, np.newaxis]) * wsqrt[np.newaxis, np.newaxis, :]  # B shape (K,3,N)
    Bbar = np.transpose(
        np.reshape(B, (K, 3 * N), order='F'))  # Bbar shape (3N, K)
    IN = np.identity(N)
    IK = np.identity(K)

    if enforce_csum:
        e = np.ones((K, 1))
        H11 = 2 * (Bbar.T @ Bbar + lam * IK)
        H11inv = np.linalg.inv(H11)
        scalar = e.T @ H11inv @ e
        vector = H11inv @ e
        G = H11inv - (vector @ vector.T) / scalar
        g = vector / scalar
    else:
        H = Bbar.T @ Bbar + lam * IK
        Hinv = np.linalg.inv(H)

    # Initialize
    R = np.identity(3)
    c = np.zeros(K)
    prev_cost = np.inf
    epsilon = 1e-12
    MaxIters = 1e3
    itr = 0
    while itr < MaxIters:
        # Fix R, update c
        if enforce_csum:
            c = 2 * (G @ Bbar.T @ np.kron(IN, R.T) @ ybar) + np.squeeze(g)
        else:
            c = Hinv @ Bbar.T @ np.kron(IN, R.T) @ ybar

        # Fix c, update R
        shape_c = np.sum(
            cad_kpts * c[:, np.newaxis, np.newaxis], axis=0)
        R = solve_wahba(shape_c, Ybar)

        # Compute cost and check convergence
        diff = Ybar - R @ shape_c
        cost = np.sum(diff ** 2) + lam * np.sum(c ** 2)
        cost_diff = np.abs(cost - prev_cost)
        if cost_diff < epsilon:
            if print_info:
                print(f'Altern converges in {itr} iterations, cost diff: {cost_diff}, final cost: {cost}.')
            break
        if itr == MaxIters - 1 and cost_diff > epsilon:
            if print_info:
                print(f'Altern does not converge in {MaxIters} iterations, cost diff: {cost_diff}, final cost: {cost}.')

        itr = itr + 1
        prev_cost = cost

    t = y_w - R @ (b_w.T @ c)

    residuals = calc_residuals(R, t, c, tgt, cad_kpts)

    return R, t, c, itr, residuals


## This is the code ##
def solve_3dcat_with_sdp(tgt, cad_kpts, lam=0.0, weights=None, enforce_csum=True, print_info=True, normalize_lam=False):
    '''
    Solve weighted outlier-free category registration using SDP relaxation
    inputs:
    tgt: numpy.array of shape (3, N)
    cad_kpts: numpy.array of shape (K, 3, N)

    outputs:
    R: numpy.array of shape (3, 3)
    t: numpy.array of shape(3, )
    c: numpy.array of shape (K, )

    '''

    P = get_vectorization_permutation()

    N = tgt.shape[1]
    K = cad_kpts.shape[0]
    if weights is None:
        weights = np.ones(N)

    if normalize_lam:
        lam_old = lam
        lam = lam_old * float(N) / float(K)
        print(f'SDP: normalize LAM: {lam_old} --> {lam}.')

    if K > 3 * N and lam == 0.0:
        raise RuntimeError('If K is larger than 3N, then lambda has to be strictly positive.')

    wsum = np.sum(weights)
    wsqrt = np.sqrt(weights)

    # compute weighted centers of tgt and cad_kpts
    # tgt has shape 3 by N
    # cad_kpts has shape K by 3 by N
    y_w = tgt @ weights / wsum  # y_w shape (3,)
    b_w = np.sum(
        cad_kpts * weights[np.newaxis, np.newaxis, :], axis=2) / wsum  # b_w shape (K,3)
    # compute relative positions
    Ybar = (tgt - y_w[:, np.newaxis]) * wsqrt[np.newaxis, :]
    ybar = np.reshape(Ybar, (3 * N,), order='F')  # vectorize
    B = (cad_kpts - b_w[:, :, np.newaxis]) * wsqrt[np.newaxis, np.newaxis, :]  # B shape (K,3,N)
    Bbar = np.transpose(
        np.reshape(B, (K, 3 * N), order='F'))  # Bbar shape (3N, K)

    IK = np.identity(K)
    I3N = np.identity(3 * N)
    I3 = np.identity(3)
    if enforce_csum:
        # compute H, Hinv and G, g
        e = np.ones((K, 1))
        H11 = 2 * (Bbar.T @ Bbar + lam * IK)
        H11inv = np.linalg.inv(H11)
        scalar = e.T @ H11inv @ e
        vector = H11inv @ e
        G = H11inv - (vector @ vector.T) / scalar
        g = vector / scalar
        # compute M and h
        M = np.concatenate((
            2 * Bbar @ G @ Bbar.T - I3N,
            2 * np.sqrt(lam) * G @ Bbar.T), axis=0)
        h = np.concatenate((
            Bbar @ g, np.sqrt(lam) * g), axis=0)  # 2021 Oct-11: added missing sqrt(lam) before g
    else:
        Hinv = np.linalg.inv(Bbar.T @ Bbar + lam * IK)
        M = np.concatenate((
            Bbar @ Hinv @ Bbar.T - I3N, np.sqrt(lam) * Hinv * Bbar.T), axis=0)
        h = np.zeros((3 * N + K, 1))  # 2021 Oct-11: added missing lam * c'*c

    # b_w, Bbar
    # G, g, M, h


    YkI3 = np.kron(Ybar.T, I3)
    Q = np.block([
        [h.T @ h, h.T @ M @ YkI3 @ P],
        [P.T @ YkI3.T @ M.T @ h, P.T @ YkI3.T @ M.T @ M @ YkI3 @ P]])

    A, b = get_rotation_relaxation_constraints()
    m = len(A)

    # mdic = {
    #     'A': A,
    #     'b': b,
    #     'C': Q
    # }
    # scipy.io.savemat('sample.mat',mdic)

    # Use CVXPY to solve a standard linear SDP
    n = 10
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(m)]
    prob = cp.Problem(
        cp.Minimize(cp.trace(Q @ X)), constraints)
    prob.solve()

    Xval = X.value

    if Xval is None:
        print(f'LAM: {lam}, Weights:')
        print(weights)
        raise ArithmeticError("SDP solver failed.")

    # Round solution and check tightness of relaxation
    f_sdp = np.trace(Q @ Xval)

    ev, evec = np.linalg.eig(Xval)
    idx = np.argsort(ev)
    evmax = ev[idx[-1]]
    evsmax = ev[idx[-2]]
    vec = evec[:, idx[-1]]
    vec = vec / vec[0]
    r = vec[1:]
    R = project_to_SO3(
        np.reshape(r, (3, 3), order='F'))
    r = np.reshape(R, (9, 1), order='F')
    rhomo = np.concatenate((
        np.array([[1.0]]), r), axis=0)
    f_est = np.squeeze(
        rhomo.T @ Q @ rhomo)

    rel_gap = abs(f_est - f_sdp) / f_est
    if print_info:
        print(f'SDP relax: lam_1={evmax},lam_2={evsmax},f_sdp={f_sdp},f_est={f_est},rel_gap={rel_gap},lam={lam}.')

    # Recover optimal c and t from R
    IN = np.identity(N)
    if enforce_csum:
        c = 2 * (G @ Bbar.T @ np.kron(IN, R.T) @ ybar) + np.squeeze(g)
    else:
        c = Hinv @ Bbar.T @ np.kron(IN, R.T) @ ybar

    t = y_w - R @ (b_w.T @ c)

    residuals = calc_residuals(R, t, c, tgt, cad_kpts)

    # return R, t, c, rel_gap, residuals, b_w, Bbar, G, g, M, h, y_w, Ybar, Q # this is for verification
    return R, t, c, rel_gap, residuals


def calc_residuals(R, t, c, tgt, cad_kpts):
    '''
    Calculate non-squared residuals
    '''
    shape_est = np.sum(
        cad_kpts * c[:, np.newaxis, np.newaxis], axis=0)
    shape_transform = R @ shape_est + t[:, np.newaxis]
    residuals = np.sqrt(
        np.sum(
            (tgt - shape_transform) ** 2, axis=0))

    return residuals


def solve_3dcat(tgt, cad_db, noise_bound, lam=0.0, gnc=True, div_factor=1.4, enforce_csum=True, normalize_lam=False,
                solver_type='sdp'):
    """ GNC category level registration (3D-3D)
    :argument tgt Input target points
    :argument cad_db Input library of CAD models
    :argument noise_bound maximum allowed inlier residual (non-squared residual)
    :argument lam regularization factor for shape coefficients, lam >= 0
    :argument gnc flag selects if GNC-TLS is used to be robust to outliers (default True)
    :argument div_factor GNC continuation factor (default 1.4)
    :argument enforce_csum flag to enforce the sum of all coefficients equal to 1
    :return: solution dictionary containing estimated R, t, c, and residuals
    """
    if solver_type not in ["sdp", "altern"]:
        raise ValueError("Unrecognized solver: {}".format(solver_type))
    # number of key points
    N = tgt.shape[1]

    if isinstance(cad_db, list):
        K = len(cad_db)
        assert N == cad_db[0]['kpts'].shape[1]
        # If cad_db is a list, then obtain keypoints from list
        cad_kpts = []
        for i in range(len(cad_db)):
            cad_kpts.append(cad_db[i]['kpts'])
        cad_kpts = np.array(cad_kpts)
    else:  # Otherwise, cad_db is already in kpts format as a np array
        cad_kpts = cad_db
        K = cad_kpts.shape[0]
        assert N == cad_kpts.shape[2]

    # If outlier free
    if not gnc:
        if solver_type == "sdp":
            R, t, c, _, residuals = solve_3dcat_with_sdp(tgt, cad_kpts, lam=lam, weights=None,
                                                         enforce_csum=enforce_csum, normalize_lam=normalize_lam)
            solution = {
                'type': 'category level registration',
                'method': 'sdp',
                'estimate': (R, t, c),
                'residuals': residuals
            }
        elif solver_type == "altern":
            R, t, c, _, residuals = solve_3dcat_with_altern(tgt, cad_kpts, lam=lam, weights=None,
                                                            enforce_csum=enforce_csum, normalize_lam=normalize_lam)
            solution = {
                'type': 'category level registration',
                'method': 'sdp',
                'estimate': (R, t, c),
                'residuals': residuals
            }

    # Default use GNC-TLS
    else:
        weights = np.ones(N)
        stop_th = 1e-6
        max_steps = 1e2
        barc2 = 1.0
        itr = 0

        pre_TLS_cost = np.inf
        cost_diff = np.inf

        solver_func = solve_3dcat_with_sdp
        if solver_type == "altern":
            solver_func = solve_3dcat_with_altern

        while itr < max_steps and cost_diff > stop_th:
            if np.sum(weights) < 1e-12:
                print('GNC encounters numerical issues, the solution is likely to be wrong.')
                break

            #  fix weights and solve for transformation
            R, t, c, _, residuals = solver_func(tgt, cad_kpts, lam=lam, weights=weights, enforce_csum=enforce_csum,
                                                print_info=False, normalize_lam=normalize_lam)

            #  fix transformations and update weights
            residuals = residuals / noise_bound
            residuals = residuals ** 2  # residuals normalized by noise_bound

            TLS_cost = np.inner(weights, residuals)
            cost_diff = np.abs(TLS_cost - pre_TLS_cost)

            if itr < 1:
                max_residual = np.max(residuals)
                mu = max(1 / (5 * max_residual / barc2 - 1), 1e-4)
                # mu = 1e-3
                print(f'GNC first iteration max residual: {max_residual}, set mu={mu}.')

            th1 = (mu + 1) / mu * barc2
            th2 = (mu) / (mu + 1) * barc2

            prev_weights = np.copy(weights)

            for i in range(N):
                if residuals[i] - th1 >= 0:
                    weights[i] = 0
                elif residuals[i] - th2 <= 0:
                    weights[i] = 1.0
                else:
                    weights[i] = np.sqrt(
                        barc2 * mu * (mu + 1) / residuals[i]) - mu
                    assert (weights[i] >= 0 and weights[i] <= 1)

            weights_diff = np.linalg.norm(weights - prev_weights)
            weights_sum = np.sum(weights)

            # print('Residuals unsquared:')
            # print(np.sqrt(residuals))
            # print('Weights updated to:')
            # print(weights)

            # print(f'Itr: {itr}, weights_diff: {weights_diff}, weights_sum: {weights_sum}, cost_diff: {cost_diff}.')

            #  increase mu
            mu = mu * div_factor
            itr = itr + 1
            pre_TLS_cost = TLS_cost

        solution = {
            'type': 'category level registration',
            'method': 'gnc',
            'estimate': (R, t, c),
            'max_steps': max_steps,
            'weights': weights,
            'itr': itr,
            'div_factor': div_factor
        }

    # calculate final residuals and add to solution dictionary
    residuals = calc_residuals(R, t, c, tgt, cad_kpts)
    normalized_residuals = residuals / noise_bound

    solution['residuals'] = residuals
    solution['normalized_residuals'] = normalized_residuals

    return solution


def solve_3dcat_irls(tgt, cad_db, noise_bound, lam=0.0, enforce_csum=True, robust_fun='TLS', normalize_lam=False):
    '''
    Solve robust estimation using iterative reweighted least squares
    '''
    # number of key points
    N = tgt.shape[1]

    if isinstance(cad_db, list):
        K = len(cad_db)
        assert N == cad_db[0]['kpts'].shape[1]
        # Obtain keypoints from cad_db
        cad_kpts = []
        for i in range(len(cad_db)):
            cad_kpts.append(cad_db[i]['kpts'])
        cad_kpts = np.array(cad_kpts)
    else:
        cad_kpts = cad_db
        K = cad_kpts.shape[0]
        assert N == cad_kpts.shape[2]

    weights = np.ones(N)
    stop_th = 1e-6
    max_steps = 1e3
    barc2 = 1.0
    if robust_fun == "GM":
        max_steps = 100
        sigma = 1.0 * barc2

    itr = 0
    prev_cost = np.inf
    cost_diff = np.inf

    while itr < max_steps and cost_diff > stop_th:
        if np.sum(weights) < 1e-12:
            print('IRLS encounters numerical issues, the solution is likely to be wrong.')
            break

        #  fix weights and solve for transformation
        R, t, c, _, residuals = solve_3dcat_with_sdp(tgt, cad_kpts, lam=lam, weights=weights, enforce_csum=enforce_csum,
                                                     print_info=False, normalize_lam=normalize_lam)

        #  calculate residuals
        residuals = residuals / noise_bound
        residuals_sq = residuals ** 2

        if robust_fun == "TLS":
            # Check convergence
            cost = np.inner(weights, residuals)
            cost_diff = np.abs(cost - prev_cost)
            # Update weights
            weights = np.zeros(N)
            weights[residuals < barc2] = 1.0
        elif robust_fun == "GM":
            # Check convergence
            cost = np.sum(residuals_sq / (sigma + residuals_sq))
            cost_diff = np.abs(cost - prev_cost)
            # Update weights
            weights = sigma ** 2 / ((sigma + residuals_sq) ** 2)
        else:
            raise RuntimeError('IRLS only supports TLS and GM function now.')

        prev_cost = cost
        itr = itr + 1

    solution = {
        'type': 'category level registration',
        'method': 'irls',
        'robust_fun': robust_fun,
        'estimate': (R, t, c),
        'max_steps': max_steps,
        'weights': weights,
        'itr': itr
    }
    residuals = calc_residuals(R, t, c, tgt, cad_kpts)
    normalized_residuals = residuals / noise_bound

    solution['residuals'] = residuals
    solution['normalized_residuals'] = normalized_residuals
    return solution