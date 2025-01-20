import pytest
import torch

from crisp.utils.math import project_SO3
from crisp.utils.evaluation_metrics import rotation_error, translation_error


@pytest.mark.parametrize("B,t_mag,s_mag, pc_size", [(1, 1, 1, 100), (20, 1, 1, 100), (100, 1, 1, 100)])
def test_umeyama_batched(B, t_mag, s_mag, pc_size):
    from crisp.models.registration import umeyama_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t
    s_est, R_est, t_est = umeyama_batched(source_points=pc_src, target_points=pc_tgt)

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,s_mag, pc_size, num_outliers", [(20, 1, 1, 100, 10), (100, 1, 1, 100, 20)])
def test_umeyama_ransac_batched(B, t_mag, s_mag, pc_size, num_outliers):
    from crisp.models.registration import umeyama_ransac_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t

    # add outliers
    pc_tgt[:, :, -num_outliers:] += 100

    s_est, R_est, t_est, est_inliers, status = umeyama_ransac_batched(
        source_points=pc_src, target_points=pc_tgt, inlier_thres=torch.ones((B, 1), device=device) * 0.1
    )

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,s_mag, pc_size, num_outliers", [(20, 1, 1, 100, 10), (100, 1, 1, 100, 20)])
def test_umeyama_ransac_batched_masked(B, t_mag, s_mag, pc_size, num_outliers):
    assert num_outliers > 0
    from crisp.models.registration import umeyama_ransac_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t

    # add outliers
    outlier_outlier = int(num_outliers / 2)
    pc_tgt[:, :, -outlier_outlier:] += 100

    # create a random mask
    masks = torch.ones((B, pc_size), device=device)
    mask_outlier = num_outliers - outlier_outlier
    masks[:, -outlier_outlier - mask_outlier : -outlier_outlier] = 0

    s_est, R_est, t_est, est_inliers, status = umeyama_ransac_batched(
        source_points=pc_src, target_points=pc_tgt, masks=masks, inlier_thres=torch.ones((B, 1), device=device) * 0.1
    )

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,s_mag, pc_size", [(1, 1, 1, 100), (20, 1, 1, 100), (100, 1, 1, 100)])
def test_weighted_umeyama_batched_uniform_weights(B, t_mag, s_mag, pc_size):
    from crisp.models.registration import weighted_umeyama_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t

    weights = torch.ones((B, pc_size), device=device)
    s_est, R_est, t_est = weighted_umeyama_batched(source_points=pc_src, target_points=pc_tgt, weights=weights)

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,s_mag, pc_size", [(1, 1, 1, 20), (20, 1, 1, 50), (100, 1, 1, 50)])
def test_weighted_umeyama_batched_arbitrary_weights(B, t_mag, s_mag, pc_size):
    from crisp.models.registration import weighted_umeyama_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t

    # set the last 10 points to be zero
    outliers = 10
    weights = torch.ones((B, pc_size), device=device)
    weights[:, -outliers:] = 0
    s_est, R_est, t_est = weighted_umeyama_batched(source_points=pc_src, target_points=pc_tgt, weights=weights)

    # now run it without the points, with weights = 1
    s_est_alt, R_est_alt, t_est_alt = weighted_umeyama_batched(
        source_points=pc_src[:, :, :-outliers], target_points=pc_tgt[:, :, :-outliers], weights=weights[:, :-outliers]
    )

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R_est_alt[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t_est_alt[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s_est_alt[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,pc_size", [(1, 1, 100), (20, 1, 100), (100, 1, 100)])
def test_weighted_arun_batched_uniform_weights(B, t_mag, pc_size):
    from crisp.models.registration import weighted_arun_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = R @ pc_src + t

    weights = torch.ones((B, pc_size), device=device)
    R_est, t_est = weighted_arun_batched(source_points=pc_src, target_points=pc_tgt, weights=weights)

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
    avg_rot_err /= B
    avg_t_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05


@pytest.mark.parametrize("B,t_mag,pc_size", [(1, 1, 20), (20, 1, 50), (100, 1, 50)])
def test_weighted_arun_batched_arbitrary_weights(B, t_mag, pc_size):
    from crisp.models.registration import weighted_arun_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = R @ pc_src + t

    # set the last 10 points to be zero
    outliers = 10
    weights = torch.ones((B, pc_size), device=device)
    weights[:, -outliers:] = 0
    R_est, t_est = weighted_arun_batched(source_points=pc_src, target_points=pc_tgt, weights=weights)

    # now run it without the points, with weights = 1
    R_est_alt, t_est_alt = weighted_arun_batched(
        source_points=pc_src[:, :, :-outliers], target_points=pc_tgt[:, :, :-outliers], weights=weights[:, :-outliers]
    )

    avg_rot_err, avg_t_err = 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R_est_alt[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t_est_alt[bid, ...])
    avg_rot_err /= B
    avg_t_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05


@pytest.mark.parametrize("B,t_mag,s_mag,pc_size,num_outliers", [(20, 1, 1, 100, 10), (100, 1, 1, 100, 20)])
def test_teaser_with_scale_batched(B, t_mag, s_mag, pc_size, num_outliers):
    from crisp.models.registration import teaser_with_scale_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    s = torch.rand((B, 1, 1), device=device) * s_mag
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = s * (R @ pc_src) + t
    masks = torch.ones((B, pc_size), device=device).to(dtype=torch.bool)

    # add outliers
    pc_tgt[:, :, -num_outliers:] += 100

    s_est, R_est, t_est, _ = teaser_with_scale_batched(
        source_points=pc_src, target_points=pc_tgt, noise_bounds=torch.ones((B, 1), device=device) * 0.1, masks=masks
    )

    avg_rot_err, avg_t_err, avg_s_err = 0, 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
        avg_s_err += torch.norm(s_est[bid] - s[bid])
    avg_rot_err /= B
    avg_t_err /= B
    avg_s_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05 and avg_s_err < 0.05


@pytest.mark.parametrize("B,t_mag,pc_size,num_outliers", [(20, 1, 100, 10), (100, 1, 100, 20)])
def test_teaser_batched(B, t_mag, pc_size, num_outliers):
    from crisp.models.registration import teaser_batched

    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.rand((B, 3, 3), device=device)
    R, _, _, _, _ = project_SO3(A)
    t = torch.rand((B, 3, 1), device=device) * t_mag
    pc_src = torch.rand((B, 3, pc_size), device=device) * 10
    pc_tgt = R @ pc_src + t
    masks = torch.ones((B, pc_size), device=device).to(dtype=torch.bool)

    # add outliers
    pc_tgt[:, :, -num_outliers:] += 100

    R_est, t_est, _ = teaser_batched(
        source_points=pc_src, target_points=pc_tgt, noise_bounds=torch.ones((B, 1), device=device) * 0.1, masks=masks
    )

    avg_rot_err, avg_t_err = 0, 0
    for bid in range(B):
        avg_rot_err += rotation_error(R_est[bid, ...], R[bid, ...])
        avg_t_err += translation_error(t_est[bid, ...], t[bid, ...])
    avg_rot_err /= B
    avg_t_err /= B

    assert avg_rot_err < 0.05 and avg_t_err < 0.05
