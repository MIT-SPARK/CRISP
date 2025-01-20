"""
pytorch utils util functions
"""

import pytorch3d.transforms as p3d
import torch


def _append_ones(points):
    """
    append ones to points so we can use it in 4-dim matrix math
    :param points: N points (x,y,z), shape Nx3
    :return: N points (x,y,z,1), shape Nx4
    """
    ones = torch.ones([points.shape[0], 1], dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], -1)


def camera_to_screen_projection(points, cam_intr):
    """
    project points from camera to screen space
    :param points: N points (x,y,z) in camera space, shape Nx3
    :param cam_intr: 4x4 camera intrinsics
    :return: N points (x,y,z) in screen space, shape Nx3
    """
    if not isinstance(cam_intr, torch.Tensor):
        cam_intr = torch.tensor(cam_intr, dtype=points.dtype, device=points.device)
    points = _append_ones(points)
    points = cam_intr.matmul(points.t()).t()[:, :3]
    points[:, :2] /= points[:, 2:]  # divide x and y corrds by z
    return points


def screen_to_camera_projection(points, cam_intr):
    """
    project points from screen to camera space
    :param points: N points (x,y,z) in screen space, shape Nx3
    :param cam_intr: 4x4 camera intrinsics
    :return: N points (x,y,z) in camera space, shape Nx3
    """
    if not isinstance(cam_intr, torch.Tensor):
        cam_intr = torch.tensor(cam_intr, dtype=points.dtype, device=points.device)
    points[:, :2] *= points[:, 2:]  # multiply x and y corrds by z
    points = _append_ones(points)
    cam_intr_inv = torch.inverse(cam_intr)
    points = cam_intr_inv.matmul(points.t()).t()[:, :3]
    return points


def camera_to_global_projection(points, cam_pose):
    """
    project points from camera to global space
    :param points: N points (x,y,z) in camera space, shape Nx3
    :param cam_pose: 4x4 camera pose
    :return: N points (x,y,z) in global (world) space, shape Nx3
    """
    if not isinstance(cam_pose, torch.Tensor):
        cam_pose = torch.tensor(cam_pose, dtype=points.dtype, device=points.device)
    points = _append_ones(points)
    return cam_pose.matmul(points.t()).t()[:, :3]


def global_to_camera_projection(points, cam_pose):
    """
    project points from global to camera space
    :param points: N points (x,y,z) in global (world) space, shape Nx3
    :param cam_pose: 4x4 camera pose
    :return: N points (x,y,z) in camera space, shape Nx3
    """
    if not isinstance(cam_pose, torch.Tensor):
        cam_pose = torch.tensor(cam_pose, dtype=points.dtype, device=points.device)
    points = _append_ones(points)
    cam_pose_inv = torch.inverse(cam_pose)
    return cam_pose_inv.matmul(points.t()).t()[:, :3]


def screen_to_global_projection(points, cam_intr, cam_pose):
    """
    project points from screen to global space
    :param points: N points (x,y,z) in screen space, shape Nx3
    :param cam_intr: 4x4 camera intrinsics
    :param cam_pose: 4x4 camera pose
    :return: N points (x,y,z) in global (world) space, shape Nx3
    """
    points = screen_to_camera_projection(points, cam_intr)
    return camera_to_global_projection(points, cam_pose)


def global_to_screen_projection(points, cam_intr, cam_pose):
    """
    project points from global to screen space
    :param points: N points (x,y,z) in global (world) space, shape Nx3
    :param cam_intr: 4x4 camera intrinsics
    :param cam_pose: 4x4 camera pose
    :return: N points (x,y,z) in screen space, shape Nx3
    """
    points = global_to_camera_projection(points, cam_pose)
    return camera_to_screen_projection(points, cam_intr)


def _extrinsic_to_intrinsic(convention: str):
    """convert a given euler angle convention from extrinsic to intrinsic"""
    for letter in convention:
        if letter in ("X", "Y", "Z"):
            raise ValueError("Extrinsic and intrinsic rotations cannot be mixed.")
    return convention[::-1].upper()


def _is_extrinsic(convention: str):
    """checks whether a given euler angle convention defines an extrinsic rotation"""
    for letter in convention:
        if letter in ("x", "y", "z"):
            return True
    return False


def rotation_matrix_to_euler_angles(rotation_matrix, convention="xyz"):
    """
    wrapper for pytorch3d.transforms.matrix_to_euler_angles()
    enables support for extrinsic rotations
    """
    extrinsic_rotation = _is_extrinsic(convention)
    if extrinsic_rotation:
        convention = _extrinsic_to_intrinsic(convention)
    euler_angles = p3d.matrix_to_euler_angles(rotation_matrix, convention)
    if extrinsic_rotation:
        euler_angles = euler_angles.flip(-1)
    return euler_angles


def euler_angles_to_rotation_matrix(euler_angles, convention="xyz"):
    """
    wrapper for pytorch3d.transforms.euler_angles_to_matrix()
    enables support for extrinsic rotations
    """
    extrinsic_rotation = _is_extrinsic(convention)
    if extrinsic_rotation:
        convention = _extrinsic_to_intrinsic(convention)
        euler_angles = euler_angles.flip(-1)
    return p3d.euler_angles_to_matrix(euler_angles, convention)


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    wrapper for kornia.rotation_matrix_to_angle_axis()
    """
    raise NotImplementedError

def angle_axis_to_rotation_matrix(angle_axis):
    """
    wrapper for kornia.angle_axis_to_rotation_matrix()
    """
    raise NotImplementedError


def euler_angles_to_angle_axis(euler_angles, convention="xyz"):
    """
    Transform euler angles to angle-axis via euler->rotmat->angle-axis
    """
    rotation_matrix = euler_angles_to_rotation_matrix(euler_angles, convention)
    return rotation_matrix_to_angle_axis(rotation_matrix)


def angle_axis_to_euler_angles(angle_axis, convention="xyz"):
    """
    Transform angle-axis to euler angles via angle-axis->rotmat->euler
    """
    rotation_matrix = angle_axis_to_rotation_matrix(angle_axis)
    return rotation_matrix_to_euler_angles(rotation_matrix, convention)


def pose_to_matrix(pose, rot_type="euler"):
    """
    Transform a given 6DoF pose of shape 1x6 to a transformation matrix of shape 4x4
    """
    matrix = torch.eye(4, dtype=pose.dtype, device=pose.device)
    if rot_type == "euler":
        matrix[:3, :3] = euler_angles_to_rotation_matrix(pose[:3], "xyz")
    elif rot_type == "rotvec":
        matrix[:3, :3] = angle_axis_to_rotation_matrix(pose[:3])
    else:
        raise ValueError(f"Unrecognized rot_type {rot_type}.")
    matrix[:3, 3] = pose[3:]
    return matrix


def matrix_to_pose(transformation_matrix):
    """
    Transform a given transformation matrix of shape 4x4 to a 6DoF pose of shape 1x6
    """
    rotation = rotation_matrix_to_euler_angles(transformation_matrix[:3, :3], "xyz")
    translation = transformation_matrix[:3, 3]
    return torch.cat([rotation, translation], -1)


def transform_points_by_matrix(points, matrix):
    """
    Transform a pointcloud of shape Nx3 according to a transformation matrix of shape 4x4
    """
    ones = torch.ones_like(points[:, :1])  # Nx1
    points = torch.cat((points, ones), dim=-1)  # Nx4
    points = matrix.matmul(points.t()).t()  # Nx4
    return points[:, 0:3]  # Nx3


def transform_points_by_pose(points, pose):
    """
    Transform a pointcloud of shape Nx3 according to a 6DoF pose of shape 1x6
    """
    matrix = pose_to_matrix(pose.view(-1))
    return transform_points_by_matrix(points.view(-1, 3), matrix)


def adjust_pose(pose, pose_change):
    """
    Adjust a 6DoF pose of shape 1x6 according to a 6D0F pose change of shape 1x6 by multiplying the respective matrices
    """
    pose_matrix = pose_to_matrix(pose)
    change_matrix = pose_to_matrix(pose_change)
    new_pose_matrix = torch.matmul(change_matrix, pose_matrix)
    return matrix_to_pose(new_pose_matrix)
