"""functions for saving pointclouds in various formats"""

import cv2
import imageio
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

from .depthmap_utils import depthmap_matrix_to_pointcloud


def save_depthmap_png(output_path, depthmap, multiplier=1e3):
    """save a given numpy depthmap"""
    depthmap = (depthmap * multiplier).astype(np.uint16)
    cv2.imwrite(output_path, depthmap)


def load_depthmap_png(depthmap_path, multiplier=1e3):
    """load a depthmap as numpy array"""
    depthmap = imageio.imread(depthmap_path).astype(np.float32) / multiplier
    return depthmap[..., np.newaxis]  # reshape to HxWx1


def save_img_png(output_path, img):
    """save a given numpy image"""
    cv2.imwrite(output_path, img)


def load_img_png(img_path):
    """load an image as numpy array"""
    img = imageio.imread(img_path).astype(np.float32)
    return img


def save_txt(output_path, matrix):
    """save a given numpy array as .txt file using np.savetxt"""
    np.savetxt(output_path, matrix)


def load_txt(txt_path):
    """load a .txt file as numpy array using np.loadtxt"""
    matrix = np.loadtxt(txt_path).astype(np.float32)
    return matrix


def save_video(filename, frames):
    """save a given list of frames as video in .mp4 format"""
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"MP4V"), 1, (width, height)
    )
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def save_pointcloud_ply(pointcloud, filename):
    """save a pointcloud as .ply using pyntcloud"""
    PyntCloud(
        pd.DataFrame(
            data={
                "x": pointcloud[:, 0],
                "y": pointcloud[:, 1],
                "z": pointcloud[:, 2],
            }
        )
    ).to_file(filename)


def save_pointcloud_xyz(pointcloud, filename, normals=None):
    """save a pointcloud (with normals) as .xyz"""
    pointcloud, normals = depthmap_matrix_to_pointcloud(pointcloud, normals)
    with open(filename, "w") as file_:
        for i, [x, y, z] in enumerate(pointcloud):
            file_.write("%.6f %.6f %.6f" % (x, y, z))
            if normals is not None:
                nx, ny, nz = normals[i]
                file_.write(" %.6f %.6f %.6f" % (nx, ny, nz))
            file_.write("\n")


def save_pointcloud_correspondences_obj(
    pointcloud1, pointcloud2, source_indices, target_indices, filename, eps=0.01
):
    """
    save the correspondences between two pointclouds as .obj
    useful for visualization in meshlab
    """
    with open(filename, "w") as file_:
        # save points of cloud 1 twice (slightly shifted) as vertices
        for i, [x, y, z] in enumerate(pointcloud1):
            file_.write("v %.6f %.6f %.6f\n" % (x - eps, y - eps, z - eps))
            file_.write("v %.6f %.6f %.6f\n" % (x + eps, y + eps, z + eps))
        # save second pointcloud as vertices
        for i, [x, y, z] in enumerate(pointcloud2):
            file_.write("v %.6f %.6f %.6f\n" % (x, y, z))
        # save correspondences as faces
        for i, s_ind in enumerate(source_indices):
            t_ind = target_indices[i]
            pt1 = s_ind * 2 + 1
            pt2 = len(pointcloud1) * 2 + t_ind + 1
            file_.write("f %d %d %d\n" % (pt1, pt1 + 1, pt2))
