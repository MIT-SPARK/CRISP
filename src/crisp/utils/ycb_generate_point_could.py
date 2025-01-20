#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import h5py as h5
from PIL import Image
import IPython
import math
import open3d as o3d
import matplotlib.pyplot as plt
from ycb_processing import get_largest_cluster, load_mesh, get_closest_cluster, save_rgbFromObj




sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'ycb-tools'))
# Define folders
default_ycb_folder = os.path.join("models", "ycb")
print("default_ycb_folder", default_ycb_folder)

from reg_utils import (render_cad,
                       generate_weighted_model,
                       project2d,
                       load_car_poses,
                       dists_from_ray,
                       load_car_model,
                       load_disparity_to_depth,
                       fix_depth_map,
                       set_axes_equal,
                       load_kpt_lib,
                       load_gsnet_keypoints,
                       select_cad_dist_bounds,
                       load_apollo_data,
                       SE3_transform,
                       load_2d_keypoints)
import apollo_stereo_utils

from get_3d_keypoints import get_corresponding_car_for_kpts
from depth_to_pcl_processing import depth_img_to_pcl, save_pcl, img_to_pcl
import learning_objects.utils.general as gu

# Parameters
ycb_data_folder = "../../third_party/ycb-tools/models/ycb/"			# Folder that contains the ycb data.
target_object = "052_extra_large_clamp" #"021_bleach_cleanser"	# Full name of the target object. ["021_bleach_cleanser", "052_extra_large_clamp", "006_mustard_bottle"]:

# viewpoint_camera = "NP1"				# Camera which the viewpoint will be generated.
# viewpoint_angle = "81"					# Relative angle of the object w.r.t the camera (angle of the turntable).


def im2col(im, psize):
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(im_pad.shape[1] // psize, psize,
                                     im_pad.shape[2] // psize, psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])

def filterDiscontinuities(depthMap):
    filt_size = 7
    thresh = 1000

    # Ensure that filter sizes are okay
    assert filt_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = (filt_size - 1) // 2
    patches = 1.0 * im2col(depthMap, filt_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depthMap * (1 - final_mark)

def registerDepthMap(unregisteredDepthMap,
                     rgbImage,
                     depthK,
                     rgbK,
                     H_RGBFromDepth):

    if unregisteredDepthMap.shape[-1] == 3:
        unregisteredDepthMap = unregisteredDepthMap[:,:,0]

    unregisteredHeight = unregisteredDepthMap.shape[0]
    unregisteredWidth = unregisteredDepthMap.shape[1]

    registeredHeight = rgbImage.shape[0]
    registeredWidth = rgbImage.shape[1]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))

    xyzDepth = np.empty((4,1))
    xyzRGB = np.empty((4,1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / depthK[0,0]
    invDepthFy = 1.0 / depthK[1,1]
    depthCx = depthK[0,2]
    depthCy = depthK[1,2]

    rgbFx = rgbK[0,0]
    rgbFy = rgbK[1,1]
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
      for u in range(unregisteredWidth):
            depth = unregisteredDepthMap[v,u]
            if depth == 0:
                continue

            xyzDepth[0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[2] = depth


            xyzRGB[0] = (H_RGBFromDepth[0,0] * xyzDepth[0] +
                         H_RGBFromDepth[0,1] * xyzDepth[1] +
                         H_RGBFromDepth[0,2] * xyzDepth[2] +
                         H_RGBFromDepth[0,3])
            xyzRGB[1] = (H_RGBFromDepth[1,0] * xyzDepth[0] +
                         H_RGBFromDepth[1,1] * xyzDepth[1] +
                         H_RGBFromDepth[1,2] * xyzDepth[2] +
                         H_RGBFromDepth[1,3])
            xyzRGB[2] = (H_RGBFromDepth[2,0] * xyzDepth[0] +
                         H_RGBFromDepth[2,1] * xyzDepth[1] +
                         H_RGBFromDepth[2,2] * xyzDepth[2] +
                         H_RGBFromDepth[2,3])

            invRGB_Z  = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > registeredDepthMap[vRGB,uRGB]:
                registeredDepthMap[vRGB,uRGB] = registeredDepth

    return registeredDepthMap


def registeredDepthMapToPointCloud(depthMap, rgbImage, rgbK,  refFromRGB, objFromref, organized=False):
    rgbCx = rgbK[0,2]
    rgbCy = rgbK[1,2]
    invRGBFx = 1.0/rgbK[0,0]
    invRGBFy = 1.0/rgbK[1,1]

    height = depthMap.shape[0]
    width = depthMap.shape[1]

    if organized:
      cloud = np.zeros((height, width, 6), dtype=np.float64)
    else:
      cloud = np.zeros((1, height*width, 6), dtype=np.float64)

    goodPointsCount = 0
    for v in range(height):
        for u in range(width):

            depth = depthMap[v,u]
            if type(depth) != np.float64:
                depth = depth[0]


            if organized:
              row = v
              col = u
            else:
              row = 0
              col = goodPointsCount

            if depth <= 0:
                if organized:
                    if depth <= 0:
                       cloud[row,col,0] = float('nan')
                       cloud[row,col,1] = float('nan')
                       cloud[row,col,2] = float('nan')
                       cloud[row,col,3] = 0
                       cloud[row,col,4] = 0
                       cloud[row,col,5] = 0
                continue
            #ORIGINAL
            cloud[row,col,0] = (u - rgbCx) * depth * invRGBFx
            cloud[row,col,1] = (v - rgbCy) * depth * invRGBFy
            cloud[row,col,2] = depth
            cloud[row,col,3] = rgbImage[v,u,0]
            cloud[row,col,4] = rgbImage[v,u,1]
            cloud[row,col,5] = rgbImage[v,u,2]
            #IF you desire the pointcloud TRANFORMED to obj frame of reference
            # x = (u - rgbCx) * depth * invRGBFx
            # y = (v - rgbCy) * depth * invRGBFy
            # z = depth
            #
            # # refFromRGB
            # x1 = (refFromRGB[0, 0] * x +
            #       refFromRGB[0, 1] * y +
            #       refFromRGB[0, 2] * z +
            #       refFromRGB[0, 3])
            # y1 = (refFromRGB[1, 0] * x +
            #       refFromRGB[1, 1] * y +
            #       refFromRGB[1, 2] * z +
            #       refFromRGB[1, 3])
            # z1 = (refFromRGB[2, 0] * x +
            #       refFromRGB[2, 1] * y +
            #       refFromRGB[2, 2] * z +
            #       refFromRGB[2, 3])
            #
            # x, y, z = x1, y1, z1
            #
            # # obj from ref
            # cloud[row, col, 0] = (objFromref[0, 0] * x +
            #                       objFromref[0, 1] * y +
            #                       objFromref[0, 2] * z +
            #                       objFromref[0, 3])
            # cloud[row, col, 1] = (objFromref[1, 0] * x +
            #                       objFromref[1, 1] * y +
            #                       objFromref[1, 2] * z +
            #                       objFromref[1, 3])
            # cloud[row, col, 2] = (objFromref[2, 0] * x +
            #                       objFromref[2, 1] * y +
            #                       objFromref[2, 2] * z +
            #                       objFromref[2, 3])
            #
            # cloud[row, col, 3] = rgbImage[v, u, 0]
            # cloud[row, col, 4] = rgbImage[v, u, 1]
            # cloud[row, col, 5] = rgbImage[v, u, 2]
            if not organized:
              goodPointsCount += 1

    if not organized:
      cloud = cloud[:,:goodPointsCount,:]
    return cloud


def writePLY(filename, cloud, faces=[], save_numpy=False, np_filename=""):
    if len(cloud.shape) != 3:
        print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(cloud.shape))
        return
    if save_numpy:
        np.save(np_filename, cloud)
    color = True if cloud.shape[2] == 6 else False
    num_points = cloud.shape[0]*cloud.shape[1]

    header_lines = [
        'ply',
        'format ascii 1.0',
        'element vertex %d' % num_points,
        'property float x',
        'property float y',
        'property float z',
        ]
    if color:
        header_lines.extend([
        'property uchar diffuse_red',
        'property uchar diffuse_green',
        'property uchar diffuse_blue',
        ])
    if faces != None:
        header_lines.extend([
        'element face %d' % len(faces),
        'property list uchar int vertex_indices'
        ])

    header_lines.extend([
      'end_header',
      ])

    f = open(filename, 'w+')
    f.write('\n'.join(header_lines))
    f.write('\n')

    lines = []
    for i in range(cloud.shape[0]):
        for j in range(cloud.shape[1]):
            if color:
                lines.append('%s %s %s %d %d %d' % tuple(cloud[i, j, :].tolist()))
            else:
                lines.append('%s %s %s' % tuple(cloud[i, j, :].tolist()))

    for face in faces:
        lines.append(('%d' + ' %d'*len(face)) % tuple([len(face)] + list(face)))

    f.write('\n'.join(lines) + '\n')
    f.close()

def writePCD(pointCloud, filename, ascii=False):
    if len(pointCloud.shape) != 3:
      print("Expected pointCloud to have 3 dimensions. Got %d instead" % len(pointCloud.shape))
      return
    with open(filename, 'w') as f:
        height = pointCloud.shape[0]
        width = pointCloud.shape[1]
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        if pointCloud.shape[2] == 3:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        else:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F F\n")
            f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH %d\n" % width)
        f.write("HEIGHT %d\n" % height)
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS %d\n" % (height * width))
        if ascii:
          f.write("DATA ascii\n")
          for row in range(height):
            for col in range(width):
                if pointCloud.shape[2] == 3:
                    f.write("%f %f %f\n" % tuple(pointCloud[row, col, :]))
                else:
                    f.write("%f %f %f" % tuple(pointCloud[row, col, :3]))
                    r = int(pointCloud[row, col, 3])
                    g = int(pointCloud[row, col, 4])
                    b = int(pointCloud[row, col, 5])
                    rgb_int = (r << 16) | (g << 8) | b
                    packed = pack('i', rgb_int)
                    rgb = unpack('f', packed)[0]
                    f.write(" %.12e\n" % rgb)
        else:
          f.write("DATA binary\n")
          if pointCloud.shape[2] == 6:
              # These are written as bgr because rgb is interpreted as a single
              # little-endian float.
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('b', np.uint8),
                             ('g', np.uint8),
                             ('r', np.uint8),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z', 'r', 'g', 'b']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)
          else:
              dt = np.dtype([('x', np.float32),
                             ('y', np.float32),
                             ('z', np.float32),
                             ('I', np.uint8)])
              pointCloud_tmp = np.zeros((height*width, 1), dtype=dt)
              for i, k in enumerate(['x', 'y', 'z']):
                  pointCloud_tmp[k] = pointCloud[:, :, i].reshape((height*width, 1))
              pointCloud_tmp.tofile(f)

def getRGBFromDepthTransform(calibration, camera, referenceCamera):
    irKey = "H_{0}_ir_from_{1}".format(camera, referenceCamera)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)

    rgbFromRef = calibration[rgbKey][:]
    irFromRef = calibration[irKey][:]

    return np.dot(rgbFromRef, np.linalg.inv(irFromRef)), np.linalg.inv(rgbFromRef)


def save_rgb_pcd(target_object, viewpoint_camera, viewpoint_angle, save_full_scene=False):
    referenceCamera="NP5"
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    depthFilename = os.path.join(ycb_data_folder + target_object, basename + ".h5")
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    calibrationFilename = os.path.join(ycb_data_folder + target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)
    objFromRefFilename = os.path.join(ycb_data_folder + target_object, 'poses',
                                      '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))

    objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]

    mask_filename = os.path.join(ycb_data_folder + target_object, "masks", basename + "_mask.pbm")
    mask = cv2.imread(mask_filename)
    kernel = np.ones((10, 10), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    temp_mask = np.asarray(erosion)
    obj_mask = np.zeros((temp_mask.shape[0], temp_mask.shape[1])).astype(np.uint8)
    for row in range(temp_mask.shape[0]):
        for col in range(temp_mask.shape[1]):
            if save_full_scene:
                obj_mask[row,col] = 1
            else:
                if temp_mask[row, col, 0] == 0:  # we want to keep black points
                    obj_mask[row, col] = 1
                else:
                    obj_mask[row, col] = 0

    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    with Image.open(rgbFilename) as image:
        rgbImage = np.array(image)
        depthK = calibration["{0}_depth_K".format(viewpoint_camera)][:]

        rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]

        depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001  # 100um to meters
        H_RGBFromDepth, refFromRGB = getRGBFromDepthTransform(calibration, viewpoint_camera, referenceCamera)

        unregisteredDepthMap = h5.File(depthFilename)["depth"][:]

        unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale

        registeredDepthMap = registerDepthMap(unregisteredDepthMap,
                                              rgbImage,
                                              depthK,
                                              rgbK,
                                              H_RGBFromDepth)
        masked_depth = cv2.bitwise_and(registeredDepthMap, registeredDepthMap, mask=obj_mask)
        pointCloud = registeredDepthMapToPointCloud(masked_depth, rgbImage, rgbK, refFromRGB, objFromRef,
                                                    organized=False)
        np_filename = ycb_data_folder + target_object + \
                                  "/clouds/rgb/pc_" + viewpoint_camera + "_" \
                                  + referenceCamera + "_" + viewpoint_angle \
                                  + "_masked_rgb.npy"
        writePLY(
            ycb_data_folder + target_object + "/clouds/rgb/pc_" + viewpoint_camera + "_" + referenceCamera + "_" + viewpoint_angle + "_masked_rgb.ply",
            pointCloud, save_numpy=True, np_filename=np_filename)


if __name__ == "__main__":
    # save_rgb_pcd("002_master_chef_can", "NP2", "207")

    # save_rgb_pcd("011_banana", "NP2", "174")

    # save_rgb_pcd("037_scissors", "NP3", "183")

    # save_rgb_pcd("052_extra_large_clamp", "NP1", "201")
    save_rgb_pcd("003_cracker_box", "NP2", "63", save_full_scene=True)


    #smoke test done
    #file existance test
    # for target_object in ["001_chips_can", "002_master_chef_can", "003_cracker_box", \
    #                       "004_sugar_box", "005_tomato_soup_can", \
    #                       "006_mustard_bottle", "007_tuna_fish_can", \
    #                       "008_pudding_box", "009_gelatin_box", \
    #                       "010_potted_meat_can", "011_banana", \
    #                       "019_pitcher_base", "021_bleach_cleanser", \
    #                       "035_power_drill", "036_wood_block", "037_scissors", \
    #                       "040_large_marker", "051_large_clamp", "061_foam_brick"]:
    #     model_mesh = load_mesh(target_object, viz=False)
    # # for target_object in ["001_chips_can", "002_master_chef_can", "003_cracker_box", \
    # #                       "004_sugar_box", "005_tomato_soup_can", \
    # #                       "006_mustard_bottle", "007_tuna_fish_can", \
    # #                       "008_pudding_box", "009_gelatin_box", \
    # #                       "010_potted_meat_can", "011_banana", \
    # #                       "019_pitcher_base", "021_bleach_cleanser", \
    # #                       "035_power_drill", "036_wood_block", "037_scissors", \
    # #                       "040_large_marker", "051_large_clamp", "061_foam_brick"]:
    # for target_object in ["021_bleach_cleanser"]:
    #     print("target_object", target_object)
    #     model_mesh = load_mesh(target_object, viz=False)
    #
    #     referenceCamera = "NP5"
    #     for viewpoint_camera in [ "NP1", "NP2", "NP3", "NP4", "NP5"]:
    #         for viewpoint_angle in range(358):
    #             # segmentation masks only exist for every third degree, so we sip the rest
    #             if viewpoint_angle%3 != 0:
    #                 continue
    #             viewpoint_angle = str(viewpoint_angle)
    #             print("processing viewpoint_camera", viewpoint_camera)
    #             print("viewpoint_angle", viewpoint_angle)
    #
    #             if not os.path.exists(ycb_data_folder+"/"+target_object+"/clouds"):
    #                 os.makedirs(ycb_data_folder+"/"+target_object+"/clouds")
    #
    #             basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    #             depthFilename = os.path.join(ycb_data_folder+ target_object, basename + ".h5")
    #             rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    #             print("depthFilename", depthFilename)
    #
    #             calibrationFilename = os.path.join(ycb_data_folder+target_object, "calibration.h5")
    #             calibration = h5.File(calibrationFilename)
    #             objFromRefFilename = os.path.join(ycb_data_folder+target_object, 'poses', '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))
    #
    #             objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]
    #
    #             mask_filename = os.path.join(ycb_data_folder + target_object, "masks", basename + "_mask.pbm")
    #             mask = cv2.imread(mask_filename)
    #             kernel = np.ones((10,10), np.uint8)
    #             erosion = cv2.erode(mask, kernel, iterations=1)
    #             temp_mask = np.asarray(erosion)
    #             obj_mask = np.zeros((temp_mask.shape[0], temp_mask.shape[1])).astype(np.uint8)
    #             for row in range(temp_mask.shape[0]):
    #                 for col in range(temp_mask.shape[1]):
    #                     if temp_mask[row,col,0] == 0: #we want to keep black points
    #                         obj_mask[row,col] = 1
    #                     else:
    #                         obj_mask[row,col] = 0
    #
    #             if not os.path.isfile(rgbFilename):
    #                 print("The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
    #                 exit(1)
    #             with Image.open(rgbFilename) as image:
    #                     rgbImage = np.array(image)
    #                     depthK = calibration["{0}_depth_K".format(viewpoint_camera)][:]
    #
    #                     rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
    #
    #                     depthScale = np.array(calibration["{0}_ir_depth_scale".format(viewpoint_camera)]) * .0001 # 100um to meters
    #                     H_RGBFromDepth, refFromRGB = getRGBFromDepthTransform(calibration, viewpoint_camera, referenceCamera)
    #
    #                     unregisteredDepthMap = h5.File(depthFilename)["depth"][:]
    #
    #                     unregisteredDepthMap = filterDiscontinuities(unregisteredDepthMap) * depthScale
    #
    #                     registeredDepthMap = registerDepthMap(unregisteredDepthMap,
    #                                                           rgbImage,
    #                                                           depthK,
    #                                                           rgbK,
    #                                                           H_RGBFromDepth)
    #                     #mask both rgbImage and depthMap with mask
    #
    #                     #
    #                     # print("registeredDepthMap.shape", registeredDepthMap.shape)
    #                     # print("unregisteredDepthMap.shape", unregisteredDepthMap.shape)
    #                     # #
    #                     # # plt.subplot()
    #                     # # registered_depth_o3d = o3d.geometry.Image(registeredDepthMap.astype(np.float32))
    #                     # # plt.imshow(registered_depth_o3d)
    #                     # # plt.show()
    #                     masked_depth = cv2.bitwise_and(registeredDepthMap, registeredDepthMap, mask=obj_mask)
    #                     # depth_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
    #                     #
    #                     # segmented_pcd = depth_img_to_pcl(depth_o3d, rgbK)
    #                     # pointCloud = registeredDepthMapToPointCloud(registeredDepthMap, rgbImage, rgbK)
    #                     # saving point cloud wrt the RGB camera pose
    #                     pointCloud = registeredDepthMapToPointCloud(masked_depth, rgbImage, rgbK, refFromRGB, objFromRef, organized=False)
    #                     writePLY(ycb_data_folder+target_object+"/clouds/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+"_masked.ply", pointCloud)
    #                     print("pointCloud.shape", pointCloud.shape)
    #                     print("Testing IO for point cloud ...")
    #             pcd = o3d.io.read_point_cloud(ycb_data_folder+target_object+"/clouds/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+viewpoint_angle+"_masked.ply")
    #             # if clustering with dbscan, uncomment the below
    #             # diameter = np.linalg.norm(
    #             #     np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))
    #             # eps = np.min(diameter)
    #             # print("eps", eps)
    #             # largest_cluster_pcd = get_largest_cluster(pcd, viz=False, eps=eps)
    #
    #             largest_cluster_pcd = pcd #get_largest_cluster(pcd, viz=False, eps=eps)
    #
    #             # further clustering with euclidian distance based thresholding
    #             if largest_cluster_pcd is not None:
    #                 rgbFromObj = save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle, only_return=False) #save gt poses!
    #                 R_true = rgbFromObj[:3, :3]
    #                 t_true = np.expand_dims(rgbFromObj[:3, 3], axis=1)
    #
    #                 model_pcd = model_mesh.sample_points_uniformly(number_of_points=3000)
    #                 model_pcd.points = o3d.utility.Vector3dVector(
    #                     (R_true @ np.asarray(model_pcd.points).transpose() + t_true).transpose())
    #                 largest_cluster_pcd = get_closest_cluster(largest_cluster_pcd, model_pcd, viz=False)
    #                 if largest_cluster_pcd is not None:
    #                     if not os.path.exists(ycb_data_folder + "/" + target_object + "/clouds/largest_cluster"):
    #                         os.makedirs(ycb_data_folder + "/" + target_object + "/clouds/largest_cluster")
    #                     o3d.io.write_point_cloud(ycb_data_folder + target_object + "/clouds/largest_cluster/pc_" + viewpoint_camera + "_" + referenceCamera + "_" + viewpoint_angle + "_masked.ply", largest_cluster_pcd)
