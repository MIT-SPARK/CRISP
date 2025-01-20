import sys, os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

PATH_TO_SCANNET_LABEL_IMAGE = '../../dataset/scan_net/scans/scene0000_00/label-filt/'
PATH_TO_SCANNET_DEPTH_IMG = '../../dataset/scan_net/scans/scene0000_00/exported_data/depth/'
PATH_TO_SCANNET_COLOR_IMG = '../../dataset/scan_net/scans/scene0000_00/exported_data/color/'
IMG_FILENAME = '25.png'
COLOR_IMG_FILENAME = '25.jpg'
SCANNET_OBJECT_ID = 63
PATH_TO_DEPTH_CAMERA_INTRINSIC = '../../dataset/scan_net/scans/scene0000_00/exported_data/intrinsic/intrinsic_depth.txt'
PATH_TO_CUSTOM_IMGS = '../../dataset/custom/'

VISUALIZE = True

def read_camera_intrinsic_extrinsic_params(path):
    matrix = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines[:-1]:
            row = line[:-1].split(" ")
            row = [float(x) for x in row]
            matrix.append(row[:-1])
    return np.array(matrix)

def cv2_depth_to_o3d_rgbd_img(cv2_depth_filepath):
    depth_o3d = o3d.io.read_image(cv2_depth_filepath)
    blank_o3d = o3d.io.read_image(PATH_TO_CUSTOM_IMGS + "blank_rgb_480.jpg")
    rgbd = o3d.create_rgbd_image_from_color_and_depth(blank_o3d, depth_o3d, convert_rgb_to_intensity=False)
    return rgbd

def depth_img_to_pcl(depth_o3d, camera_intrinsic, cv2_depth_filepath = None):
    # if cv2_depth_filepath is not None:
    #     assert os.path.exists(cv2_depth_filepath)
    #     depth_o3d = o3d.io.read_image(cv2_depth_filepath)

    print(depth_o3d)
    plt.subplot()
    plt.imshow(depth_o3d)
    plt.show()
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = camera_intrinsic
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, cam)
    print("pcd", pcd)
    if len(pcd.points) == 0:
        return pcd
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                                max_nn=30))
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0., 0., 0.]))
    pcd.paint_uniform_color([.5, .5, .5])
    if VISUALIZE:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def img_to_pcl(rgbd, camera_intrinsic, viz=False):
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = camera_intrinsic
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.geometry.PointCloud.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                                max_nn=30))
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_location=np.array([0., 0., 0.]))
    # pcd.paint_uniform_color([.5, .5, 1])
    print(pcd)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def save_pcl(path, pointcloud):
    if os.path.exists(path):
        print("saved pointcloud file already exists, NOT overwriting")
        return
    o3d.io.write_point_cloud(path, pointcloud)

def create_blank_img(filename, height, width):
    blank_image = np.zeros((height, width, 3), np.uint16)
    cv2.imwrite(PATH_TO_CUSTOM_IMGS + filename, blank_image)


if __name__ == "__main__":
    img = cv2.imread(PATH_TO_SCANNET_LABEL_IMAGE + IMG_FILENAME, cv2.IMREAD_UNCHANGED)
    depth_img = cv2.imread(PATH_TO_SCANNET_DEPTH_IMG + IMG_FILENAME, cv2.IMREAD_UNCHANGED)
    color_img = cv2.imread(PATH_TO_SCANNET_COLOR_IMG + COLOR_IMG_FILENAME, cv2.IMREAD_UNCHANGED)
    depth_intrinsic = read_camera_intrinsic_extrinsic_params(PATH_TO_DEPTH_CAMERA_INTRINSIC)
    rgbd = cv2_depth_to_o3d_rgbd_img(PATH_TO_SCANNET_DEPTH_IMG + IMG_FILENAME)
    # img_to_pcl(rgbd, depth_intrinsic)
    depth_img_to_pcl(PATH_TO_SCANNET_DEPTH_IMG + IMG_FILENAME, depth_intrinsic)


    # print("depth_img stats")
    # print(depth_img.shape)
    # print(depth_img.dtype)
    # print(depth_img.size)
    # depth_height, depth_width = depth_img.shape
    #
    #
    #
    # print("color_img stats")
    # print(color_img.shape)
    # print(color_img.dtype)
    # print(color_img.size)
    #
    # This doesn't work because downsampling only works when img's dimensions is a multiple of the original
    # downsampled_color_img = cv2.pyrDown(color_img, dstsize=(depth_width, depth_height))
    # cv2.imshow("color image", color_img)
    # cv2.waitKey(0)
    # cv2.imshow("downsampled_color_img image", downsampled_color_img)
    # cv2.waitKey(0)
    #
    # print("HI")
    # print("label img stats")
    # print(img.shape)
    # print(img.dtype)
    # print(img.size)
    #
    # lower = np.array(SCANNET_OBJECT_ID, dtype="uint16")
    # upper = np.array(SCANNET_OBJECT_ID, dtype="uint16")
    #
    # object_mask = cv2.inRange(img, lower, upper)
    # masked_depth_image = cv2.bitwise_and(color_img, color_img, mask=object_mask)
    # indices = np.where(img == SCANNET_OBJECT_ID)
    # indices = np.transpose(indices)
    # # # print(indices)
    # # # for index in indices:
    # # #     row = index[0]
    # # #     col = index[1]
    # # #     # print(img[row][col])
    # # #     assert img[row][col] == SCANNET_OBJECT_ID
    # #
    # ids_in_scan = set()
    # for row in range(img.shape[0]):
    #     for col in range(img.shape[1]):
    #         if img[row, col] != 0:
    #             ids_in_scan.add(img[row,col])
    # print(ids_in_scan)
    # cv2.imshow("test", masked_depth_image)
    # cv2.waitKey(0)


    # cv2.imshow("original image", color_img)
    # cv2.waitKey(0)
    #
    # cv2.destroyAllWindows()
    # print("THERE")