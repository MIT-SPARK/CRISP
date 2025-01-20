import copy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import sys
import pickle as pkl
import open3d as o3d

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'apolloscape', 'dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'apolloscape', 'dataset-api', 'utils'))
import apollo_utils as uts
from car_instance.car_models import *

from reg_utils import (project2d,
                       load_car_model,
                       load_disparity_to_depth,
                       fix_depth_map,
                       set_axes_equal,
                       load_kpt_lib)
import apollo_stereo_utils

from learning_objects.utils.visualization_utils import visualize_model_n_keypoints
import learning_objects.utils.general as gu


# USE_IGNORE_MASKS: Use Apollo3D ignore masks to skip keypoints that are too far away
USE_IGNORE_MASKS = True

USE_FG_MASKS = True

VISUALIZE = False

USE_RGB = False

USE_CLUSTER = True


STEREO_RGB_ROOT = "../../dataset/apollo_car_3d/stereo/stereo_train_merged/cameras/"
STEREO_RGB_MASKS_ROOT = "../../dataset/apollo_car_3d/stereo/stereo_masks/"
DISPARITY_MAP_ROOT = "../../dataset/apollo_car_3d/stereo/stereo_train_merged/disparity/"
FG_MASK_ROOT = "../../dataset/apollo_car_3d/stereo/stereo_fg_mask_merged/"
IGNORE_MASK_ROOT = "../../dataset/apollo_car_3d/3d-car-understanding-train/train/ignore_mask/"
PATH_TO_SAVED_POINTCLOUDS = "../../dataset/apollo_car_3d/pointclouds/new/"
PATH_TO_CAR_MODELS = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models_json/"
PATH_TO_O3D_CAR_MODELS = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models_o3d/"

# intrinsics (cropped camera 5) from jingnan
K_cropped = np.array([[2301.3147, 0, 1489.8536],
                      [0, 2301.3147, 479.1750],
                      [0, 0, 1]])
K_cropped_inv = np.linalg.inv(K_cropped)

# calculate stereo params
stereo_params = apollo_stereo_utils.get_stereo_rectify_params()


def apply_mask_to_depth(depth_map, car_mask):
    if depth_map.ndim == 3:
        depth_array = depth_map[:, :, 2]
        # np.set_printoptions(threshold=sys.maxsize)
        # print("depth array before", depth_array)
        # we need to multiply the original depth values by 1000 to get it in mm from meters
        depth_array = depth_array * 1000
    else:
        depth_array = depth_map
    car_mask = car_mask.astype(np.uint8) #converting mask of T/F into 0,1, using 8bit so can be applied as mask
    assert depth_array.shape == car_mask.shape, "car mask and depth map are not the same dimensions"
    res = cv2.bitwise_and(depth_array, depth_array, mask = car_mask)
    return res

def apply_mask_to_rgb(rgb_o3d, car_mask):
    rgb_array = np.asarray(rgb_o3d).astype(np.uint8)
    car_mask_array = np.asarray(car_mask)
    car_mask_array = np.where(car_mask_array > 0, 1, 0)
    car_mask_array = car_mask_array.astype(np.uint8) #converting mask of T/F into 0,1, using 8bit so can be applied as mask
    res = cv2.bitwise_and(rgb_array, rgb_array, mask = car_mask_array)
    return res

def get_mask_for_unrect_car_pose(im_name, car_pose, H_homo, K_orig_inv, K_cropped, K_orig):
    '''
    for a car pose in the unrectified image, load the car mask dictionary and idx
    of the matched car corresponding to the mask in the rectified rgb image
    from the stereo dataset if it exists, else return None
    :param car_id:
    :param car_pose: car_data['pose']
    :param im_name:
    :return:
    '''
    try:
        with open(STEREO_RGB_MASKS_ROOT + 'mask_data/' + im_name + '.pkl', 'rb') as handle:
            mask_data = pkl.load(handle)
    except EnvironmentError:
        print("WARNING: SKIPPING. mask for image {} doesn't exist".format(im_name))
        return None, None, None
    print(im_name)

    # load car pose and get the point in stereo image coordinates
    unrectC_T_W = uts.trans_vec_to_mat(car_pose[:3], car_pose[3:])
    unrect_pts_car = [[car_pose[0]], [car_pose[1]], [car_pose[2]], [1]]
    unrectC_carVerts = unrectC_T_W @ unrect_pts_car
    unrectI_car = project2d(K_orig, unrectC_carVerts)
    unrect_rays = K_orig_inv @ unrectI_car
    cropped_rays = H_homo @ unrect_rays

    ## to homogeneous coordinates
    cropped_rays /= cropped_rays[2, :]
    ## project rays to image
    cropped_pt_car = K_cropped @ cropped_rays

    car_masks = mask_data['car_masks']
    car_rois = mask_data['car_rois']
    dist_to_roi = dict()
    for i in range(len(car_rois)):
        y_min = car_rois[i][0]
        x_min = car_rois[i][1]
        y_max = car_rois[i][2]
        x_max = car_rois[i][3]
        mid_y = (y_min+y_max)/2
        mid_x = (x_min+x_max)/2
        y_dist = y_max - y_min
        x_dist = x_max - x_min
        y_min = mid_y - .3 * y_dist
        y_max = mid_y + .3 * y_dist
        x_min = mid_x - .3 * x_dist
        x_max = mid_x + .3 * x_dist

        diff_x = np.abs(mid_x - cropped_pt_car[0])
        diff_y = np.abs(mid_y - cropped_pt_car[1])

        # if cropped_pt_car lies within a percentage of the center of the roi (done so to account for overlapping cars),
        # rank roi by distance from center of roi
        # to estimated center of car, and return index of that roi/mask
        if (cropped_pt_car[0] > x_min and
            cropped_pt_car[0] < x_max and
            cropped_pt_car[1] > y_min and
            cropped_pt_car[1] < y_max):
            dist = np.sqrt(diff_x**2 + diff_y**2)[0]
            dist_to_roi[dist] = i
    if len(dist_to_roi) == 0:
        return None, None, None
    min_idx = dist_to_roi[min(dist_to_roi.keys())]
    car_mask = car_masks[:, :, min_idx]
    mask_pts = np.asarray(np.where(car_mask))#.T
    car_roi = car_rois[min_idx]

    if VISUALIZE:
        im_path = "../../dataset/apollo_car_3d/stereo/stereo_train_merged/cameras/" + im_name + '.jpg'
        im = Image.open(im_path)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(im)
        ax.scatter(cropped_pt_car[0, :], cropped_pt_car[1, :], s=500, alpha=0.5, color='green')
        ax.scatter(mask_pts[1], mask_pts[0], s=1, alpha=0.2, color='red')
        plt.show()
    return min_idx, car_mask, car_roi

def load_stereo_rgb_im(im_name):
    '''
    loads an open3d instance of an image
    :param im_name:
    :return:
    '''
    rgb_im_path = os.path.join(STEREO_RGB_ROOT, im_name + ".jpg")
    if not os.path.exists(rgb_im_path):
        print("WARNING STEREO RGB IMAGE DOES NOT EXIST", im_name)
        return None
    rgb = o3d.io.read_image(rgb_im_path)
    return rgb

def load_depth_im(im_name):
    disp_im_path = os.path.join(DISPARITY_MAP_ROOT, im_name + ".png")
    if not os.path.exists(disp_im_path):
        print("WARNING DISPARITY IMAGE DOES NOT EXIST", im_name)
        return None
    depth_map = load_disparity_to_depth(disp_im_path, stereo_params['Q'])
    fix_depth_map(depth_map)
    return depth_map

def in_ignore_mask(ignore_mask, keypoints, threshold=0.5):
    """Return true if the provided keypoints are the

    :param ignore_mask:
    :param keypoints:
    :return:
    """
    n = keypoints.shape[1]
    ignore_flags = np.zeros(n, )
    for i in range(n):
        cpt = keypoints[:2, i]
        # set flag to true if the color is not black (all zeros)
        # if the color is not black, that means the keypoint is in ignored regions
        ignore_flags[i] = ignore_mask[int(cpt[1]), int(cpt[0])] > 0

    if np.sum(ignore_flags) > threshold * n:
        return True
    else:
        return False


def get_stereo_depths(unrect_pts, depth_map, H_homo, K_orig_inv, K_cropped, unrect_car_gt, patch_size=0):
    '''
    Get depths for 2D points in the unrectified image

    :param unrect_pts:
    :param depth_map: depth map from disparity map provided
    :param H_homo: homography transformation from unrectified to rectified image?
    :param K_orig_inv: inverse intrinsic matrix for original camera
    :param K_cropped: intrinsic matrix for cropped camera
    :param unrect_car_gt:
    :param patch_size: size of the patch used to query depths, set to zero if not using patch depth query
    :return:
    '''
    # get rays in the stereo cropped frame
    unrect_rays = K_orig_inv @ unrect_pts
    cropped_rays = H_homo @ unrect_rays
    ## to homogeneous coordinates
    cropped_rays /= cropped_rays[2, :]

    ## project rays to image
    cropped_pts = K_cropped @ cropped_rays

    # This array stores the distances of each point w.r.t. optical center in the stereo / cropped frame
    cropped_dists = np.zeros((cropped_pts.shape[1],))
    if patch_size == 0 or patch_size is None:
        for idx in range(cropped_pts.shape[1]): #number of pts
            x = min(int(cropped_pts[0, idx]), depth_map.shape[1] - 1)
            y = min(int(cropped_pts[1, idx]), depth_map.shape[0] - 1)
            c_3d_pt = depth_map[y, x]
            cropped_dists[idx] = np.sqrt(c_3d_pt[0] ** 2 + c_3d_pt[1] ** 2 + c_3d_pt[2] ** 2)
    else: #this shouldn't ever be used...
        def dist_func(a):
            return np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

        for idx in range(cropped_pts.shape[1]):
            x = min(int(croppedI_car[0, idx]), depth_map.shape[1] - 1)
            y = min(int(croppedI_car[1, idx]), depth_map.shape[0] - 1)
            patch_x_left = max(y - patch_size, 0)
            patch_y_low = max(x - patch_size, 0)
            depth_patch = depth_map[patch_x_left:patch_x_left + patch_size, patch_y_low:patch_y_low + patch_size]
            dists_patch = np.apply_along_axis(dist_func, 2, depth_patch)
            croppedC_dists[idx] = np.amin(dists_patch)

    # Transform depth to original camera frame
    # We assume shared optical center between cropped and unrectified image, so distances from optical center to the points should stay constant
    unrect_rays_est = copy.deepcopy(unrect_rays)
    for idx in range(unrect_rays_est.shape[1]):
        temp = unrect_rays_est[:, idx] # one point
        unrect_rays_est[:, idx] = temp * cropped_dists[idx] / np.linalg.norm(temp)

    if VISUALIZE:
        # Compare with ground truth
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.scatter(unrect_car_gt[0, :], unrect_car_gt[1, :], unrect_car_gt[2, :], c='g')
        ax.scatter(unrect_rays_est[0, :], unrect_rays_est[1, :], unrect_rays_est[2, :], c='r')
        set_axes_equal(ax)
        plt.show()

    return unrect_rays_est


def save_models_as_o3d_mesh(dir_location_src=PATH_TO_CAR_MODELS, dir_location_dest=PATH_TO_O3D_CAR_MODELS, viz=True):
    model_names = car_name2id.keys()
    model_locations = ["{}/{}.json".format(dir_location_src, model_name) for model_name in model_names]
    o3d_model_locations = ["{}/{}.ply".format(dir_location_dest, model_name) for model_name in model_names]

    for loc in range(len(model_locations)):
        W_carVerts, car_faces = load_car_model(model_locations[loc])
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.transpose(W_carVerts[:3, :]))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(car_faces)
        o3d_mesh.compute_vertex_normals()  # normals on the side of the car are flipped, RIP.
        o3d_mesh.compute_triangle_normals()
        if viz:
            o3d_mesh.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries([o3d_mesh])
        o3d.io.write_triangle_mesh(o3d_model_locations[loc], o3d_mesh)



def generate_depth_data(model_name, radius_multiple = [1.2, 3.0],
                        num_of_points=100000, num_of_depth_images_per_radius=200,
                        dir_location=PATH_TO_O3D_CAR_MODELS):
    """ Generates depth point clouds of the CAD model """
    radius_multiple = np.asarray(radius_multiple)

    model_location = "{}/{}.ply".format(dir_location, model_name)
    # get model
    o3d_mesh = o3d.io.read_triangle_mesh(model_location)
    o3d_mesh.compute_triangle_normals()
    o3d_mesh.paint_uniform_color([1, 0.706, 0])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices))
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(100)
    # corrected_normals = np.asarray(pcd.normals)
    # print(corrected_normals)
    # o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(corrected_normals)
    # o3d.visualization.draw_geometries([o3d_mesh])
    #
    #
    # get keypoints_xyz (do i need this)
    cad_db, available_cads = load_kpt_lib(
        PATH_TO_CAR_MODELS + "../../keypoints_3d/0711_pp_checked")
    cad_db_dict = {x["name"]: x["kpts"] for x in cad_db}
    keypoints_xyz = np.transpose(cad_db_dict[model_name][:3, :])
    model_pcd = o3d_mesh.sample_points_uniformly(number_of_points=num_of_points)
    center = model_pcd.get_center()
    model_pcd.translate(-center)
    o3d_mesh.translate(-center)
    keypoints_xyz = keypoints_xyz - center
    model_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # determining size of the 3D object
    diameter = np.linalg.norm(np.asarray(model_pcd.get_max_bound()) - np.asarray(model_pcd.get_min_bound()))

    # determining camera locations and radius
    camera_distance_vector = diameter * radius_multiple
    camera_locations = gu.get_camera_locations(camera_distance_vector,
                                               number_of_locations=num_of_depth_images_per_radius)
    radius = gu.get_radius(object_diameter=diameter, cam_location=np.max(camera_distance_vector))

    # visualizing 3D object and all the camera locations
    _ = visualize_model_n_keypoints([model_pcd], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)
    _ = visualize_model_n_keypoints([o3d_mesh], keypoints_xyz=keypoints_xyz, camera_locations=camera_locations)

    #location to save data
    location = PATH_TO_CAR_MODELS + "../../../processed/" + str(model_name) + "/"

    # generating radius for view sampling
    gu.sample_depth_pcd(centered_pcd=model_pcd, camera_locations=camera_locations, radius=radius, folder_name=location)

    # save keypoints_xyz at location #todo[lisa]: fix path
    np.save(file=location + 'keypoints_xyz.npy', arr=keypoints_xyz)


if __name__ == "__main__":
    # dataset = load_apollo_data(split="train")
    # K_orig = dataset["K"]
    # K_mat = uts.intrinsic_vec_to_mat(K_orig)
    # invK = np.linalg.inv(K_mat)
    # image_data = []
    # # Iterate through all the images in the dataset
    # for sample_data in tqdm(dataset["data"]):
    #     # all_kpts = np.zeros((3,1))
    #     # all_unrectC_carVerts = np.zeros((4,1))
    #     cars_data = load_car_poses(sample_data["car_metadata"])
    #     depth_map = load_depth_im(sample_data['im_name'])
    #     if depth_map is None:
    #         print("WARNING: SKIPPING IMAGE BECAUSE DISPARITY IMAGE DOES NOT EXIST")
    #         continue
    #     if USE_FG_MASKS:
    #         fg_mask_path = os.path.join(FG_MASK_ROOT, sample_data['im_name'] + ".png")
    #         fg_mask = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
    #         if fg_mask is None:
    #             print("WARNING: SKIPPING CAR BECAUSE FOREGROUND MASK DOESN'T EXIST")
    #             continue
    #     # load all gt cars to load the keypoints
    #     unrectI_car_pointset = []
    #     for car_data in cars_data:
    #         # load car model
    #         car_name = car_id2name[car_data["car_id"]]
    #         W_carVerts, car_faces = load_car_model("{}/{}.json".format(dataset["car_models_dir"], car_name.name))
    #         W_carVerts = np.vstack((W_carVerts, np.ones((1, W_carVerts.shape[1]))))
    #
    #         # load car pose
    #         car_pose = np.array(car_data['pose'])
    #         unrectC_T_W = uts.trans_vec_to_mat(car_pose[:3], car_pose[3:])
    #
    #         # apply pose transformation
    #         unrectC_carVerts = unrectC_T_W @ W_carVerts
    #         unrectI_car = project2d(dataset["K"], unrectC_carVerts)
    #         unrectI_car_pointset.append((car_name, unrectI_car, unrectC_carVerts, unrectC_T_W))
    #     unrectI_kpts_db = load_2d_keypoints(sample_data["keypoints_dir"])
    #     car_name_2_2d_kpt = {}
    #     # get dictionary corresponding car to kpts in a frame
    #     temp_car_2d_pointset = copy.deepcopy(unrectI_car_pointset)
    #     for kpts_group in unrectI_kpts_db:
    #         if len(kpts_group) < 10:
    #             print("WARNING: SKIPPING CAR BECAUSE LESS THAN 10 KEYPOINTS")
    #             continue
    #         idx_of_car = get_corresponding_car_for_kpts(kpts_group, temp_car_2d_pointset)
    #         car_name_kpts = temp_car_2d_pointset[idx_of_car][0]
    #         car_name_2_2d_kpt[car_name_kpts.name] = kpts_group
    #     for car_data_count, car_data in enumerate(cars_data):
    #         # load car model
    #         car_name = car_id2name[car_data["car_id"]]
    #         if car_name.name not in car_name_2_2d_kpt:
    #             continue
    #
    #         # # load car model
    #         # W_carVerts, car_faces = load_car_model("{}/{}.json".format(dataset["car_models_dir"], car_name.name))
    #         #
    #         # # gt_mesh = trimesh.Trimesh(vertices=np.transpose(W_carVerts[:3, :]), faces=car_faces)
    #         # # W_carVerts are xyz homogeneous coordinates of the car
    #         # W_carVerts = np.vstack((W_carVerts, np.ones((1, W_carVerts.shape[1]))))
    #         ## load car pose
    #         car_pose = np.array(car_data['pose'])
    #         ##### thesholding not processing cars if they're too far away
    #         if car_pose[3:][2] > 35:
    #             print("WARNING SKIPPING CAR BECAUSE FURTHER THAN 35m FROM CAMERA")
    #             continue
    #         #####
    #         # unrectC_T_W = uts.trans_vec_to_mat(car_pose[:3], car_pose[3:])
    #         #
    #         # # apply pose transformation
    #         # unrectC_carVerts = unrectC_T_W @ W_carVerts
    #         # unrectI_car = project2d(dataset["K"], unrectC_carVerts)
    #         # keypoints
    #         pts = car_name_2_2d_kpt[car_name.name]
    #         if len(pts) < 10:
    #             print("WARNING: SKIPPING CAR BECAUSE LESS THAN 10 KEYPOINTS")
    #             continue
    #         pts_np = np.array([[x,y] for x,y in pts.values()])
    #         pts_np = pts_np.T
    #         pts_np = np.vstack((pts_np, np.ones(pts_np.shape[1])))
    #         if USE_IGNORE_MASKS:
    #             ignore_mask_path = os.path.join(IGNORE_MASK_ROOT, sample_data['im_name'] + ".jpg")
    #             ignore_mask = cv2.imread(ignore_mask_path, cv2.IMREAD_GRAYSCALE)
    #             to_ignore = in_ignore_mask(ignore_mask, pts_np)
    #             if to_ignore:
    #                 print("WARNING: SKIPPING CAR BECAUSE KEYPOINTS OF CAR IN THE IGNORE MASK")
    #                 continue
    #
    #         idx, car_mask, car_roi = get_mask_for_unrect_car_pose(sample_data['im_name'], car_pose, stereo_params['H_homo'], invK, K_cropped, K_orig)
    #         if idx is None or car_mask is None or car_roi is None:
    #             print("WARNING: SKIPPING CAR BECAUSE LACK OF GOOD CAR SEGMENTATION MASK")
    #             continue
    #         depth_mask = apply_mask_to_depth(depth_map, car_mask)
    #         if USE_FG_MASKS:
    #             depth_mask = apply_mask_to_depth(depth_mask, fg_mask)
    #         # depth_mask = depth_mask.astype(np.uint16)
    #         # depth_mask = depth_mask.astype(np.float16)
    #         # cv2.imwrite(DISPARITY_MAP_ROOT + '../temp_depth_map.png', depth_mask)
    #         # depth_o3d = o3d.io.read_image(DISPARITY_MAP_ROOT + '../temp_depth_map.png')
    #         depth_o3d = o3d.geometry.Image(depth_mask.astype(np.uint16))
    #
    #
    #         if USE_RGB:
    #             rgb_o3d = load_stereo_rgb_im(sample_data['im_name'])
    #             rgb_masked = apply_mask_to_rgb(rgb_o3d, depth_o3d)
    #             rgb_o3d = o3d.geometry.Image(np.asarray(rgb_masked))
    #             depth_array_meters = np.array(depth_o3d)*.001
    #             depth_o3d = o3d.geometry.Image(depth_array_meters.astype(np.float32))
    #             rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
    #
    #             if VISUALIZE:
    #                 fig, ax = plt.subplots(1, 2, figsize= (15, 15))
    #                 ax[0].imshow(rgbd.color)
    #                 ax[1].imshow(rgbd.depth)
    #             plt.show()
    #
    #             pcd = img_to_pcl(rgbd, K_cropped)
    #         else:
    #             pcd = depth_img_to_pcl(depth_o3d, K_cropped)
    #         if len(pcd.points) < 100:
    #             print("less than 100 points in this point cloud, skipping")
    #             continue
    #         if os.path.exists(PATH_TO_SAVED_POINTCLOUDS + sample_data['im_name'] + '_' + car_name.name + '_' + str(car_data_count) + '.pcd'):
    #             print("CONTINUING")
    #             continue
    #         save_pcl(PATH_TO_SAVED_POINTCLOUDS + sample_data['im_name'] + '_' + car_name.name + '_' + str(car_data_count) + '.pcd', pcd)
    #
    #         if USE_CLUSTER:
    #             #downsample pcd
    #             if len(pcd.points) > 70000: #maybe cap at 80000?
    #                 print("DOWNSAMPLING to do clustering")
    #                 pcd = pcd.voxel_down_sample(voxel_size=0.01)
    #                 print("pcd now has reduced number of points", len(pcd.points))
    #
    #             # cluster = np.array(pcd.cluster_dbscan(eps= .75, min_points = 800, print_progress=True))
    #             cluster = np.array(pcd.cluster_dbscan(eps=.75, min_points=600, print_progress=True))
    #             max_label = cluster.max()
    #             print("number of clusters", max_label + 1)
    #             if max_label + 1 == 0:
    #                 continue
    #
    #             counts = np.bincount(np.where(cluster > 0, cluster, 0))
    #             largest_cluster_label = np.argmax(counts)
    #             bool_indices_of_largest_cluster = (cluster == largest_cluster_label)
    #             largest_cluster_points = np.array(pcd.points)[bool_indices_of_largest_cluster]
    #
    #
    #             cmap = plt.get_cmap("tab20")
    #             colors = cmap(cluster / (max_label if max_label > 0 else 1))
    #             colors[cluster < 0] = 0
    #             pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #
    #             primary_cluster_pcd = o3d.geometry.PointCloud()
    #             primary_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    #
    #             if len(pcd.points) < 100:
    #                 print("less than 100 points in this point cloud, skipping")
    #                 continue
    #
    #             o3d.geometry.PointCloud.estimate_normals(primary_cluster_pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
    #                                                                                                             max_nn=30))
    #             o3d.geometry.PointCloud.orient_normals_towards_camera_location(primary_cluster_pcd, camera_location=np.array([0., 0., 0.]))
    #             primary_cluster_pcd.paint_uniform_color([.5, .5, .5])
    #             if VISUALIZE:
    #                 o3d.visualization.draw_geometries([pcd])
    #                 o3d.visualization.draw_geometries([primary_cluster_pcd])
    #             save_pcl(PATH_TO_SAVED_POINTCLOUDS + "/largest_cluster/" + sample_data['im_name'] + '_' + car_name.name + '_' + str(car_data_count) + '.pcd', primary_cluster_pcd)

    # generate_depth_data(generate_depth_data"biyadi-F3")
    save_models_as_o3d_mesh(viz=False)
