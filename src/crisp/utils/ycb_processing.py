import copy
import cv2
import datetime
import json
import matplotlib.pyplot as plt
from PIL import Image
import h5py as h5
import numpy as np
import os
import pickle
import sys
import time
import trimesh
from tqdm import tqdm
from scipy.spatial import ConvexHull
import scipy.io
import pickle as pkl
import open3d as o3d
import random
import torch
from pytorch3d import ops


sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'ycb-tools'))
# Define folders
default_ycb_folder = os.path.join("models", "ycb")
referenceCamera = "NP5"

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

from get_3d_keypoints import project2d, plot_pts_on_image
from depth_to_pcl_processing import depth_img_to_pcl, save_pcl, img_to_pcl
import learning_objects.utils.general as gu
from learning_objects.datasets.ycb import DepthYCB

# ycb_data_folder = "../../third_party/ycb-tools/models/ycb/"  # Folder that contains the ycb data.
ycb_data_folder = "../../../../../../../media/lisa/Figgy/datasets_rajat/ycb/models/ycb/"
ycb_video_data_folder = "../../../../../../../media/lisa/Figgy/detectron2_datasets/YCB_Video_Dataset/"

YCB_VIDEO_ID_TO_MODEL = ["002_master_chef_can", "003_cracker_box", \
                          "004_sugar_box", "005_tomato_soup_can", \
                          "006_mustard_bottle", "007_tuna_fish_can", \
                          "008_pudding_box", "009_gelatin_box", \
                          "010_potted_meat_can", "011_banana", \
                          "019_pitcher_base","021_bleach_cleanser", \
                          "024_bowl", "025_mug", \
                          "035_power_drill", "036_wood_block", "037_scissors",\
                          "040_large_marker","051_large_clamp", \
                          "052_extra_large_clamp","061_foam_brick"]

def load_mesh(target_object, viz=True, calculate_data=False, data_folder = ycb_data_folder, mesh_filename = "poisson/nontextured.ply"):
    # mesh_filename = os.path.join(ycb_data_folder + target_object, "google_16k", "nontextured.ply") #pitcher google_16k is fine, drill is not
    mesh_filename = os.path.join(data_folder + target_object, mesh_filename)

    mesh = o3d.io.read_triangle_mesh(mesh_filename)
    mesh.compute_vertex_normals()
    if calculate_data:
        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        self_intersecting = mesh.is_self_intersecting()
        watertight = mesh.is_watertight()
        orientable = mesh.is_orientable()

        print(f"  edge_manifold:          {edge_manifold}")
        print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
        print(f"  vertex_manifold:        {vertex_manifold}")
        print(f"  self_intersecting:      {self_intersecting}")
        print(f"  watertight:             {watertight}")
        print(f"  orientable:             {orientable}")
    if viz:
        keypoint_radius = 0.005

        keypoint_markers = []
        keypoints_xyz = [[.1,0,0], [0,.1,0], [0,0,.1], [0,0,0]]
        for xyz in [keypoints_xyz[0]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[1]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.0, 0.8, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[2]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.0, 0.0, 0.8])
            keypoint_markers.append(new_mesh)
        for xyz in [keypoints_xyz[3]]:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.8, 0.8])
            keypoint_markers.append(new_mesh)
        o3d.visualization.draw_geometries([mesh] + keypoint_markers)
    return mesh

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def save_kpts(kpt_idxs, points_xyz, target_oject):
    """
    Takes list of kpt indices in the points_xyz np.ndarray and saves into npy file
    :param kpt_idxs:
    :param points_xyz:
    :return:
    """
    kpt_xyz = np.asarray(points_xyz)[kpt_idxs]
    kpt_filename = os.path.join(ycb_data_folder + target_object, "kpts_xyz.npy")
    np.save(kpt_filename, kpt_xyz)
    loaded_array = np.load(kpt_filename)
    print("saved and loaded array", loaded_array)

def load_obj_from_ref_H(target_object, camera, viewpoint_angle):
    calibrationFilename = os.path.join(ycb_data_folder + target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)
    rgbKey = "H_{0}_from_{1}".format(camera, referenceCamera)
    rgbFromRef = calibration[rgbKey][:]

    objFromRefFilename = os.path.join(ycb_data_folder + target_object, 'poses',
                                      '{0}_{1}_pose.h5'.format(referenceCamera, viewpoint_angle))
    objFromRef = h5.File(objFromRefFilename)['H_table_from_reference_camera'][:]
    print("objFromRef", objFromRef)
    print("rgbFromRef", rgbFromRef)

    return np.asarray(rgbFromRef), np.asarray(objFromRef)

def load_intrinsics(target_object, viewpoint_camera):
    calibrationFilename = os.path.join(ycb_data_folder + target_object, "calibration.h5")
    calibration = h5.File(calibrationFilename)
    rgbK = calibration["{0}_rgb_K".format(viewpoint_camera)][:]
    return rgbK

def get_closest_cluster(input_pcd, gt_pcd, viz=False):
    # downsample pcd
    if len(input_pcd.points) > 70000:  # maybe cap at 80000?
        print("DOWNSAMPLING to do clustering")
        input_pcd = input_pcd.voxel_down_sample(voxel_size=0.001)
        print("pcd now has reduced number of points", len(input_pcd.points))

    input_points = torch.Tensor(np.array(input_pcd.points)).unsqueeze(0)
    gt_pcd = torch.Tensor(np.array(gt_pcd.points)).unsqueeze(0)
    print(input_points.shape)
    print(gt_pcd.shape)
    sq_dist, _, _ = ops.knn_points(input_points, gt_pcd, K=1, return_sorted=False)
    #mask sq_dist (if it is a numpy array)
    print("sq_dist", sq_dist.shape)
    mask_dist = torch.sqrt(sq_dist).le(0.01).squeeze(0) #keep points within a centimeter to the closest point
    print("mask_dist shape", mask_dist.shape)
    mask_dist = torch.hstack((mask_dist, mask_dist, mask_dist))
    print("mask_dist shape", mask_dist.shape)
    output_points = torch.masked_select(input_points.squeeze(0), mask_dist)
    output_points = output_points.reshape((-1, 3))
    print("output points shape", output_points.shape)
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(output_points.numpy()) #nts: .numpy() is optimized conversion to numpy array

    if viz:
        input_pcd.paint_uniform_color([.5, 0, 0])

        o3d.visualization.draw_geometries([input_pcd])
        output_pcd.paint_uniform_color([0, .5, 0])

        o3d.visualization.draw_geometries([output_pcd])

    if len(output_pcd.points) < 500:
        print("less than 500 points in this point cloud after closest point clustering, skipping")
        return None


    return output_pcd



def get_largest_cluster(pcd, viz=False, eps=0.5):
    #downsample pcd
    if len(pcd.points) > 70000: #maybe cap at 80000?
        print("DOWNSAMPLING to do clustering")
        pcd = pcd.voxel_down_sample(voxel_size=0.001)
        print("pcd now has reduced number of points", len(pcd.points))

    cluster = np.array(pcd.cluster_dbscan(eps=eps, min_points=400, print_progress=True))
    max_label = cluster.max()
    print("number of clusters", max_label + 1)
    if max_label + 1 == 0:
        print("No clusters found")
        return None

    counts = np.bincount(np.where(cluster > 0, cluster, 0))
    largest_cluster_label = np.argmax(counts)
    bool_indices_of_largest_cluster = (cluster == largest_cluster_label)
    largest_cluster_points = np.array(pcd.points)[bool_indices_of_largest_cluster]


    cmap = plt.get_cmap("tab20")
    colors = cmap(cluster / (max_label if max_label > 0 else 1))
    colors[cluster < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    primary_cluster_pcd = o3d.geometry.PointCloud()
    primary_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

    if len(pcd.points) < 500:
        print("less than 500 points in this point cloud, skipping")
        return None

    o3d.geometry.PointCloud.estimate_normals(primary_cluster_pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                                                    max_nn=30))
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(primary_cluster_pcd, camera_location=np.array([0., 0., 0.]))
    primary_cluster_pcd.paint_uniform_color([.5, .5, .5])
    if viz:
        o3d.visualization.draw_geometries([pcd])
        o3d.visualization.draw_geometries([primary_cluster_pcd])
    return primary_cluster_pcd

def load_image_and_model_ycb_video(scene, frame):
    data_folder = os.path.join(ycb_video_data_folder, "data", str(scene).zfill(4))

    basename = "{0}-".format(frame)
    rgbFilename = os.path.join(data_folder, basename + "color.png")
    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    rgbK = np.array([[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]])

    # we need to transform the mesh to world coordinates wrt the rgb camera
    metadataFilename = os.path.join(data_folder, basename + "meta.mat")
    metadata = scipy.io.loadmat(metadataFilename)
    for obj in range(len(metadata['cls_indexes'])):
        model_id = metadata['cls_indexes'][obj][0] -1
        objFromCam = metadata['poses'][:, :, obj]
        print("model_id", model_id)
        target_object = YCB_VIDEO_ID_TO_MODEL[model_id]
        mesh = load_mesh(target_object, viz=True, data_folder=ycb_video_data_folder + "/models/",
                         mesh_filename="textured.obj")

        points = mesh.vertices  # these are mesh model points (we assume 0,0,0 is at the center bottom)?
        points_array = np.asarray(points)

        #mesh points in the rgb coordinates are H_world_to_camera @ points_in_world
        objFromCam = np.vstack((objFromCam, np.zeros((objFromCam.shape[1]))))
        objFromCam[3,3] = 1
        print(objFromCam.shape)

        camera_from_obj = objFromCam #np.linalg.inv(objFromCam) gt poses are defined other way around
        rmat_cam_from_obj = camera_from_obj[:3,:3]

        rvect_cam_from_obj, _ = cv2.Rodrigues(rmat_cam_from_obj)
        tvect_cam_from_obj = camera_from_obj[:3,3]

        points_homo = np.hstack((points_array, np.ones((points_array.shape[0], 1))))
        print("points_homo.shape", points_homo.shape)
        image_pts = rgbK @ camera_from_obj[:3, :] @ points_homo.T  # transform the world points to camera coordinates
        image_pts /= image_pts[-1, :]
        print("image_pts.shape", image_pts.shape)
        plot_pts_on_image(image_pts, rgbFilename)


def load_image_and_model2(target_object, viewpoint_camera, viewpoint_angle):
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    mesh = load_mesh(target_object)
    points = mesh.vertices #these are mesh model points (we assume 0,0,0 is at the center bottom)?
    points_array = np.asarray(points)

    rgbFromRef, objFromRef = load_obj_from_ref_H(target_object, viewpoint_camera, viewpoint_angle)
    table_from_camera = objFromRef @ np.linalg.inv(rgbFromRef) #we want transformation from camera to ref, then ref to table

    rgbK = load_intrinsics(target_object, viewpoint_camera)


    # we need to transform the mesh to world coordinates wrt the rgb camera
    print("points.shape", points_array.shape)
    #mesh points in the rgb coordinates are H_world_to_camera @ points_in_world
    camera_from_table = rgbFromRef @ np.linalg.inv(objFromRef)
    rmat_cam_from_table = camera_from_table[:3,:3]
    rvect_cam_from_table, _ = cv2.Rodrigues(rmat_cam_from_table)
    tvect_cam_from_table = camera_from_table[:3,3]

    ### real pcl
    # pcd = o3d.io.read_point_cloud(ycb_data_folder+target_object+"/clouds/largest_cluster/pc_"+viewpoint_camera+"_"+referenceCamera+"_"+str(viewpoint_angle)+"_masked.ply")
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # points_array = np.asarray(pcd.points)
    # print(points_array[:3])
    # points_homo = np.hstack((points_array, np.ones((points_array.shape[0], 1))))
    # print("points_homo.shape", points_homo.shape)
    # image_pts = rgbK @ points_array.T  # transform the world points to camera coordinates
    # image_pts /= image_pts[-1, :]
    # print("image_pts.shape", image_pts.shape)
    # plot_pts_on_image(image_pts, rgbFilename)
    ###
    #####
    # points_array = np.asarray([[0,0,0], [.1,0,0], [0,.1,0], [0,-.1,0]])
    print(points_array[:3])
    points_homo = np.hstack((points_array, np.ones((points_array.shape[0],1))))
    print("points_homo.shape", points_homo.shape)
    image_pts = rgbK @ camera_from_table[:3,:] @ points_homo.T #transform the world points to camera coordinates
    image_pts /= image_pts[-1, :]
    print("image_pts.shape", image_pts.shape)
    plot_pts_on_image(image_pts, rgbFilename)
    #####

    ##I have gt coordinates of the poses in objFromRef, I need to save instead a npy file 4x4 transformation
    ## of obj from rgbCamera to compare against points on pcl

def save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle, only_return = False):
    '''
    Calculates the transformation matrix from the object origin of the poisson reconstructed mesh model wrt the rgb viewpoint camera at framew viewpoint_angle
    :param target_object: the object we're calculating transformations from
    :param viewpoint_camera: the rgb camera ("NP1", "NP2", etc.) of interest
    :param viewpoint_angle: the angle the camera is viewing the object from (only used to locate files due to naming conventions)
    :param only_return:
    :return:
    '''
    basename = "{0}_{1}".format(viewpoint_camera, viewpoint_angle)
    rgbFilename = os.path.join(ycb_data_folder + target_object, basename + ".jpg")
    if not os.path.isfile(rgbFilename):
        print(
            "The rgbd data is not available for the target object \"%s\". Please download the data first." % target_object)
        exit(1)
    rgbFromRef, objFromRef = load_obj_from_ref_H(target_object, viewpoint_camera, viewpoint_angle)
    objFromRGB = objFromRef @ np.linalg.inv(rgbFromRef)  # we want transformation from camera to ref, then ref to table
    rgbFromObj = rgbFromRef @ np.linalg.inv(objFromRef)
    if only_return:
        return rgbFromObj

    if not os.path.exists(ycb_data_folder + "/" + target_object + "/poses/gt_wrt_rgb"):
        os.makedirs(ycb_data_folder + "/" + target_object + "/poses/gt_wrt_rgb")

    rgbFromObj_filename = os.path.join(ycb_data_folder + target_object, "poses/gt_wrt_rgb/", '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
    np.save(rgbFromObj_filename, rgbFromObj)
    loaded_array = np.load(rgbFromObj_filename)
    print("saved and loaded array", loaded_array)
    return rgbFromObj

def viz_and_save_depth_pc(model_id, split='train', save_loc='./temp'):
    save_loc = save_loc + "/" + str(model_id) + '/' + str(split) + '/'
    dataset = DepthYCB(model_id=model_id,
                       split=split,
                       num_of_points=500)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    for i, data in enumerate(dataloader):
        input_point_cloud, keypoints_target, R_target, t_target = data
        # displaying only the first item in the batch
        input_point_cloud = input_point_cloud[0, ...].to('cpu')
        target_keypoints = keypoints_target[0, ...].to('cpu')

        input_point_cloud = gu.pos_tensor_to_o3d(input_point_cloud)
        input_point_cloud.paint_uniform_color([0.7, 0.7, 0.7])

        target_keypoints = target_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz in target_keypoints:
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.01)
            kpt_mesh.translate(xyz)
            kpt_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(kpt_mesh)
        target_keypoints = keypoint_markers

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcl_for_vis = o3d.geometry.PointCloud()
        pcl_for_vis.points = input_point_cloud.points
        vis.add_geometry(pcl_for_vis)
        for keypoint in target_keypoints:
            vis.add_geometry(keypoint)
        # image = vis.capture_screen_image(save_loc + '/' + str(i) + '.png', True)
        vis.run()
        vis.destroy_window()

        # o3d.visualization.draw_geometries([input_point_cloud] + target_keypoints)

def get_degenerate_angles(): #+/- 10 degrees, -1 means all angles
    def is_degenerate(target_object, filename, obj_to_degenerate_cam_angles):
        # returns true if in hardcoded dictionary, o.w. false
        _, reference_cam, _, angle, _ = filename.split("_")
        ref_idx = int(reference_cam[-1])-1
        ref_angle = int(angle)
        for degenerate_angle in obj_to_degenerate_cam_angles[target_object][ref_idx]:
            if degenerate_angle == -1:
                return True
            assert ref_idx !=  4
            if degenerate_angle - 10 < ref_angle and degenerate_angle + 10 > ref_angle:
                return True
        return False
    obj_to_degenerate_cam_angles = {"003_cracker_box": [[0, 96, 186, 270, 357],[],[],[],[-1]],
     "004_sugar_box": [[3,93,183, 273, 357],[],[],[],[-1]],
     "008_pudding_box": [[0,177,267,357],[],[],[219,348],[-1]],
     "009_gelatin_box": [[6,186,357], [], [], [], [-1]],
     "036_wood_block": [[6,99,189,279], [], [], [], [-1]],
     "061_foam_brick": [[], [], [], [], [-1]],
     "001_chips_can": [[], [], [], [], [-1]],
     "002_master_chef_can": [[], [], [], [], [-1]],
     "005_tomato_soup_can": [[], [], [], [], [-1]],
     "007_tuna_fish_can": [[], [], [], [], [-1]]
     }
    for target_object in obj_to_degenerate_cam_angles:
        train_split_new_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                                "train_split_wo_degeneracy.npy")
        val_split_new_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                                "val_split_wo_degeneracy.npy")
        test_split_new_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                                "test_split_wo_degeneracy.npy")
        train_split_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                                "train_split.npy")
        val_split_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                            "val_split.npy")
        test_split_filename = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster", \
                                            "test_split.npy")
        train_array = np.load(train_split_filename)
        val_array = np.load(val_split_filename)
        test_array = np.load(test_split_filename)
        train_wo_degeneracy_array = []
        val_wo_degeneracy_array = []
        test_wo_degeneracy_array = []
        for filename in train_array:
            if not is_degenerate(target_object, filename, obj_to_degenerate_cam_angles):
                train_wo_degeneracy_array.append(filename)
        for filename in val_array:
            if not is_degenerate(target_object, filename, obj_to_degenerate_cam_angles):
                val_wo_degeneracy_array.append(filename)
        for filename in test_array:
            if not is_degenerate(target_object, filename, obj_to_degenerate_cam_angles):
                test_wo_degeneracy_array.append(filename)
        print("target_object", target_object)
        print("number of pcls in new train_wo_degeneracy_array", len(train_wo_degeneracy_array))
        print("number of pcls in new val_wo_degeneracy_array", len(val_wo_degeneracy_array))
        print("number of pcls in new test_wo_degeneracy_array", len(test_wo_degeneracy_array))
        np.save(train_split_new_filename, np.array(train_wo_degeneracy_array))
        np.save(val_split_new_filename, np.array(val_wo_degeneracy_array))
        np.save(test_split_new_filename, np.array(test_wo_degeneracy_array))

def do_data_augmentation(pcd_filepath, filename, percentage_points_to_remove, num_augmentations= 9):
    print(pcd_filepath)
    pcd = o3d.io.read_point_cloud(pcd_filepath + '/' + filename)
    #o3d.visualization.draw_geometries([pcd])
    pcd = np.array(pcd.points)

    for aug_idx in range(num_augmentations):
        num_pts_to_remove = int(percentage_points_to_remove * pcd.shape[0])
        new_points = pcd.copy()
        for i in range(num_pts_to_remove):
            rand_idx_to_remove = np.random.randint(new_points.shape[0])
            new_points = np.delete(new_points, rand_idx_to_remove, 0)
        print("FINISHED REMOVING PERCENTAGE OF POINTS")
        if new_points.shape[0] < 500:
            print("less than 500 points in this point cloud, not removing any more")
            new_points = pcd.copy()
        random_noise = np.random.normal(0, .001, new_points.shape)
        new_points = new_points + random_noise
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(new_points)  # nts: .numpy() is optimized conversion to numpy array
        #o3d.visualization.draw_geometries([output_pcd])
        data_augmented_pcd_filepath = pcd_filepath[:-15] + 'data_augmentation/' + filename[:-4] + "_aug_" + str(aug_idx).zfill(3) + ".ply"
        print(data_augmented_pcd_filepath)
        if not os.path.exists(pcd_filepath[:-15] + 'data_augmentation/'):
            os.makedirs(pcd_filepath[:-15] + 'data_augmentation/')
        o3d.io.write_point_cloud(data_augmented_pcd_filepath, output_pcd)


if __name__=="__main__":
    # get_degenerate_angles()
    # target_object = "001_chips_can"#"021_bleach_cleanser" #"019_pitcher_base"#"035_power_drill"	# Full name of the target object.
    # for target_object in ["007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "011_banana"]:
    # for target_object in ["024_bowl", "036_wood_block", "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]:
    # for target_object in ["004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle"]:
    # for target_object in ["001_chips_can", "002_master_chef_can", "003_cracker_box", \
    #                       "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", \
    #                       "007_tuna_fish_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", \
    #                       "011_banana", "019_pitcher_base", "021_bleach_cleanser", \
    #                       "035_power_drill", "036_wood_block", "037_scissors", \
    #                       "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]:
    # for target_object in ["021_bleach_cleanser"]: #fix the primes

        # mesh = load_mesh(target_object, viz=True)
        # points = mesh.vertices
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = points
        # picked_points_idx = pick_points(pcd)
        # save_kpts(picked_points_idx, points, target_object)

        # ### testing transformations
        # load_image_and_model2(target_object, "NP1", 30)

        # ### saving ground truth transformations wrt saved point cloud frame of reference
        # for viewpoint_camera in ["NP1", "NP2", "NP3", "NP4", "NP5"]:
        #     for viewpoint_angle in range(358):
        #         if viewpoint_angle%3 != 0:
        #             continue
        #         save_rgbFromObj(target_object, viewpoint_camera, viewpoint_angle)

        ### making train/val/testing splits:
        # pcd_filepath = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster")
        #
        # saved_point_clouds = []
        # for filename in os.listdir(pcd_filepath):
        #     if filename in ["train_split.npy", "val_split.npy", "test_split.npy"]:
        #         continue
        #     saved_point_clouds.append(filename)
        # random.shuffle(saved_point_clouds)
        # num_test = int(.1 * len(saved_point_clouds))
        # num_val = num_test
        # num_train = len(saved_point_clouds) - num_test - num_val
        #
        # train_split = saved_point_clouds[:num_train]
        # val_split = saved_point_clouds[num_train:num_train+num_val]
        # test_split = saved_point_clouds[num_train+num_val:]
        # #save splits in npy
        # split_path = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster")
        # np.save(split_path + '/train_split.npy', train_split)
        # np.save(split_path + '/val_split.npy', val_split)
        # np.save(split_path + '/test_split.npy', test_split)

        # ### vis and save images
        # for obj in ["019_pitcher_base", "021_bleach_cleanser"]:
        #     viz_and_save_depth_pc(obj, 'test')

        # ### for data augmentation
        # print("target_object", target_object)
        # pcd_filepath = os.path.join(ycb_data_folder + target_object, "clouds", "largest_cluster")
        # for filename in os.listdir(pcd_filepath):
        #     do_data_augmentation(pcd_filepath, filename, percentage_points_to_remove=.1)
        #
        # ### making train/val/testing splits for data augmented dataset:
        # pcd_augmented_filepath = os.path.join(ycb_data_folder + target_object, "clouds", "data_augmentation")
        # saved_point_clouds = []
        # for filename in os.listdir(pcd_augmented_filepath):
        #     if filename in ["train_split.npy", "val_split.npy", "test_split.npy"]:
        #         continue
        #     saved_point_clouds.append(filename)
        # random.shuffle(saved_point_clouds)
        # num_test = int(.1 * len(saved_point_clouds))
        # num_val = num_test
        # num_train = len(saved_point_clouds) - num_test - num_val
        #
        # train_split = saved_point_clouds[:num_train]
        # val_split = saved_point_clouds[num_train:num_train+num_val]
        # test_split = saved_point_clouds[num_train+num_val:]
        # #save splits in npy
        # split_path = os.path.join(ycb_data_folder + target_object, "clouds", "data_augmentation")
        # np.save(split_path + '/train_split.npy', train_split)
        # np.save(split_path + '/val_split.npy', val_split)
        # np.save(split_path + '/test_split.npy', test_split)
    mixed_dataset_filepath = os.path.join(ycb_data_folder + "/mixed/")
    saved_training_point_clouds = []
    saved_validation_point_clouds = []

    for target_object in ["002_master_chef_can", "006_mustard_bottle", "037_scissors", "052_extra_large_clamp", "011_banana"]:
        # ### making train/val/testing splits for MIXED data augmented dataset:
        pcd_augmented_filepath = os.path.join(ycb_data_folder + target_object, "clouds", "data_augmentation/")

        train_split_filenames = np.load(pcd_augmented_filepath + 'train_split.npy')
        train_random_subset = np.random.choice(train_split_filenames, size=1000, replace=False)
        for filename in train_random_subset:
            saved_training_point_clouds.append(os.path.join(target_object, filename))
        val_split_filenames = np.load(pcd_augmented_filepath + 'val_split.npy')
        val_random_subset = np.random.choice(train_split_filenames, size=50, replace=False)
        for filename in val_random_subset:
            saved_validation_point_clouds.append(os.path.join(target_object, filename))

    # at this point, 1000 of each model from it's training/val split will be in saved_point_cloud
    print("number of point clouds in saved_training_point_clouds", len(saved_training_point_clouds))
    print(saved_training_point_clouds[0])
    print("number of point clouds in saved_val_point_clouds", len(saved_validation_point_clouds))
    print(saved_validation_point_clouds[0])

    split_path = os.path.join(mixed_dataset_filepath)
    np.save(split_path + '/train_split.npy', saved_training_point_clouds)
    np.save(split_path + '/val_split.npy', saved_validation_point_clouds)

