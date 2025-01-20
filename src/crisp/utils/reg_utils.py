import sys, os, json, copy

import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import open3d as o3d
import trimesh
import numpy as np
import pymeshfix
import meshlab_pickedpoints
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import KDTree

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'apolloscape', 'dataset-api'))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '../casper3d', 'third_party', 'apolloscape', 'dataset-api', 'utils'))
import apollo_utils as uts
from car_instance.car_models import *

# solvers
# sys.path.append("../solvers")
# import category_gnc as cgnc

# pyrender
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

TOTAL_NUM_KPTS = 66


def load_single_2d_kpt_file(filepath):
    """Load a single keypoint file

    :param filepath:
    :return:
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        kpts = {}
        for line in lines:
            tokens = line.split("\t")
            id = int(tokens[0])
            x = float(tokens[1])
            y = float(tokens[2])
            kpts[id] = (x, y)
    return kpts


def load_2d_keypoints(kpts_dir_path):
    """ Load all 2D keypoints in directory

    :param kpts_dir_path:
    :return: a list of dictionaries containing 2D keypoints
    """
    all_kpts = []
    sorted_filelist = os.listdir(kpts_dir_path)
    for file in sorted_filelist:
        if file.endswith(".txt"):
            kpts = load_single_2d_kpt_file(os.path.join(kpts_dir_path, file))
            all_kpts.append(kpts)

    return all_kpts


def load_kpt_lib(dir_root):
    """Load directory of PP files

    :param dir_root:
    :return:
    """
    kpt_db = []
    avaialble_cads = set()
    # load keypoints - vehicle dicts only for cars with full number of keypoints
    for file in sorted(os.listdir(dir_root)):
        if file.endswith(".pp"):
            with open(os.path.join(dir_root, file), 'r') as f:
                pts = meshlab_pickedpoints.load(f)
                car_name = file.split(".")[0]
                flag = len(pts) == TOTAL_NUM_KPTS
                if flag:
                    kpt_list = [None] * TOTAL_NUM_KPTS
                    for kpt, pt in pts.items():
                        kpt_list[int(kpt)] = pt
                        if pt[0] == 0 and pt[1] == 0:
                            flag = False
                            break
                else:
                    print("WARNING: {} missing points".format(file))
                if flag:
                    kpt_db.append({'kpts': np.transpose(np.array(kpt_list)), 'name': car_name})
                    avaialble_cads.add(car_name)
        else:
            print("WARNING: {} unknown file".format(file))
    return kpt_db, avaialble_cads


def load_split_entries(split, sample_data = False):
    if sample_data:
        root_dir = "../../dataset/apollo_car_3d/3d_car_instance_sample"
    else:
        root_dir = "../../dataset/apollo_car_3d/3d-car-understanding-train/train"
    split_dir = os.path.join(root_dir, "split")
    if split == "train":
        train_list = "train.txt" if sample_data else "train-list.txt"
        split_file_path = os.path.join(split_dir, train_list)
    elif split == "validation":
        val_list = "val.txt" if sample_data else "validation-list.txt"
        split_file_path = os.path.join(split_dir, val_list)
    else:
        raise ValueError("Wrong dataset split type.")

    with open(split_file_path, "r") as split_file:
        valid_entries = split_file.readlines()
        valid_entries = [x.strip() for x in valid_entries]

    return valid_entries

def load_apollo_data(split="train"):
    root_dir = "../../dataset/apollo_car_3d/3d-car-understanding-train/train"
    valid_entries = load_split_entries(split)
    data_collection = []
    image_dir = root_dir + "/images"
    for file in os.listdir(image_dir):
        if file.endswith(".jpg") and file in valid_entries:
            data = {}
            image_name = file.split(".")[0]
            data["im"] = os.path.join(image_dir, file)
            data["im_name"] = image_name
            data["car_metadata"] = os.path.join(root_dir, "car_poses", image_name + ".json")
            data["keypoints_dir"] = os.path.join(root_dir, "keypoints", image_name + "/")
            data_collection.append(data)

    # intrinsic (cam 5)
    # [fx, fy, cx, cy]
    K = np.array([2304.54786556982, 2305.875668062, 1686.23787612802, 1354.98486439791])
    # car_model_dir_path = "/data/dataset/apolloscape/pku-autonomous-driving/car_models_json/"
    car_model_dir_path = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models_json"

    results = {"data": data_collection, "K": K, "car_models_dir": car_model_dir_path}
    return results


def load_car_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_car_model(json_path):
    '''

    :param json_path:
    :return: verts: np.array where (x,y,z) coordinates are the columns
    faces: list of list of vertices that define a face
    '''
    with open(json_path, "r") as f:
        data = json.load(f)

    # verts
    verts = np.transpose(np.array(data["vertices"]))
    # shouldn't matter if x axis is flipped because cars are symmetrical
    # x axis right, y axis up, z axis forward
    verts[1, :] *= -1
    # verts[[0, 1], :] *= -1
    # faces
    faces = np.array(data["faces"]) - 1

    return verts, faces


def project2d(input_K, p):
    """Project 3D points to 2D image

    :return:
    """
    if input_K.shape[0] != 3:
        K = uts.intrinsic_vec_to_mat(input_K)
    else:
        K = input_K
    pi = np.zeros((3, 4))
    pi[:3, :3] = np.identity(3)
    #no rotation and translation
    image_pts = K @ pi @ p
    image_pts /= image_pts[-1, :]
    return image_pts


def generate_weighted_model(solution, cad_db):
    """Generate a weighted set of points

    :param solution:
    :param cad_db:
    :return:
    """
    # number of CAD models available equals to number of model weights
    assert solution['estimate'][2].shape[0] == len(cad_db)

    # number of keypoints
    N = cad_db[0]['kpts'].shape[1]
    # number of available CADs
    K = len(cad_db)

    model_weights = solution['estimate'][2]
    summed_model = np.zeros((3, N))
    for i in range(K):
        summed_model += model_weights[i].astype('float64') * cad_db[i]['kpts'].astype('float64')

    return summed_model


def render_tsdf_mask(mesh, pose, img_width, img_height, K):
    # apply rotation and translation
    if np.ndim(pose) == 1:
        mat = uts.trans_vec_to_mat(pose[:3], pose[3:])
    elif np.ndim(pose) == 2:
        mat = pose
    else:
        raise ValueError("Incorrect pose input.")

    # render
    color = render_tsdf_mesh(mat, mesh, K, img_width=img_width, img_height=img_height)

    # Thresholding
    grey = cv2.cvtColor(color, cv2.COLOR_BGRA2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)

    return mask, color


def render_tsdf_mesh(unrectC_T_W_GT, mesh, K, img_width=1280, img_height=1280):
    """Render a CAD model on 2D (for TSDF meshes)
    :return:
    """
    assert np.ndim(K) == 1
    #
    # Render GT
    #
    # project GT model to image
    # convert from world frame to camera frame
    T_rotx = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
    T_opengl = T_rotx @ unrectC_T_W_GT

    # generate trimesh
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    colors = np.tile(np.array([[255, 205, 0, 255]]), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = colors

    mesh_render = pyrender.Mesh.from_trimesh(mesh)

    # render in scene
    scene = pyrender.Scene(ambient_light=[.1, .1, .1], bg_color=[0, 0, 0, 0])
    camera = pyrender.IntrinsicsCamera(K[0], K[1], K[2], K[3])
    light = pyrender.PointLight(color=[255, 255, 255], intensity=2e3, range=100)

    scene.add(mesh_render, pose=T_opengl)
    scene.add(light, pose=np.array([[1, 0, 0, 10], [0, 1, 0, 50], [0, 0, 1, -10], [0, 0, 0, 1]]))
    scene.add(camera, pose=np.eye(4))

    # project to camera
    r = pyrender.OffscreenRenderer(img_width, img_height)
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT)

    return color


def render_cad(unrectC_T_W_GT, W_carVerts, car_faces, K, img_width=3384, img_height=2710):
    """Render a CAD model on 2D

    :return:
    """
    assert np.ndim(K) == 1
    #
    # Render GT
    #
    # project GT model to image
    # convert from world frame to camera frame
    T_rotx = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
    T_opengl = T_rotx @ unrectC_T_W_GT

    # generate trimesh
    mesh = trimesh.Trimesh(vertices=np.transpose(W_carVerts[:3, :]), faces=car_faces)
    colors = np.tile(np.array([[255, 205, 0, 255]]), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = colors

    mesh_render = pyrender.Mesh.from_trimesh(mesh)

    # render in scene
    scene = pyrender.Scene(ambient_light=[.1, .1, .1], bg_color=[0, 0, 0, 0])
    camera = pyrender.IntrinsicsCamera(K[0], K[1], K[2], K[3])
    light = pyrender.PointLight(color=[255, 255, 255], intensity=2e3, range=100)

    scene.add(mesh_render, pose=T_opengl)
    scene.add(light, pose=np.array([[1, 0, 0, 10], [0, 1, 0, 50], [0, 0, 1, -10], [0, 0, 0, 1]]))
    scene.add(camera, pose=np.eye(4))

    # project to camera
    r = pyrender.OffscreenRenderer(img_width, img_height)
    color, _ = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT)

    return color


def render_cads(meshes, transforms, K, img_width=3384, img_height=2710, repair=False, distinct_colors=False,
                flags=pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT):
    """Render multiple CADs
    """
    T_rotx = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
    scene = pyrender.Scene(ambient_light=[.1, .1, .1], bg_color=[0, 0, 0, 0])
    camera = pyrender.IntrinsicsCamera(K[0], K[1], K[2], K[3])
    light = pyrender.PointLight(color=[255, 255, 255], intensity=2e3, range=100)
    scene.add(light, pose=np.array([[1, 0, 0, 10], [0, 1, 0, 50], [0, 0, 1, -10], [0, 0, 0, 1]]))
    scene.add(camera, pose=np.eye(4))

    color_palette = [[255, 205, 0, 255],
                     [255, 0, 205, 255],
                     [0, 205, 255, 255],
                     [125, 205, 122, 255],
                     [255, 0, 205, 255],
                     ]
    count = 0
    for mesh, unrectC_T_W_GT in zip(meshes, transforms):
        T_opengl = T_rotx @ unrectC_T_W_GT

        # generate trimesh
        if repair:
            mesh.fill_holes()

        if not distinct_colors:
            colors = np.tile(np.array([[255, 205, 0, 255]]), (len(mesh.vertices), 1))
        else:
            colors = np.tile(np.array([color_palette[count % len(color_palette)]]), (len(mesh.vertices), 1))

        mesh.visual.vertex_colors = colors
        mesh_render = pyrender.Mesh.from_trimesh(mesh)

        # render in scene
        scene.add(mesh_render, pose=T_opengl)
        count += 1

    # project to camera
    r = pyrender.OffscreenRenderer(img_width, img_height)
    color, _ = r.render(scene, flags=flags)

    return color


def plot_scatter_on_img(img, points, radius=5, color=(0, 0, 255), thickness=-1):
    """Plot scatter points on opencv image
    """
    img_canvas = copy.deepcopy(img)
    for i in range(points.shape[1]):
        img_canvas = cv2.circle(img, (int(points[0, i]), int(points[1, i])), radius, color, thickness)
    return img_canvas


def est_model_to_mesh(solution, cad_db):
    points = generate_weighted_model(solution, cad_db)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.transpose(points))
    # pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(100)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #    pcd, depth=5)
    # o3d.visualization.draw_geometries([mesh])
    # radii = [0.5, 0.5, 0.5, 0.5]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #    pcd, o3d.utility.DoubleVector(radii))
    mesh, _ = pcd.compute_convex_hull()
    return mesh


def select_cad_dist_bounds(ids_to_select, original_cad_dist_min, original_cad_dist_max, cad_dist_i_map, cad_dist_j_map):
    """Select CAD db distance bounds by the indices of semantic keypoints provided.

    :param ids_to_select:
    :param original_cad_dist_min:
    :param original_cad_dist_max:
    :param cad_dist_i_map:
    :param cad_dist_j_map:
    :return:
    """
    # obtain the indices of distance bounds array that we want to keep
    counter = 0
    bound_idx = []
    for i, j in zip(list(cad_dist_i_map), list(cad_dist_j_map)):
        # if both (i,j) are in the semantic keypoint list, select
        if i in ids_to_select and j in ids_to_select:
            bound_idx.append(counter)
        counter += 1

    # select
    dist_min = original_cad_dist_min[np.array(bound_idx).astype("int")]
    dist_max = original_cad_dist_max[np.array(bound_idx).astype("int")]

    return dist_min, dist_max


def dists_from_ray(ray, points):
    """Find distances from ray to 3D points
    """
    # normalize the ray
    n = ray / np.linalg.norm(ray)
    a = np.zeros((3, 1))
    dists = []
    for i in range(points.shape[1]):
        p = points[:3, i]
        dist = np.linalg.norm((a - p) - np.dot((a - p), n) * n)
        dists.append(dist)

    return dists


def load_disparity_to_depth(im_path, Q):
    '''Return depth images are 3-channel depths (x,y,z)

    :param im_path:
    :return:
    '''
    scaled_disparity = np.float32(cv2.imread(im_path, cv2.IMREAD_UNCHANGED))
    disparity = scaled_disparity / 200
    depth_im = cv2.reprojectImageTo3D(disparity, Q)
    return depth_im


# To set 3D axes equal scale for visualization
# Credit: https://stackoverflow.com/questions/13685386/
# matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def fix_depth_map(depth_map):
    """
    Credit:
    https://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    :param depth_map:
    :return:
    """
    mask = np.isnan(depth_map)
    depth_map[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth_map[~mask])


def SE3_transform(R, t, points):
    """Transform points using a rotation matrix and a translation vector

    :param R:
    :param t:
    :param points:
    :return:
    """
    assert points.shape[0] == 3
    result = R @ points + t
    return result
