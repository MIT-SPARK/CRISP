import sys, os, json, copy

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import numpy as np
import pymeshfix
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import KDTree

sys.path.append("../../third_party/apolloscape/dataset-api/")
sys.path.append("../../third_party/apolloscape/dataset-api/utils/")
import apollo_utils as uts
from car_instance.car_models import *

# parameters
VISUALIZE = True
QUERY_RADIUS = 3
# APOLLO_DATA_ROOT_DIR = "/data/dataset/apolloscape/car-instance/3d-car-understanding-train/train"
PATH_TO_CAR_MODELS = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models_json/"
PATH_TO_PKL_CAR_MODELS = "../../dataset/apollo_car_3d/3d_car_instance_sample/car_models/"
PATH_TO_TRAIN_FOLDER = "../../dataset/apollo_car_3d/3d-car-understanding-train/train/"

#this code is almost identical to robin's code, it was used to get the 3d car keypoints in keypoints_3d
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
    image_pts = K @ pi @ p
    image_pts /= image_pts[-1, :]
    return image_pts

def plot_pts_on_image(pts, image_path):
    """Plot a bunch of 2D pts on image

    :param pts:
    :param image_path:
    :return:
    """
    im = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.set_window_title("points on image")
    ax.imshow(im)
    ax.scatter(pts[0, :], pts[1, :], s=5)
    ax.axis('off')
    plt.show()
    return

def load_car_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_car_model(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # verts
    verts = np.transpose(np.array(data["vertices"]))
    verts[1, :] *= -1

    # faces
    faces = np.array(data["faces"]) - 1

    return verts, faces

def plot_2d_kpts(kpt_dict, image_path, scatter_pts=None):
    """Plot 2D kpts

    :param kpt_dict:
    :return:
    """
    pts = np.zeros((2, len(kpt_dict)))
    idx = 0
    for k, v in kpt_dict.items():
        pts[0, idx] = v[0]
        pts[1, idx] = v[1]
        idx += 1

    im = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    fig.canvas.set_window_title("points on image")
    ax.imshow(im)
    if scatter_pts is not None:
        ax.scatter(scatter_pts[0, :], scatter_pts[1, :], s=5, c='b')
    ax.scatter(pts[0, :], pts[1, :], s=10, c='r')
    ax.axis('off')
    plt.show()
    return

def load_single_kpt_file(filepath):
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
            kpts = load_single_kpt_file(os.path.join(kpts_dir_path, file))
            all_kpts.append(kpts)

    return all_kpts

def get_corresponding_car_for_kpts(kpts_group, car_2d_points):
    """Find the corresponding car name for a group of kpts

    :param kpts_group:
    :param car_2d_points:
    :return:
    """

    def centeroidnp(arr):
        hull = ConvexHull(arr)
        cx = np.mean(hull.points[hull.vertices, 0])
        cy = np.mean(hull.points[hull.vertices, 1])
        return np.array([cx, cy])

    kpts = [[p[0], p[1]] for _, p in kpts_group.items()]
    kpt_mean = centeroidnp(np.array(kpts))

    rank = []
    for i in range(len(car_2d_points)):
        car_info, val, _, _ = car_2d_points[i]
        car_mean = centeroidnp(val[:2, :].T)
        dist = np.sqrt(np.sum(np.square(car_mean - kpt_mean)))
        rank.append((dist, car_info, i))

    rank.sort(key=lambda x: x[0])
    # return the index
    return rank[0][2]

def load_apollo_data():
    root_dir = PATH_TO_TRAIN_FOLDER
    data_collection = []
    image_dir = root_dir + "/images"
    for file in os.listdir(image_dir):
        if file.endswith(".jpg"):
            data = {}
            image_name = file.split(".")[0]
            data["im"] = os.path.join(image_dir, file)
            data["car_metadata"] = os.path.join(root_dir, "car_poses", image_name + ".json")
            data["keypoints_dir"] = os.path.join(root_dir, "keypoints", image_name + "/")
            data_collection.append(data)

    # intrinsic (cam 5)
    K = np.array([2304.54786556982, 2305.875668062, 1686.23787612802, 1354.98486439791])
    car_model_dir_path = PATH_TO_CAR_MODELS

    results = {"data": data_collection, "K": K, "car_models_dir": car_model_dir_path}
    return results

def postprocess_kpt_db(db):
    """Postprocess the keypoints db

    :param db:
    :return: a new kpts db such that:
    the actual keypoint values are averaged from lists
    and if there is no data for a specific keypoint, (0, 0) is filled
    Warnings will be print for car models missing keypoints
    """
    new_db = copy.deepcopy(db)
    for car_name, kpt_dict in db.items():
        for kpt_id, kpt_list in kpt_dict.items():
            if len(kpt_list) > 0:
                avg_kpt = np.median(np.array(kpt_list), axis=0)
                new_db[car_name][kpt_id] = avg_kpt[:3]
            else:
                new_db[car_name][kpt_id] = np.array([0, 0, 0])

    return new_db


def dump_kpt_db(db, location="./keypoints_3d/"):
    """
    Dump keypoint file
    :param db:
    :param location:
    :return:
    """
    for car_name, kpt_dict in db.items():
        pp_str = meshlab_pickedpoints.dumps(kpt_dict)
        with open(os.path.join(location, "{}.pp".format(car_name)), "w") as text_file:
            text_file.write(pp_str)
    return


if __name__ == "__main__":
    print("=====Attempting to determine keypoints=====")

    dataset = load_apollo_data()
    # dictionary
    # key=car name
    # value=dictionary(key=keypoint_id, value=(x,y))
    TOTAL_NUM_KPTS = 66
    keypoints_db = {model_tuple.name: {kpt_id: [] for kpt_id in range(TOTAL_NUM_KPTS)} for model_tuple in models}

    # iterate through all frames
    for sample_data in tqdm(dataset["data"]):
        try:
            cars_data = load_car_poses(sample_data["car_metadata"])

            # load 2D kpts database
            unrectI_kpts_db = load_2d_keypoints(sample_data["keypoints_dir"])

            # load all gt cars
            unrectI_car_pointset = []
            for car_data in cars_data:
                # load car model
                car_name = car_id2name[car_data["car_id"]]
                W_carVerts, car_faces = load_car_model("{}/{}.json".format(dataset["car_models_dir"], car_name.name))
                W_carVerts = np.vstack((W_carVerts, np.ones((1, W_carVerts.shape[1]))))

                # load car pose
                car_pose = np.array(car_data['pose'])
                unrectC_T_W = uts.trans_vec_to_mat(car_pose[:3], car_pose[3:])

                # apply pose transformation
                unrectC_carVerts = unrectC_T_W @ W_carVerts
                unrectI_car = project2d(dataset["K"], unrectC_carVerts)
                # if VISUALIZE:
                #     plot_pts_on_image(unrectI_car, sample_data["im"])
                # (name, 2d projected points, 3d points in camera frame)
                unrectI_car_pointset.append((car_name, unrectI_car, unrectC_carVerts, unrectC_T_W))

            # skip if no cars on frame
            if len(unrectI_car_pointset) == 0:
                continue

            # for all 2D keypoints, identify closest projected 3D points that have the lowest depth
            # for car_kpts in unrectI_kpts_db:
            #    # car_kpts is the keypoints for one car
            temp_car_2d_pointset = copy.deepcopy(unrectI_car_pointset)
            for kpts_group in unrectI_kpts_db:
                # Determine which car this current group of keypoints belong to
                idx_of_car = get_corresponding_car_for_kpts(kpts_group, temp_car_2d_pointset)
                car_name = temp_car_2d_pointset[idx_of_car][0]

                # double check
                _, unrectI_carGT, unrectC_carVerts, unrectC_T_W = temp_car_2d_pointset[idx_of_car]
                if VISUALIZE:
                    plot_2d_kpts(kpts_group, sample_data["im"], unrectI_carGT)

                # remove idx from the temp list so that the same car in the image will not be
                # selected multiple times
                del temp_car_2d_pointset[idx_of_car]

                # Find 3D points on the model corresponding to the keypoints
                # build KD tree for the 2D car points
                tree_carGT = KDTree(unrectI_carGT[:2, :].T)
                # build KD tree of the 2D semantic kpts
                kpts_idx_list = [i for i, _ in kpts_group.items()]
                tree_kpts = KDTree(np.array([x for _, x in kpts_group.items()]).squeeze())
                # query ball from kpt tree to carGT tree
                indexes = tree_kpts.query_ball_tree(tree_carGT, QUERY_RADIUS)

                for ii in range(len(indexes)):
                    kpt_idx = kpts_idx_list[ii]
                    # sort the list of carGT points by their depths
                    unrectC_carVerts_selected = unrectC_carVerts[:, indexes[ii]]
                    if unrectC_carVerts_selected.size == 0:
                        continue
                    else:
                        unrectC_carVerts_selected = unrectC_carVerts_selected[:,
                                                    np.argsort(unrectC_carVerts_selected[2, :])]

                        # select the point with the lowest depth
                        unrectC_p = unrectC_carVerts_selected[:, 0]

                        # get it in world frame
                        W_p = np.linalg.inv(unrectC_T_W) @ unrectC_p

                        # Save the point to keypoint db
                        keypoints_db[car_name.name][kpt_idx].append(W_p)

                # check result
                if VISUALIZE:
                    print("HI")
                    W_points = np.transpose(
                        np.array([x for _, x in keypoints_db[car_name.name].items() if len(x) > 0]).squeeze())
                    unrectC_points = unrectC_T_W @ W_points
                    unrectI_points = project2d(dataset["K"], unrectC_points)
                    plot_pts_on_image(unrectI_points, sample_data["im"])

                # break the loop if all gt cars have been selected
                # this will happen if number of kpts sets > number of GT cars
                if len(temp_car_2d_pointset) == 0:
                    break

            # post process db (avg the keypoints)
            keypoints_db_tobedumped = postprocess_kpt_db(keypoints_db)

            # dump kpt db to pp file
            dump_kpt_db(keypoints_db_tobedumped)
        except:
            continue
