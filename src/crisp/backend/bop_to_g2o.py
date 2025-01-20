import copy
import cv2 as cv
import csv
import os

from crisp.backend.slam import PoseNode, PoseEdge

# project local imports
from crisp.datasets import bop
from crisp.utils.math import se3_inverse_numpy

"""
Utils
"""


def save_g2o(nodes, edges, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=" ")

        for _node in nodes:
            row = _node.to_g2o_entry_list()
            writer.writerow(row)

        if edges is not None:
            for _edge in edges:
                row = _edge.to_g2o_entry_list()
                writer.writerow(row)


def save_object_index(object_index, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)

        for key, val in object_index.items():
            writer.writerow([key, val])


def read_object_index(filename):
    object_index = dict()
    with open(filename, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            object_label = row[0]
            object_idx = int(row[1])
            object_index[object_label] = object_idx

    return object_index


"""
BOP2G2O and BOP Visualization Code
"""


def bop_to_g2o(
    ds_name="ycbv",
    split="train_real",
    save_dir="../datasets/bopg2o/",
    bop_ds_dir="data/bop/bop_datasets",
    use_TWO_poses=True,
    pose_noise_scale=0.01,
):
    """Load YCBV data through BOP dataset and dump the object poses into a g2o file."""
    # creating directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # setting up the dataset
    dataset = bop.BOPDataset(ds_name="ycbv", split="train_real", bop_ds_dir=bop_ds_dir)

    # initializing
    object_index = dict()
    nodes = []  # VERTEX_SE3:QUAT idx x y z q1 q2 q3 q4
    nodes_gt = []  # VERTEX_SE3:QUAT idx x y z q1 q2 q3 q4
    edges = []  # EDGE_SE3:QUAT idx1 idx2 tx ty tz q1 q2 q3 q4
    # 100.000000 0.000000 0.000000 0.000000 0.000000 0.000000
    # 100.000000 0.000000 0.000000 0.000000 0.000000
    # 100.000000 0.000000 0.000000 0.000000
    # 400.000000 0.000000 0.000000
    # 400.000000 0.000000   400.000000
    current_available_idx = 0
    scene_id_prev = dataset[0][2]["frame_info"]["scene_id"]
    TWC_prev = None
    cam_idx_prev = None

    # iterating over camera poses
    for idx, (rgb, mask, obs) in enumerate(dataset):
        # current scene id
        scene_id = obs["frame_info"]["scene_id"]
        print(f"scene id: {scene_id}, image index: {idx}")
        # if scene is over, then save and restart
        if scene_id != scene_id_prev:
            print("-------------------------")
            print(f"scene id {scene_id_prev} done. Saving scene to g2o.")
            print("-------------------------")
            # saving gt file:
            file_name_gt = save_dir + ds_name + "_" + split + "_" + str(scene_id_prev) + "_gt.g2o"
            _temp = [_node.average() for _node in nodes]
            _temp = [_edge.average() for _edge in edges]
            save_g2o(nodes, edges, file_name_gt)

            # adding noise, resetting init poses (except the first), and saving:
            file_name = save_dir + ds_name + "_" + split + "_" + str(scene_id_prev) + ".g2o"
            if pose_noise_scale != 0:
                _ = [_node.average() for _node in nodes]
                _ = [_edge.average() for _edge in edges]
                _temp = [_node.add_noise(pose_noise_scale) for _node in nodes[1:]]
                _temp = [_edge.add_noise(pose_noise_scale) for _edge in edges[1:]]
            else:
                _temp = [_node.average() for _node in nodes]
                _temp = [_edge.average() for _edge in edges]
            save_g2o(nodes, edges, file_name)

            # saving object index object_index
            file_name = save_dir + ds_name + "_" + split + "_" + str(scene_id_prev) + ".object_index"
            save_object_index(object_index, file_name)

            # initializing
            object_index = dict()
            nodes = []
            nodes_gt = []
            edges = []
            current_available_idx = 0
            scene_id_prev = scene_id
            TWC_prev = None
            cam_idx_prev = None

        # add the camera pose a new node
        camera_node = PoseNode(idx=current_available_idx)
        cam_idx = current_available_idx
        current_available_idx += 1
        # TWC = obs['camera']['TWC']
        TWC = obs["camera"]["T0C"]
        camera_node.add_pose(TWC)
        nodes.append(copy.deepcopy(camera_node))
        del camera_node

        # add relative camera pose
        if TWC_prev is not None:
            # T = TWC_prev @ se3_inverse_numpy(TWC)
            camera_relative_edge = PoseEdge(idx_from=cam_idx_prev, idx_to=cam_idx)
            # we need: T^prev_current
            T = se3_inverse_numpy(TWC_prev) @ TWC
            camera_relative_edge.add_pose(T)
            edges.append(copy.deepcopy(camera_relative_edge))
            del camera_relative_edge

        # check and add new objects in the scene
        _objects = [obs["objects"][i]["label"] for i in range(len(obs["objects"]))]
        _known_objects = object_index.keys()
        _new_objects = [i for i, _obj in enumerate(_objects) if _obj not in _known_objects]

        for i, _obj in enumerate(_objects):
            if i in _new_objects:
                obj_node = PoseNode(idx=current_available_idx)
                object_index[obs["objects"][i]["label"]] = current_available_idx
                current_available_idx += 1
                # TWO = obs['objects'][i]['TWO']
                if use_TWO_poses:
                    TWO = obs["objects"][i]["T0O"]
                    obj_node.add_pose(TWO)
                    nodes.append(copy.deepcopy(obj_node))
                    del obj_node
            else:
                if use_TWO_poses:
                    obj_idx = object_index[obs["objects"][i]["label"]]
                    # TWO = obs['objects'][i]['TWO']
                    TWO = obs["objects"][i]["T0O"]
                    nodes[obj_idx].add_pose(TWO)

        # adding camera-object edges
        for i in range(len(obs["objects"])):
            # TWO = obs['objects'][i]['TWO']
            TWO = obs["objects"][i]["T0O"]
            obj_node_idx = object_index[obs["objects"][i]["label"]]
            cam_obj_edge = PoseEdge(idx_from=cam_idx, idx_to=obj_node_idx)
            cam_obj_edge.add_pose(se3_inverse_numpy(TWC) @ TWO)
            edges.append(copy.deepcopy(cam_obj_edge))
            del cam_obj_edge

        # update TWC_prev, cam_idx_prev
        cam_idx_prev = cam_idx
        TWC_prev = TWC

    return None


def bop_visualize(ds_name="ycbv", split="train_real"):
    dataset = bop.BOPDataset(ds_name, split)

    for idx, (rgb, mask, obs) in enumerate(dataset):
        scene_id = obs["frame_info"]["scene_id"]
        img_cv = cv.cvtColor(rgb.numpy(), cv.COLOR_BGR2RGB)
        cv.imshow(f"Camera in Scene {scene_id}", img_cv)
        cv.waitKey(1)

    cv.destroyAllWindows()
    return None


if __name__ == "__main__":
    print("test")
    # bop_visualize()
    bop_to_g2o()
