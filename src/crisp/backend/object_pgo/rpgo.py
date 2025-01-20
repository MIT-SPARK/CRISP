import abc
from abc import ABC
import os
import numpy as np
import gtsam
import time
import matplotlib.pyplot as plt
import pyvista as pv
from crisp.backend import bop_to_g2o
from scipy.spatial.transform import Rotation

"""
    helper functions 
"""


def g2o_pose_to_gtsam_pose(g2o_pose):
    t = np.asarray(g2o_pose)[:3].reshape(3, 1)
    q = np.asarray(g2o_pose)[3:]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array"""
    return np.array([x, y, z, a, b, c], dtype=float)


def info2mat(info):
    mat = np.zeros((6, 6))
    ix = 0
    for i in range(mat.shape[0]):
        mat[i, i:] = info[ix : ix + (6 - i)]
        mat[i:, i] = info[ix : ix + (6 - i)]
        ix += 6 - i

    return mat


def read_g2o(fn):
    verticies, edges = [], []
    with open(fn) as f:
        for line in f:
            line = line.split()
            if line[0] == "VERTEX_SE3:QUAT":
                v = int(line[1])
                pose = np.array(line[2:], dtype=np.float32)
                verticies.append([v, pose])

            elif line[0] == "EDGE_SE3:QUAT":
                u = int(line[1])
                v = int(line[2])
                pose = np.array(line[3:10], dtype=np.float32)
                info = np.array(line[10:], dtype=np.float32)

                info = info2mat(info)
                edges.append([u, v, pose, info, line])

    return verticies, edges


def write_g2o(pose_graph, fn):
    import csv

    verticies, edges = pose_graph
    with open(fn, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for v, pose in verticies:
            row = ["VERTEX_SE3:QUAT", v] + pose.tolist()
            writer.writerow(row)
        for edge in edges:
            writer.writerow(edge[-1])


# def write_xyz(pose_graph, fn):
#     import csv
#     verticies, edges = pose_graph
#     with open(fn, 'w') as f:
#         writer = csv.writer(f, delimiter=' ')
#         for (v, pose) in verticies:
#             row = pose.tolist()
#             writer.writerow(row)


def plot_and_save(points, pngname, title="", axlim=None):
    # points = points.detach().cpu().numpy()
    plt.figure(figsize=(7, 7))
    ax = plt.axes(projection="3d")
    ax.plot3D(points[:, 0], points[:, 1], points[:, 2], "b")
    plt.title(title)
    if axlim is not None:
        ax.set_xlim(axlim[0])
        ax.set_ylim(axlim[1])
        ax.set_zlim(axlim[2])
    plt.savefig(pngname)
    print("Saving PGO plot: ", pngname)
    return ax.get_xlim(), ax.get_ylim(), ax.get_zlim()


def consecutive(node_idx_to, node_idx_from):
    return abs(node_idx_to - node_idx_from) == 1


"""
PGO
"""


class PGOBase(ABC):
    def __init__(self, in_file, out_file, plot_folder="./"):
        self.in_file = in_file
        self.out_file = out_file
        self.plot_folder = plot_folder

    def plot_input_graph(self):
        self.plot_graph_from_file(self.in_file, self.plot_folder + "pgo_input.png")

    def plot_output_graph(self):
        self.plot_graph_from_file(self.out_file, self.plot_folder + "pgo_output.png")

    def plot_graph_from_file(self, _file, _file_name):
        # plotting
        _graph = read_g2o(_file)
        _nodes = []
        for idx, _node in enumerate(_graph[0]):
            _nodes.append(_graph[0][idx][1][:3].reshape(1, 3))
        _nodes = np.concatenate(_nodes, axis=0)
        _edges = [_graph[1][i][:2] for i in range(len(_graph[1]))]

        self._nodes = _nodes
        self._edges = _edges
        plot_and_save(_nodes, _file_name, _file_name.split("/")[-1][:-4])
        self.visualize()

    def visualize(self):
        pv.set_plot_theme("document")
        pl = pv.Plotter()

        # adding points
        pl.add_points(self._nodes, render_points_as_spheres=True, point_size=4.0)

        # adding lines
        lines = []
        for _e in self._edges:
            lines.append(self._nodes[_e[0]])
            lines.append(self._nodes[_e[1]])

        lines = np.asarray(lines)
        pl.add_lines(lines, color="brown", width=1.0)

        pl.add_floor()
        # pl.add_axes_at_origin(xlabel=None, ylabel=None, zlabel=None)

        pl.show()

    @abc.abstractmethod
    def solve(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_gtsam_graph(self):
        raise NotImplementedError

    def save_gtsam_graph(self):
        save_graph = gtsam.NonlinearFactorGraph()
        for factor_idx in range(self.graph.size() - 1):
            factor = self.graph.at(factor_idx)

            node_idx_from = factor.keys()[0]
            node_idx_to = factor.keys()[1]

            measurement = factor.measured()

            noise_model = self.odomModel
            # because gtsam.writeG2o does not support to save robust factor,
            # so this is a just, naive solution to save the node values.

            save_graph.add(gtsam.BetweenFactorPose3(node_idx_from, node_idx_to, measurement, noise_model))

        print("Writing Pose Graph: ", self.out_file)
        gtsam.writeG2o(save_graph, self.result, self.out_file)


class PGO(PGOBase):
    def __init__(self, in_file, out_file, plot_folder="./", gnc=True, init=True, plot=True):
        super().__init__(in_file, out_file, plot_folder=plot_folder)

        # constants (user variables)
        self.priorModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
        self.odomModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1))
        self.is3D = True
        self.gnc = gnc
        self.init = init
        self.plot = plot

        # gtsam graph
        self.graph, self.initial = None, None
        self.result = None

        # plotting the input
        if self.plot:
            self.plot_input_graph()
        self._nodes = None
        self._edges = None

        # timing
        self.compute_time = None

    def solve(self):
        # initialize with chordal method
        if self.graph is None:
            self.graph, self.initial = self.get_gtsam_graph()

        if self.init:
            self.initial = gtsam.InitializePose3.initialize(self.graph)

        # optimizer
        params = gtsam.LevenbergMarquardtParams()
        # params.setVerbosity("ERROR")  # this will show info about stopping conds
        if self.gnc:
            gnc_params = gtsam.GncLMParams(params)
            # gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.VALUES)
            optimizer = gtsam.GncLMOptimizer(self.graph, self.initial, gnc_params)
        else:
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)

        # run opt
        start = time.time()
        self.result = optimizer.optimize()
        end = time.time()
        self.compute_time = end - start

        # printing, saving, and output
        print("Optimization complete")
        print(f"PGO Loss (input): {self.graph.error(self.initial):.2f}")
        print(f"PGO Loss (output): {self.graph.error(self.result):.2f}")
        print(f"PGO Compute Time: {1000 * self.compute_time:.2f} ms")

        self.save_gtsam_graph()
        if self.plot:
            self.plot_output_graph()

    def get_gtsam_graph(self):
        print("Reading Pose Graph: ", self.in_file)
        initial_graph, initial = gtsam.readG2o(self.in_file, self.is3D)
        graph = gtsam.NonlinearFactorGraph()
        for factor_idx in range(initial_graph.size()):
            factor = initial_graph.at(factor_idx)

            node_idx_from = factor.keys()[0]
            node_idx_to = factor.keys()[1]

            measurement = factor.measured()
            noise_model = self.odomModel

            graph.add(gtsam.BetweenFactorPose3(node_idx_from, node_idx_to, measurement, noise_model))

        graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), self.priorModel))

        return graph, initial


class ObjectPGO(PGOBase):
    def __init__(
        self,
        in_file,
        out_file,
        object_index_file,
        obj_pose_meas_variance=None,
        plot_folder="./",
        gnc=True,
        init=True,
        plot=True,
        odom_as_inliers=True,
        obj_meas_as_outliers=False,
        inlier_cost_thres=0.7,
        first_odom_node_prior=None,
    ):
        super().__init__(in_file, out_file, plot_folder=plot_folder)

        # constants (user variables)
        self.priorModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
        self.odomModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1))
        self.is3D = True
        self.gnc = gnc
        self.init = init
        self.plot = plot
        self.odom_as_inliers = odom_as_inliers
        self.obj_meas_as_outliers = obj_meas_as_outliers
        self.inlier_cost_thres = inlier_cost_thres

        # gtsam graph
        self.graph, self.initial = None, None
        assert first_odom_node_prior.shape[0] == first_odom_node_prior.shape[1] == 4
        self.first_odom_node_prior = gtsam.Pose3(first_odom_node_prior)
        self.result = None

        # plotting the input
        if self.plot:
            self.plot_input_graph()
        self._nodes = None
        self._edges = None

        # timing
        self.compute_time = None

        self.object_index_file = object_index_file
        self.object_index = bop_to_g2o.read_object_index(object_index_file)

        # noise model for the object pose measurements
        self.objPoseModel = gtsam.noiseModel.Diagonal.Variances(obj_pose_meas_variance)

    def solve(self):
        # initialize with chordal method
        if self.graph is None:
            print("Creating factor graph.")
            self.graph, self.initial = self.get_gtsam_graph(first_odom_node_prior=self.first_odom_node_prior)

        if self.init:
            self.initial = gtsam.InitializePose3.initialize(self.graph)

        # optimizer
        params = gtsam.LevenbergMarquardtParams()
        # params.setVerbosity("ERROR")  # this will show info about stopping conds
        if self.gnc:
            gnc_params = gtsam.GncLMParams(params)
            # the optimizer weights are on the edges
            # set odometry to be inliers
            if self.odom_as_inliers:
                print("Setting all odometry as inliers.")
                known_inliers = [
                    x for x in range(self.graph.size()) if self.graph.at(x).keys()[-1] not in self.object_index.values()
                ]
                gnc_params.setKnownInliers(known_inliers)

            # set object edges to be outliers
            if self.obj_meas_as_outliers:
                print("Setting all object measurements as outliers.")
                known_outliers = [
                    x for x in range(self.graph.size()) if self.graph.at(x).keys()[-1] in self.object_index.values()
                ]
                gnc_params.setKnownOutliers(known_outliers)

            # gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.VALUES)
            # optimizer.setInlierCostThresholds(10)
            optimizer = gtsam.GncLMOptimizer(self.graph, self.initial, gnc_params)
            optimizer.setInlierCostThresholdsAtProbability(self.inlier_cost_thres)
        else:
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)

        # run opt
        start = time.time()
        self.result = optimizer.optimize()
        end = time.time()
        self.compute_time = end - start

        # printing, saving, and output
        print("Optimization complete")
        print(f"PGO Loss (input): {self.graph.error(self.initial):.2f}")
        print(f"PGO Loss (output): {self.graph.error(self.result):.2f}")

        print(f"PGO Inlier Loss (output): {self.get_inlier_error(optimizer.getWeights(), self.initial):.2f}")
        print(f"PGO Inlier Loss (output): {self.get_inlier_error(optimizer.getWeights(), self.result):.2f}")
        print(
            f"PGO Inlier Counts | Ratio: {int(np.sum(optimizer.getWeights()))} "
            f"{int(np.sum(optimizer.getWeights())) / len(optimizer.getWeights()):.2f}"
        )

        print(f"PGO Compute Time: {1000 * self.compute_time:.2f} ms")

        self.save_gtsam_graph()
        if self.plot:
            self.plot_output_graph()

    def get_inlier_error(self, weights, values):
        total_err = 0
        for i, w in enumerate(list(weights)):
            factor = self.graph.at(i)
            total_err += w * factor.error(values)
        return total_err

    def is_in_object_index(self, idx):
        return idx in self.object_index.values()

    def get_gtsam_graph(self, first_odom_node_prior=None):
        print("Reading Pose Graph: ", self.in_file)
        initial_graph, initial = gtsam.readG2o(self.in_file, self.is3D)
        graph = gtsam.NonlinearFactorGraph()
        for factor_idx in range(initial_graph.size()):
            factor = initial_graph.at(factor_idx)

            node_idx_from = factor.keys()[0]
            node_idx_to = factor.keys()[1]

            if self.is_in_object_index(node_idx_to):
                noise_model = self.objPoseModel
            else:
                noise_model = self.odomModel

            measurement = factor.measured()

            graph.add(gtsam.BetweenFactorPose3(node_idx_from, node_idx_to, measurement, noise_model))

        if first_odom_node_prior is None:
            graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), self.priorModel))
        else:
            graph.add(gtsam.PriorFactorPose3(0, first_odom_node_prior, self.priorModel))

        # graph.keys() are the nodes' ids
        # size of the graph is the number of edges
        # initial are the initial values for the nodes
        return graph, initial

    def plot_input_object_pose_graph(self, object_index, shape_models):
        pl = self.plot_object_pose_graph(self.in_file, object_index, shape_models)
        pl.show()

    def plot_output_object_pose_graph(self, object_index, shape_models):
        pl = self.plot_object_pose_graph(self.out_file, object_index, shape_models)
        pl.show()

    def plot_object_pose_graph(self, g2o_file, object_index, shape_models, plotter=None):
        """
        :param object_index: dictionary assigning object_labels to object index
        :param shape_models: dictionary assigning object_labels to CAD model mesh
        :return: pyvisa visualization of camera locations and posed CAD mesh models
        """

        _graph = read_g2o(g2o_file)
        _nodes = []
        for idx, _node in enumerate(_graph[0]):
            _nodes.append(_graph[0][idx][1].reshape(1, -1))
        _nodes = np.concatenate(_nodes, axis=0)
        _edges = [_graph[1][i][:2] for i in range(len(_graph[1]))]
        self._nodes = _nodes
        self._edges = _edges

        pv.set_plot_theme("document")
        if plotter is None:
            pl = pv.Plotter()
        else:
            pl = plotter

        # adding points
        pl.add_points(self._nodes[:, :3], render_points_as_spheres=True, point_size=4.0)

        # # adding lines
        # lines = []
        # for _e in self._edges:
        #     lines.append(self._nodes[_e[0], :3])
        #     lines.append(self._nodes[_e[1], :3])
        #
        # lines = np.asarray(lines)
        # pl.add_lines(lines, color='brown', width=1.0)

        for label, idx in object_index.items():
            # get object pose
            g2o_vec = _nodes[idx, :]
            t = g2o_vec[:3].reshape(3, 1)
            q = g2o_vec[3:]
            _R = Rotation.from_quat(q)
            R = _R.as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3:] = t

            # get object cad model
            reader_ = pv.get_reader(shape_models[label])
            mesh = reader_.read()
            mesh = mesh.scale([0.001, 0.001, 0.001])  # cad model scale is mm, pose scale is in m
            mesh = mesh.transform(T)

            # add posed cad model to pl
            pl.add_mesh(mesh)
            del mesh

        # pl.add_floor()
        return pl
