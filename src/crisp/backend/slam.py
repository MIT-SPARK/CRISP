import gtsam
import time
import numpy as np
from scipy.spatial.transform import Rotation
from crisp.utils.math import se3_inverse_numpy


def are_equal_symbols(symbol_1, symbol_2):
    """Helper function to compare two gtsam symbols"""
    return symbol_1.equals(symbol_2, 1e-7)


def get_key_chr(key):
    """Get the character prefix of the gtsam symbol"""
    return chr(gtsam.Symbol(key).chr())


def poses2odom(T, index_mapping_fn=lambda x: x):
    """Convert a torch tensor of world_T_cam poses to a list of odometry measurements.

    Parameters
    ----------
    T
    index_mapping_fn: function mapping the index of the pose in the batched tensor to the index of the pose node in the
    factor graph.
    """
    B = T.shape[0]
    odom = []
    for i in range(1, B):
        world_T_cam_prev = T[i - 1, ...].squeeze().numpy(force=True).astype(np.float64)
        world_T_cam_i = T[i, ...].squeeze().numpy(force=True).astype(np.float64)
        prev_T_cur = se3_inverse_numpy(world_T_cam_prev) @ world_T_cam_i
        odom.append((index_mapping_fn(i - 1), index_mapping_fn(i), prev_T_cur))
    return odom


class ObjectPGOSolver:
    """Simple PGO solver based on gtsam"""

    def __init__(
        self,
        odom_noise_variance,
        obj_noise_variance,
        prior_noise_variance,
        odom_as_inliers=True,
        obj_meas_as_outliers=False,
    ):
        self.nodes = []
        self.edges = []
        self.graph = gtsam.NonlinearFactorGraph()

        # object indexing (symbols used in gtsam graph)
        # symbol prefix is to distinguish between object nodes and camera nodes (camera nodes are indexed by integers)
        self.object_index = {}
        self.object_symbol_prefix = "l"
        self.odom_symbol_prefix = "x"
        self.obj_symbol = lambda j: gtsam.Symbol(self.object_symbol_prefix, j)
        self.odom_symbol = lambda j: gtsam.Symbol(self.odom_symbol_prefix, j)
        self.current_available_idx = 0

        # optimization parameters
        self.odom_as_inliers = odom_as_inliers
        self.obj_meas_as_outliers = obj_meas_as_outliers

        # noise models
        self.odom_prior_model = gtsam.noiseModel.Diagonal.Variances(np.array(prior_noise_variance, dtype=np.float64))
        self.odom_model = gtsam.noiseModel.Diagonal.Variances(np.array(odom_noise_variance, dtype=np.float64))
        self.object_pose_model = gtsam.noiseModel.Diagonal.Variances(np.array(obj_noise_variance, dtype=np.float64))

    def add_obj_pose(self, frame, obj_label, T_cam_obj):
        """Add object pose measurement.
        Will check the object label and add a new node if it is not in the object index.
        """
        # create or load the gtsam key for the object node
        if obj_label not in self.object_index.keys():
            self.object_index[obj_label] = self.obj_symbol(self.current_available_idx)
            self.current_available_idx += 1
        self.graph.add(
            gtsam.BetweenFactorPose3(
                self.odom_symbol(frame).key(),
                self.object_index[obj_label].key(),
                gtsam.gtsam.Pose3(T_cam_obj),
                self.object_pose_model,
            )
        )

    def add_odom(self, frame_i, frame_j, T_ij):
        """Add odometry measurement between frame_i and frame_j.
        Keys in the factor graph are the frame index.
        """
        self.graph.add(
            gtsam.BetweenFactorPose3(
                self.odom_symbol(frame_i).key(),
                self.odom_symbol(frame_j).key(),
                gtsam.gtsam.Pose3(T_ij),
                self.odom_model,
            )
        )

    def add_first_odom_prior(self, pose=None):
        """Add a prior on the first odometry node"""
        if pose is None:
            self.graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), self.odom_prior_model))
        else:
            self.graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(pose), self.odom_prior_model))

    def get_odometry_edges(self):
        """Return indices of edges representing odometry measurements"""
        return [
            x
            for x in range(self.graph.size())
            if get_key_chr(self.graph.at(x).keys()[0]) != self.object_symbol_prefix
            and get_key_chr(self.graph.at(x).keys()[1]) != self.object_symbol_prefix
        ]

    def get_obj_pose_edges(self):
        """Return indices of edges representing object pose measurements"""
        return [
            x
            for x in range(self.graph.size())
            if get_key_chr(self.graph.at(x).keys()[0]) == self.object_symbol_prefix
            or get_key_chr(self.graph.at(x).keys()[1]) == self.object_symbol_prefix
        ]

    def get_inlier_error(self, weights, values):
        """Calculate factor graph errors with consideration of weights"""
        total_err = 0
        for i, w in enumerate(list(weights)):
            factor = self.graph.at(i)
            total_err += w * factor.error(values)
        return total_err

    def solve(self, initials=None):
        if initials is None:
            initials = gtsam.InitializePose3.initialize(self.graph)

        params = gtsam.LevenbergMarquardtParams()
        # gnc optimizer
        gnc_params = gtsam.GncLMParams(params)

        # the optimizer weights are on the edges
        # set odometry to be inliers
        if self.odom_as_inliers:
            print("Setting all odometry as inliers.")
            gnc_params.setKnownInliers(self.get_odometry_edges())

        # set object edges to be outliers
        if self.obj_meas_as_outliers:
            print("Setting all object measurements as outliers.")
            gnc_params.setKnownOutliers(self.get_obj_pose_edges())

        # gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.VALUES)
        # optimizer.setInlierCostThresholds(10)
        optimizer = gtsam.GncLMOptimizer(self.graph, initials, gnc_params)

        start = time.time()
        result = optimizer.optimize()
        end = time.time()
        compute_time = end - start

        print("Optimization complete")
        print(f"PGO Loss (input): {self.graph.error(initials):.2f}")
        print(f"PGO Loss (output): {self.graph.error(result):.2f}")
        print(f"PGO Inlier Loss (output): {self.get_inlier_error(optimizer.getWeights(), initials):.2f}")
        print(f"PGO Inlier Loss (output): {self.get_inlier_error(optimizer.getWeights(), result):.2f}")
        print(
            f"PGO Inlier Counts | Ratio: {int(np.sum(optimizer.getWeights()))} "
            f"{int(np.sum(optimizer.getWeights())) / len(optimizer.getWeights()):.2f}"
        )
        print(f"PGO Compute Time: {1000 * compute_time:.2f} ms")

        return result


class PoseElement:
    def __init__(self, t=None, R=None, noise_scale=0.01):
        if t is None:
            self.t = []
        if R is None:
            self.R = []
        self.q = []  # QUAT representation
        self.noise_scale = noise_scale
        self.ready = False  # This flag will read False till average() is executed

    def add_translation(self, t):
        self.t.append(t)

    def add_rotation(self, R):
        self.R.append(R)
        self.update_q()

    def add_pose(self, T):
        self.R.append(T[:3, :3])
        self.t.append(T[:3, 3:])
        self.update_q()

    def average(self):
        if self.t == [] or self.R == [] or self.q == []:
            raise ValueError("t, R, q are unassigned to print the g2o vector")

        if len(self.t) == 1:
            self.t = [self.t[0]]
            self.R = [self.R[0]]
            self.q = [self.q[0]]
        else:
            # averaging the translation
            t = np.zeros_like(self.t[0])
            for s in self.t:
                t += s
            t = t / len(self.t)
            self.t = [t]

            # averaging the rotation
            R = np.zeros_like(self.R[0])
            for S in self.R:
                R += S
            R = R / len(self.R)
            self.R = [R]

            # quat
            self.q = []
            self.update_q()

        self.ready = True

    def get_g2o_vector(self):
        """"""
        raise NotImplementedError

    def add_noise(self, noise_scale):
        self.average()
        self.t = [self.t[0] + np.random.normal(0.0, noise_scale, (3, 1))]
        self.R = [self.rotation_noise(noise_scale) @ self.R[0]]
        self.update_q()

    def update_q(self):
        _scipy_R = Rotation.from_matrix(self.R[-1])
        self.q.append(_scipy_R.as_quat())

    def rotation_noise(self, noise_scale):
        phi = np.random.normal(0.0, noise_scale, 3)
        phi = phi.reshape((3, 1))
        phi_abs = np.linalg.norm(phi, ord=2)

        phi_n = phi / phi_abs
        phi_cross = np.zeros((3, 3))
        phi_cross[0, 1] = -phi_n[2]
        phi_cross[0, 2] = phi_n[1]
        phi_cross[1, 2] = -phi_n[0]
        phi_cross = phi_cross.transpose() + phi_cross

        R_noise = np.cos(phi_abs) * np.eye(3)
        R_noise += np.sin(phi_abs) * phi_cross
        R_noise += (1 - np.cos(phi_abs)) * phi_n @ phi_n.transpose()

        return R_noise

    def reset(self):
        self.t = [np.zeros((3, 1))]
        self.R = [np.eye(3)]
        self.q = []
        self.update_q()


class PoseNode(PoseElement):
    def __init__(self, idx, t=None, R=None):
        super().__init__(t, R)
        self.idx = [idx]

    def get_g2o_vector(self):
        if not self.ready:
            raise ValueError("PoseElement is not ready to generate g2o vector. Execute average() first.")

        # fmt: off
        return (
                self.idx
                + self.t[0].reshape(3, ).tolist()
                + self.q[0].reshape(4, ).tolist()
        )
        # fmt: on

    def to_g2o_entry_list(self):
        """return a list ready to be written as g2o line"""
        return ["VERTEX_SE3:QUAT"] + self.get_g2o_vector()


class PoseEdge(PoseElement):
    def __init__(self, idx_from, idx_to, t=None, R=None):
        super().__init__(t, R)
        self.idx_from = [idx_from]
        self.idx_to = [idx_to]
        # fmt: off
        self.info_mat = [
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0,
            400.0, 0.0, 0.0,
            400.0, 0.0,
            400.0,
        ]
        # fmt: on

    def get_g2o_vector(self):
        if not self.ready:
            raise ValueError("PoseElement is not ready to generate g2o vector. Execute average() first.")

        # fmt: off
        return (
                self.idx_from
                + self.idx_to
                + self.t[0].reshape(3, ).tolist()
                + self.q[0].reshape(4, ).tolist()
                + self.info_mat
        )
        # fmt: on

    def to_g2o_entry_list(self):
        """return a list ready to be written as g2o line"""
        return ["EDGE_SE3:QUAT"] + self.get_g2o_vector()
