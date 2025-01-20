"""ICP unit tests"""

import torch
import unittest
from difficp.icp.correspondence_function import match_points_nn, tensor_dot, correspondence_weighting_and_rejection
from difficp.icp.distance_function import point_to_point_dist, point_to_plane_dist
from difficp.icp.icp import ICP6DoF
from difficp.utils.geometry_utils import transform_points_by_pose, pose_to_matrix
from tqdm import trange

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def tensor(array):
    return torch.tensor(array, dtype=DTYPE, device=DEVICE)


def int_tensor(array):
    return torch.tensor(array, dtype=torch.int64, device=DEVICE)


class TestICP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = tensor(
            [
                [-3, -3, -3],
                [7.35, 7.35, 7.35],
                [0, 0, 0],
                [-1.0, -1.0, -1.0],
                [-2, 2.5, -5],
                [0.1, 1.3, 2.7],
            ]
        )
        self.target = tensor(
            [
                [-3, -3, -3],
                [0, 0, 0],
                [7.35, 7.35, 7.35],
                [-0.4, 2.3, -0.3],
                [11, 11, 11],
                [12, 12, 12],
                [0, 1, 2],
                [-1.5, 2.0, -4.5],
            ]
        )
        self.source_indices = int_tensor([0, 2, 1, 4, 5, 3])
        self.target_indices = int_tensor([0, 1, 2, 7, 3, 6])
        self.source_normals = tensor(
            [
                [0, 0, 1],
                [0, 0, 1],
                [-0.5773503, -0.5773503, -0.5773503],
                [0.5773503, 0.5773503, 0.5773503],
                [-0.5773503, 0.5773503, 0.5773503],
                [0, 0.8944272, 0.4472136],
            ]
        )
        self.target_normals = tensor(
            [
                [0, 0, 1],
                [0.5773503, 0.5773503, 0.5773503],
                [0.7071068, 0.7071068, 0],
                [0, 0.9486833, 0.31622777],
                [1, 0, 0],
                [0, 1, 0],
                [0.5773503, 0.5773503, 0.5773503],
                [-0.5773503, -0.5773503, -0.5773503],
            ]
        )
        self.abs_dists = [0, 0, 0, 1.5, 4.3, 6]
        self.point_dists = tensor([0, 0, 0, 0.75, 10.25, 14])
        self.plane_dists = tensor([0, 0, 0, 0.08333333, 0, 12])
        self.normal_dists = tensor([1, -1, 0, -1 / 3, 0.989949, 1])

    def test_match_points_nn(self):
        # test case 1
        source2 = tensor(
            [
                [-5, -5, -5],
                [-1, -1, -1],
                [0.5, 0.5, 0.5],
                [1.5, 1.5, 1.5],
                [0, 1.5, 0],
                [3, 0.1, 3],
                [-1, -2, -1],
            ]
        )
        target3 = tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                [0.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ]
        )
        output1 = match_points_nn(sources=source2, targets=target3)
        result1 = (
            source2[int_tensor([0, 1, 2, 3, 4, 5, 6])],
            target3[int_tensor([0, 0, 0, 1, 2, 3, 0])],
        )
        self.assertTrue(torch.equal(output1[0], result1[0]))
        self.assertTrue(torch.equal(output1[1], result1[1]))
        # test case 2
        target3 = tensor(
            [
                [0.0, 0.0, 0.0],
                [-3.0, -3.0, -3.0],
                [0.0, -3.0, 0.0],
                [-3.0, 0.0, -3.0],
            ]
        )
        output2 = match_points_nn(sources=source2, targets=target3)
        result2 = (
            source2[int_tensor([0, 1, 2, 3, 4, 5, 6])],
            target3[int_tensor([1, 0, 0, 0, 0, 0, 2])],
        )
        self.assertTrue(torch.equal(output2[0], result2[0]))
        self.assertTrue(torch.equal(output2[0], result2[0]))

    def test_point_to_point_dist(self):
        # test case 1
        output1 = point_to_point_dist(
            sources=self.source[self.source_indices],
            targets=self.target[self.target_indices],
        )
        self.assertTrue(
            torch.allclose(torch.sum(output1.reshape(-1, 3) ** 2, -1), self.point_dists)
        )
        # test case 2
        source_indices2 = int_tensor([5, 3, 1, 2, 4, 0])
        target_indices2 = int_tensor([-2, -3, -7, -8, -4, -6])
        output2 = point_to_point_dist(
            sources=self.source[source_indices2],
            targets=self.target[target_indices2],
        )
        result2 = tensor([0.59, 507, 162.0675, 27, 497.25, 321.3675])
        self.assertTrue(
            torch.allclose(torch.sum(output2.reshape(-1, 3) ** 2, -1), result2)
        )

    def test_point_to_plane_dist(self):
        output = point_to_plane_dist(
            sources=self.source[self.source_indices],
            targets=self.target[self.target_indices],
            target_normals=self.target_normals[self.target_indices],
        )
        self.assertTrue(torch.allclose(output ** 2, self.plane_dists))

    def test_normal_dist(self):
        output = tensor_dot(
            tensor1=self.source_normals[self.source_indices],
            tensor2=self.target_normals[self.target_indices],
        )
        self.assertTrue(torch.allclose(output, self.normal_dists))

    def test_correspondence_rejection(self):
        rejection_dists = [0.1, 4.2, None]
        rejection_normals = [0.99, 0.8, None]
        results = [
            [
                (int_tensor([0]), int_tensor([0])),
                (int_tensor([0]), int_tensor([0])),
                (int_tensor([0, 2, 1]), int_tensor([0, 1, 2])),
            ],
            [
                (int_tensor([0]), int_tensor([0])),
                (int_tensor([0]), int_tensor([0])),
                (int_tensor([0, 2, 1, 4]), int_tensor([0, 1, 2, 7])),
            ],
            [
                (int_tensor([0, 3]), int_tensor([0, 6])),
                (int_tensor([0, 5, 3]), int_tensor([0, 3, 6])),
                (
                    int_tensor([0, 2, 1, 4, 5, 3]),
                    int_tensor([0, 1, 2, 7, 3, 6]),
                ),
            ],
        ]
        for i, rejection_dist in enumerate(rejection_dists):
            for j, rejection_normal in enumerate(rejection_normals):
                output = correspondence_weighting_and_rejection(
                    sources=self.source[self.source_indices],
                    targets=self.target[self.target_indices],
                    source_normals=self.source_normals[self.source_indices],
                    target_normals=self.target_normals[self.target_indices],
                    rejection_dist=rejection_dist,
                    rejection_normal=rejection_normal,
                )
                source_inds = results[i][j][0]
                target_inds = results[i][j][1]
                self.assertTrue(torch.equal(output[0], self.source[source_inds]))
                self.assertTrue(torch.equal(output[1], self.target[target_inds]))
                self.assertTrue(
                    torch.equal(output[2], self.source_normals[source_inds])
                )
                self.assertTrue(
                    torch.equal(output[3], self.target_normals[target_inds])
                )

    def _test_icp_solver(self, solver_type, atol=2e-6):
        for i in trange(100):
            icp = ICP6DoF(solver_type=solver_type, differentiable=False)
            random_pose = torch.rand(6, dtype=DTYPE, device=DEVICE)
            target = transform_points_by_pose(self.source, random_pose)
            output = icp.solver(
                pose=torch.eye(4, dtype=torch.float32, device=DEVICE),
                sources=self.source,
                targets=target,
            )
            self.assertTrue(
                torch.allclose(pose_to_matrix(random_pose), output, atol=atol, rtol=0)
            )

    def test_icp_solver_svd(self):
        self._test_icp_solver("svd", 2e-6)

    def test_icp_solver_lm(self):
        self._test_icp_solver("lm", 2e-6)

    def _test_icp_call(self, solver_type, atol=2e-6, threshold=90):
        icp = ICP6DoF(
            iters_max=100,
            mse_threshold=1e-5,
            corr_threshold=10,
            solver_type=solver_type,
            corr_type="nn",
            dist_type="point",
            differentiable=False,
        )
        close = 0
        for i in trange(100):
            random_pose = torch.rand(6, dtype=DTYPE, device=DEVICE)
            target = transform_points_by_pose(self.source, random_pose)
            output = icp(self.source, target)[0]
            if torch.allclose(pose_to_matrix(random_pose), output, atol=atol, rtol=0):
                close += 1
        print(close)
        self.assertGreater(close, threshold)

    def _test_differentiable(self, solver_type):
        torch.autograd.set_detect_anomaly(True)
        icp = ICP6DoF(
            iters_max=10,
            mse_threshold=-1,  # dont stop early
            corr_threshold=-1,  # dont stop early
            solver_type=solver_type,
            corr_type="nn",
            dist_type="point",
            differentiable=True,
        )
        random_pose = torch.rand(6, dtype=DTYPE, device=DEVICE)
        source = self.source.detach()
        target = transform_points_by_pose(self.source, random_pose).detach()
        zero_pose = torch.zeros(6, dtype=DTYPE, device=DEVICE)
        source.requires_grad_(True)
        target.requires_grad_(True)
        zero_pose.requires_grad_(True)
        output, iters, _ = icp(
            source,
            target,
            init_pose=zero_pose,
        )
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, random_pose)
        loss.backward()
        # print(source.grad, iters)
        self.assertIsNotNone(source.grad)
        self.assertFalse(torch.allclose(source.grad, torch.zeros_like(source.grad)))
        self.assertIsNotNone(target.grad)
        self.assertFalse(torch.allclose(target.grad, torch.zeros_like(target.grad)))
        self.assertIsNotNone(zero_pose.grad)
        self.assertFalse(
            torch.allclose(zero_pose.grad, torch.zeros_like(zero_pose.grad))
        )
        self.assertEqual(iters, 10)

    def test_icp_call_svd(self):
        self._test_icp_call("svd", 2e-6, 90)
        # self._test_differentiable("svd")

    def test_icp_call_lm(self):
        self._test_icp_call("lm", 2e-6, 90)
        # self._test_differentiable("lm")


if __name__ == "__main__":
    unittest.main()
