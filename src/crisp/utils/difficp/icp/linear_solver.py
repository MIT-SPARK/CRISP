"""Solvers for linear equations"""

import torch


class LinearSolverLU(torch.autograd.Function):
    """
    LU linear solver.
    from https://github.com/DeformableFriends/NeuralTracking/blob/main/model/model.py
    """

    @staticmethod
    def forward(ctx, A, b):
        A_LU, pivots = torch.lu(A)
        x = torch.lu_solve(b, A_LU, pivots)
        ctx.save_for_backward(A_LU, pivots, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_LU, pivots, x = ctx.saved_tensors
        grad_b = torch.lu_solve(grad_x, A_LU, pivots)
        grad_A = -torch.matmul(grad_b, x.view(1, -1))
        return grad_A, grad_b
