import pytest
import numpy as np
import torch
import torch.nn as nn


def test_inf_norm_lipschitz_constraint():
    """Test inf norm lipschitz constraint"""
    from crisp.models.joint import inf_norm_lipschitz_constraint

    # Dense/linear layer
    m = nn.Linear(20, 30)
    # weight shape: (30, 20)
    weight = m.weight
    output = inf_norm_lipschitz_constraint(weight, 1.0)
    assert output.shape[0] == weight.shape[0]
    assert output.shape[1] == weight.shape[1]

    # Conv2D layer
    m = nn.Conv2d(20, 30, 3)
    # weight shape: (30, 20, 3, 3)
    weight = m.weight
    output = inf_norm_lipschitz_constraint(weight, 1.0)
    assert output.shape[0] == weight.shape[0]
    assert output.shape[1] == weight.shape[1]


def test_one_norm_lipschitz_constraint():
    """Test one norm lipschitz constraint"""
    from crisp.models.joint import one_norm_lipschitz_constraint

    # Dense/linear layer
    m = nn.Linear(20, 30)
    # weight shape: (30, 20)
    weight = m.weight
    output = one_norm_lipschitz_constraint(weight, 1.0)
    assert output.shape[0] == weight.shape[0]
    assert output.shape[1] == weight.shape[1]

    # Conv2D layer
    m = nn.Conv2d(20, 30, 3)
    # weight shape: (30, 20, 3, 3)
    weight = m.weight
    output = one_norm_lipschitz_constraint(weight, 1.0)
    assert output.shape[0] == weight.shape[0]
    assert output.shape[1] == weight.shape[1]


def test_two_norm_linear_lipschitz_constraint():
    """Test two norm lipschitz constraint"""
    from crisp.models.joint import two_norm_linear_lipschitz_constraint

    # Dense/linear layer
    m = nn.Linear(20, 30)
    # weight shape: (30, 20)
    weight = m.weight
    output = two_norm_linear_lipschitz_constraint(weight, 1.0)
    assert output.shape[0] == weight.shape[0]
    assert output.shape[1] == weight.shape[1]
