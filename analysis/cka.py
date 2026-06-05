"""Centered kernel alignment for representation comparison."""

from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute linear CKA between two activation matrices.

    Args:
        x: Tensor shaped ``[num_examples, feature_dim_x]``.
        y: Tensor shaped ``[num_examples, feature_dim_y]``.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects 2D activation matrices")
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear_cka expects the same number of examples")

    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cross_covariance = x.T @ y
    x_covariance = x.T @ x
    y_covariance = y.T @ y

    numerator = torch.linalg.matrix_norm(cross_covariance, ord="fro") ** 2
    denominator = torch.linalg.matrix_norm(x_covariance, ord="fro") * torch.linalg.matrix_norm(
        y_covariance,
        ord="fro",
    )
    if float(denominator) == 0.0:
        return 0.0
    return float((numerator / denominator).clamp(min=0.0, max=1.0))

