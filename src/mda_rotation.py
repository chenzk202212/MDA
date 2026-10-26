"""Shared feature rotation, Procrustes targets, and adaptation checkpoints."""

from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def proper_polar(matrix):
    """Closest proper orthogonal matrix (no reflection)."""
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    signs = torch.ones(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    signs[-1] = torch.linalg.det(u @ vh).sign()
    return (u * signs.unsqueeze(0)) @ vh


@torch.no_grad()
def optimal_rotation(source, target):
    """Solve min_R ||source @ R.T - target||_F with R in SO(d)."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("Procrustes inputs must have the same nonempty [N, D] shape")
    return proper_polar(target.T @ source)


class TaskRotations(nn.Module):
    def __init__(self, names, dimension, device=None, dtype=None):
        super().__init__()
        self.names = tuple(names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("Rotation task names must be unique")
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.eye(dimension, device=device, dtype=dtype))
            for _ in self.names
        ])

    def matrix_for(self, name):
        # Unseen tasks have no trained task-specific rotation.
        return self.matrices[self.names.index(name)] if name in self.names else None

    def forward(self, features, name):
        matrix = self.matrix_for(name)
        return features if matrix is None else F.linear(features, matrix)

    @torch.no_grad()
    def project_(self):
        for matrix in self.matrices:
            matrix.copy_(proper_polar(matrix))


def generate_etf(d, c):
    """Unit class targets of shape [C,D], including when D exceeds C.

    A simplex is exact for D >= C-1; below that we use its normalized truncated
    embedding, which is an approximation rather than an exact simplex ETF.
    """
    if d < 1 or c < 2:
        raise ValueError("Feature dimension must be positive and class count >= 2")
    center = torch.eye(c) - torch.ones(c, c) / c
    u, _, _ = torch.linalg.svd(center, full_matrices=False)
    width = min(d, c - 1)
    targets = F.pad(u[:, :width], (0, d - width))
    return F.normalize(targets, dim=1)


def soft_assignments(features, targets, temperature=1.0):
    """Task-local cosine assignments; no sample labels are needed."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Assignment temperature must be finite and positive")
    return (F.normalize(features, dim=1) @ targets.T / temperature).softmax(dim=1)


def alignment_loss(features, assignments, targets):
    """Expected squared distance under differentiable soft assignments."""
    normalized = F.normalize(features, dim=1)
    distances = (normalized.square().sum(1, keepdim=True)
                 + targets.square().sum(1).unsqueeze(0)
                 - 2 * normalized @ targets.T)
    return (assignments * distances).sum(1).mean()


def procrustes_penalty(raw_features, assignments, targets, matrix):
    # Use this task's UNROTATED features. A fit from already rotated features
    # would describe an incremental correction, not an absolute target for R.
    with torch.no_grad():
        normalized = F.normalize(raw_features.detach(), dim=1)
        weights = assignments.detach()
        mass = weights.sum(0)
        present = mass > torch.finfo(weights.dtype).tiny
        means = weights[:, present].T @ normalized / mass[present, None]
        reference = optimal_rotation(means, targets[present])
    return (matrix - reference).square().sum()


def save_alignment_checkpoint(path, lambdas, rotations, optimizer, epoch, targets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adaptation_method": "unlabeled_soft_assignment",
        "epoch": epoch,
        "lambdas": lambdas.detach().cpu(),
        "rotation_names": list(rotations.names),
        "rotations": {key: value.detach().cpu() for key, value in rotations.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "targets": targets.detach().cpu(),
    }, path)


def load_alignment_checkpoint(path, lambdas, rotations, optimizer, targets):
    state = torch.load(path, map_location=lambdas.device, weights_only=True)
    required = {"epoch", "lambdas", "rotation_names", "rotations", "optimizer", "targets"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Expected an alignment.pt checkpoint including rotations and optimizer state")
    if state.get("adaptation_method") != "unlabeled_soft_assignment":
        raise ValueError("Resume requires an unlabeled soft-assignment checkpoint; start a new run")
    if tuple(state["rotation_names"]) != rotations.names:
        raise ValueError("Checkpoint task names/order do not match the experiment")
    if state["lambdas"].shape != lambdas.shape or state["targets"].shape != targets.shape:
        raise ValueError("Checkpoint coefficient/target dimensions do not match the experiment")
    with torch.no_grad():
        lambdas.copy_(state["lambdas"])
        targets.copy_(state["targets"])
    rotations.load_state_dict(state["rotations"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"])
