"""Training loop for the baseline model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from less_geometry_same_semantics.training.targets import encode_targets


def compute_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    relation_loss_weight: float = 0.5,
) -> torch.Tensor:
    """Compute the multi-task semantic loss."""

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    object_loss = bce(logits["object_logits"], targets["object_targets"])
    attribute_loss = bce(logits["attribute_logits"], targets["attribute_targets"])
    relation_loss = bce(logits["relation_logits"], targets["relation_targets"])
    scene_loss = ce(logits["scene_logits"], targets["scene_targets"])
    return object_loss + attribute_loss + scene_loss + relation_loss_weight * relation_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    relation_loss_weight: float = 0.5,
) -> dict[str, float]:
    """Train for one epoch and return aggregate loss stats."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        points = batch["points"].to(device)
        mask = batch["mask"].to(device)
        targets = encode_targets(batch["targets"], device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(points, mask)
        loss = compute_loss(logits, targets, relation_loss_weight=relation_loss_weight)
        loss.backward()
        optimizer.step()

        batch_size = points.shape[0]
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size

    return {"loss": total_loss / max(1, total_samples)}
