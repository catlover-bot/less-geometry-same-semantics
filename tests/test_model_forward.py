from __future__ import annotations

import torch

from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.schemas.schema import is_valid_semantic_output


def test_minimal_end_to_end_forward_pass() -> None:
    model = PointSemanticsModel(
        encoder_hidden_dim=16,
        token_dim=24,
        compressed_tokens=4,
        decoder_hidden_dim=32,
    )
    points = torch.randn(2, 32, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)

    logits = model(points, mask)
    outputs = model.predict(points, mask, constrained=True)

    assert logits["object_logits"].shape[0] == 2
    assert logits["compressed_tokens"].shape == (2, 4, 24)
    assert len(outputs) == 2
    assert all(is_valid_semantic_output(output) for output in outputs)


def test_graph_modes_are_ablatable() -> None:
    points = torch.randn(2, 32, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)
    for graph_mode in ("no_graph", "simple_graph", "richer_graph"):
        model = PointSemanticsModel(
            encoder_hidden_dim=16,
            token_dim=24,
            compressed_tokens=4,
            decoder_hidden_dim=32,
            graph_mode=graph_mode,
            object_slots=5,
        )
        logits = model(points, mask)
        assert logits["graph_adjacency"].shape[0] == 2
        assert logits["object_logits"].shape[0] == 2


def test_input_adaptation_branch_runs() -> None:
    model = PointSemanticsModel(
        encoder_hidden_dim=16,
        token_dim=24,
        compressed_tokens=4,
        decoder_hidden_dim=32,
        adaptation_enabled=True,
        adaptation_mode="denoise",
    )
    points = torch.randn(2, 32, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)

    logits = model(points, mask)

    assert logits["object_logits"].shape[0] == 2
