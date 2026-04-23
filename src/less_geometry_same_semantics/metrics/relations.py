"""Relation-level metrics."""

from __future__ import annotations

from typing import Any

def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from aggregate counts."""

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0 if fp == 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def relation_tuple(relation: dict[str, Any]) -> tuple[str, str, str]:
    """Normalize a relation dict to a hashable tuple."""

    return (
        str(relation.get("subject", "")),
        str(relation.get("predicate", "")),
        str(relation.get("object", "")),
    )


def relation_prf1(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """Micro-average exact relation triple precision/recall/F1."""

    tp = fp = fn = 0
    for pred, ref in zip(predictions, references, strict=True):
        pred_relations = {relation_tuple(rel) for rel in pred.get("relations", [])}
        ref_relations = {relation_tuple(rel) for rel in ref.get("relations", [])}
        tp += len(pred_relations & ref_relations)
        fp += len(pred_relations - ref_relations)
        fn += len(ref_relations - pred_relations)
    return precision_recall_f1_from_counts(tp, fp, fn)
