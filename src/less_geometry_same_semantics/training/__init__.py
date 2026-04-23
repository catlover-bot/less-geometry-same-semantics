"""Training and evaluation loops."""

from less_geometry_same_semantics.training.evaluate import evaluate_model, evaluate_predictions
from less_geometry_same_semantics.training.loop import train_one_epoch

__all__ = ["evaluate_model", "evaluate_predictions", "train_one_epoch"]
