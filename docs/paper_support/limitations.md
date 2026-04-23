# Limitations Draft

- The first model uses a lightweight object abstraction stage rather than a full detector or instance proposal network.
- The shared schema intentionally collapses labels into a coarse taxonomy, which improves comparability but loses fine-grained categories.
- ARKitScenes relation labels are derived using conservative geometry heuristics, not human scene-graph annotations.
- Public dataset layout variations may require small adapter extensions.
- Caption evaluation is optional and not a primary metric in the current framework.
- The framework measures semantic preservation, not geometric reconstruction fidelity.
