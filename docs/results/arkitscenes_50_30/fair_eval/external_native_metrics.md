# External-Native Fairness Diagnostics

These metrics are diagnostic and are more favorable to box-producing external baselines than the shared semantic JSON metric.

| model | condition | scenes | empty_scenes | empty_rate | gt_boxes | pred_boxes | presence_f1 | box_iou025_f1 | box_iou05_f1 | derived_relation_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpatialLM | clean | 30 | 6 | 0.2000 | 266 | 173 | 0.6624 | 0.6515 | 0.5558 | 0.4382 |
| SpatialLM | severe_corruption | 30 | 30 | 1.0000 | 266 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| VoteNet | clean | 30 | 2 | 0.0667 | 266 | 163 | 0.6871 | 0.2424 | 0.0466 | 0.3226 |
| VoteNet | severe_corruption | 30 | 29 | 0.9667 | 266 | 20 | 0.0192 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

- `presence_f1` measures coarse category presence from boxes.
- `box_iou025_f1` and `box_iou05_f1` are detector-style greedy box matching diagnostics.
- `derived_relation_f1` derives relations from predicted boxes for both SpatialLM and VoteNet using the same coarse heuristic.
- LGS is not included here because it does not output boxes; this table complements, not replaces, the shared schema evaluation.
