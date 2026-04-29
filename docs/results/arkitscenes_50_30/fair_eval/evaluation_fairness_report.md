# Evaluation Fairness Report

## 1. Input leakage audit

# Input Leakage Audit

## Question

Does LGS use annotation-derived pseudo-points as input on the expanded ARKitScenes Validation-30 set?

## Result

Validation samples: 30  
Samples with mesh/point_path: 30  
Samples using annotation fallback/no point_path: 0

The first inspected validation samples used paths such as:

- `/cl/work11/hirotaka-m/ARKitScenes/3dod/Validation/41069021/41069021_3dod_mesh.ply`
- `/cl/work11/hirotaka-m/ARKitScenes/3dod/Validation/41069025/41069025_3dod_mesh.ply`

## Interpretation

For the expanded ARKitScenes Validation-30 evaluation, LGS uses mesh-derived point inputs from `<scene_id>_3dod_mesh.ply`. Annotation-derived pseudo-point fallback is not used.

The semantic targets are still derived from ARKitScenes 3DOD annotations. Relations are bbox-heuristic relations, not native human scene-graph annotations.

## Safe claim

LGS is evaluated on mesh-derived point inputs and annotation-derived semantic targets. The current evidence does not indicate annotation-point fallback leakage for the Validation-30 comparison.

## Caveat

This audit addresses input-point leakage. It does not make relation supervision human-annotated, nor does it convert the object-level metric into 3D detection AP.


## 2. Current shared-schema comparison

This is the current task-specialized comparison. It is useful for evaluating coarse semantic retention, but it is favorable to LGS because LGS natively predicts the shared schema.

| case | baseline | group | family | kind | execution | availability | condition | status | alignment | object_f1 | relation_f1 | count_exact | scene_accuracy | json_validity | json_mode | latency_ms | memory_mb | parameter_count | compression_ratio | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightweight_input_denoising__extreme_compression | LGS input denoising adaptation | main | robustness | internal_model | local | available_now | extreme_compression | completed_local | json=native, relations=native, scene=native | 0.8430 | 0.0000 | 0.0000 | 0.4111 | 1.0000 | native | 14.2237 | 1463.5977 | 88000.0000 | 28.2045 | Lightweight deterministic test-time input adaptation baseline inspired by CloudFixer-style robustness comparisons. |
| lightweight_input_denoising__severe_corruption | LGS input denoising adaptation | main | robustness | internal_model | local | available_now | severe_corruption | completed_local | json=native, relations=native, scene=native | 0.8430 | 0.0000 | 0.0000 | 0.4111 | 1.0000 | native | 14.4733 | 1430.8008 | 88000.0000 | 4.0810 | Lightweight deterministic test-time input adaptation baseline inspired by CloudFixer-style robustness comparisons. |
| lightweight_structured__clean | LGS lightweight structured | main | lightweight | internal_model | local | available_now | clean | completed_local | json=native, relations=native, scene=native | 0.8430 | 0.0000 | 0.0000 | 0.4111 | 1.0000 | native | 13.8383 | 1269.4336 | 88000.0000 | 1.0000 | Native lightweight structured baseline from this repository. |
| lightweight_structured__severe_corruption | LGS lightweight structured | main | lightweight | internal_model | local | available_now | severe_corruption | completed_local | json=native, relations=native, scene=native | 0.8430 | 0.0000 | 0.0000 | 0.4111 | 1.0000 | native | 13.7969 | 1283.1445 | 88000.0000 | 4.0810 | Native lightweight structured baseline from this repository. |
| spatiallm_import__clean | SpatialLM imported structured outputs | main | heavy_upper_bound | imported_structured | external_import | imported | clean | imported | json=converted, relations=mapped, scene=mapped | 0.6624 | 0.0000 | 0.0000 | 0.7667 | 1.0000 | converted | 0.0000 | 0.0000 | 0.0000 | 1.0000 | Heavy upper-bound comparison path. Full in-repo integration is intentionally out of scope; evaluate exported scene-level structured outputs instead. |
| spatiallm_import__severe_corruption | SpatialLM imported structured outputs | main | heavy_upper_bound | imported_structured | external_import | imported | severe_corruption | imported | json=converted, relations=mapped, scene=mapped | 0.0000 | 0.0000 | 0.0000 | 0.7667 | 1.0000 | converted | 0.0000 | 0.0000 | 0.0000 | 4.0810 | Heavy upper-bound comparison path. Full in-repo integration is intentionally out of scope; evaluate exported scene-level structured outputs instead. |
| votenet_import__clean | VoteNet imported detections | main | standard_3d | imported_detector | external_import | imported | clean | imported | json=converted, relations=derived, scene=derived | 0.6871 | 0.1784 | 0.0000 | 0.5000 | 1.0000 | converted | 1504.2168 | 0.0000 | 0.0000 | 1.0000 | Standard non-LLM 3D detection baseline evaluated after converting 3D detections into the shared coarse semantic JSON space. |
| votenet_import__severe_corruption | VoteNet imported detections | main | standard_3d | imported_detector | external_import | imported | severe_corruption | imported | json=converted, relations=derived, scene=derived | 0.0192 | 0.0000 | 0.0000 | 0.7333 | 1.0000 | converted | 1496.0518 | 0.0000 | 0.0000 | 4.0810 | Standard non-LLM 3D detection baseline evaluated after converting 3D detections into the shared coarse semantic JSON space. |


## 3. External-native diagnostics

This table gives SpatialLM and VoteNet a more favorable diagnostic by evaluating their raw box-like outputs with box matching and output-collapse metrics.

# External-Native Fairness Diagnostics

These metrics are diagnostic and are more favorable to box-producing external baselines than the shared semantic JSON metric.

| model | condition | scenes | empty_scenes | empty_rate | gt_boxes | pred_boxes | presence_f1 | box_iou025_f1 | box_iou050_f1 | derived_relation_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SpatialLM | clean | 30 | 6 | 0.2000 | 266 | 173 | 0.6624 | 0.6515 |  | 0.4382 |
| SpatialLM | severe_corruption | 30 | 30 | 1.0000 | 266 | 0 | 0.0000 | 0.0000 |  | 0.0000 |
| VoteNet | clean | 30 | 2 | 0.0667 | 266 | 163 | 0.6871 | 0.0000 |  | 0.2727 |
| VoteNet | severe_corruption | 30 | 29 | 0.9667 | 266 | 20 | 0.0192 | 0.0000 |  | 0.0000 |

## Interpretation

- `presence_f1` measures coarse category presence from boxes.
- `box_iou025_f1` and `box_iou050_f1` are detector-style greedy box matching diagnostics.
- `derived_relation_f1` derives relations from predicted boxes for both SpatialLM and VoteNet using the same coarse heuristic.
- LGS is not included here because it does not output boxes; this table complements, not replaces, the shared schema evaluation.


## 4. Recommended interpretation

The main claim should not be:

> LGS is a better general 3D detector or general 3D-language model.

The safer claim is:

> LGS is a task-specialized lightweight structured predictor that preserves coarse object-level semantics under severe geometric corruption. External 3D baselines collapse under the same shared-schema conversion, and external-native diagnostics should be reported as complementary evidence.

## 5. Metrics to emphasize

Primary:
- coarse object-level F1 under shared schema
- retention from clean to severe
- latency and parameter count
- empty-output rate of external baselines

Auxiliary:
- relation F1
- scene accuracy
- box-IoU diagnostics for external box-producing baselines

Limitations:
- relation targets are bbox-derived heuristics
- current object F1 is not 3D detection AP
- SpatialLM was evaluated through available `all/arch/object` layout interfaces, not native relation triples
