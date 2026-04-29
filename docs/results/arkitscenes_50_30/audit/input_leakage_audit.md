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
