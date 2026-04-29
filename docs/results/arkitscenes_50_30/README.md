# ARKitScenes 50/30 Experiments

This folder summarizes the expanded ARKitScenes experiments.

## Dataset split

- Training scenes: 50
- Validation scenes: 30

## Compared systems

- LGS: lightweight task-specialized structured semantic predictor
- SpatialLM: imported 3D-LM/layout baseline
- VoteNet: imported 3D detector baseline

## Key measured model sizes

| Model | Parameters | Parameters (M) |
|---|---:|---:|
| LGS | 88,000 | 0.088 |
| VoteNet | 953,828 | 0.954 |
| SpatialLM | 603,511,168 | 603.511 |

## Key latency values

| Model | Condition | Latency |
|---|---|---:|
| LGS | clean/severe | ~13.8 ms/sample |
| SpatialLM | clean | 146.6173 ms/sample |
| SpatialLM | severe | 138.0709 ms/sample |
| VoteNet | clean/severe | ~1500 ms/sample |

## Interpretation

LGS should be interpreted as a task-specialized lightweight structured predictor for coarse semantic retention, not as a general replacement for 3D detectors or 3D-language models.

The strongest supported claim is:

> LGS preserves coarse object-level semantics under severe geometric corruption while remaining much smaller and faster than the imported external baselines.

## Important caveats

- The main shared-schema evaluation is favorable to LGS because LGS natively predicts the shared JSON schema.
- SpatialLM is strong on clean box/layout diagnostics.
- VoteNet requires coordinate alignment for box-IoU diagnostics.
- Relation metrics are auxiliary because relation targets are bbox-derived heuristics.
- The current ARKitScenes subset is dominated by cabinet, chair, table, sofa, and lamp.
