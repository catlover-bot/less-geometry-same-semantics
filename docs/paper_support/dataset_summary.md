# Dataset Summary

Fill this section from dataset diagnostics artifacts:

- `diagnostics.json`
- `summary.csv`
- `object_category_histogram.csv`
- `relation_category_histogram.csv`
- histogram plots

Report:

- split sizes
- skipped / invalid samples
- average points per scene
- average objects per scene
- average relations per scene
- object and relation category coverage

Known assumption: ARKitScenes provides object boxes, not explicit scene-graph relation triplets. Relations in this framework are derived from conservative bounding-box heuristics and should be reported as heuristic graph supervision.
