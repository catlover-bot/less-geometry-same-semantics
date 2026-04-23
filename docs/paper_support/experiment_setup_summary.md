# Experiment Setup Summary

Primary benchmark: ARKitScenes 3DOD.

Legacy 3RScan/3DSSG and ScanNet configs are archived and are not part of the default paper workflow.

Main matrix:

- clean vs severe corruption
- raw vs compressed point/token budget
- no graph vs simple graph bottleneck
- unconstrained vs schema-constrained decoding
- no adaptation vs lightweight input normalization

Corruption families:

- geometry degradation
- coordinate perturbation
- local structural corruption
- token/point compression
