# Problem Statement

We study whether task-relevant semantics can be preserved under aggressive point-cloud degradation and lightweight language modeling, even when geometric fidelity is not preserved.

The central representation is a graph-centric semantic bottleneck:

```text
point cloud -> object abstractions / relations / scene graph -> structured semantic output
```

The framework prioritizes object categories, object counts, coarse attributes, coarse spatial relations, scene type, and optional short captions over high-fidelity geometric reconstruction.
