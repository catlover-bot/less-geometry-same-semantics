#!/bin/bash
# Environment template for external baseline handoff.

export ARKITSCENES_ROOT=/path/to/ARKitScenes
export EXPORT_ROOT=/path/to/server_exports
export CONDITION=clean
export SPLIT=Validation

# Save:
# - $EXPORT_ROOT/<baseline>/<condition>_export.json
# - $EXPORT_ROOT/<baseline>/<condition>_run_metadata.json
