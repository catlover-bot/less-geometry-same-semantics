#!/bin/bash
# Generic Slurm template for external baseline exports.
# Replace placeholders before use on the university server.

#SBATCH --job-name=lgss-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export ARKITSCENES_ROOT=/path/to/ARKitScenes
export EXPORT_ROOT=/path/to/server_exports
export BASELINE_REPO=/path/to/baseline_repo

cd "$BASELINE_REPO"

# 1. Run the baseline's own inference/export command here.
# 2. Save one export JSON per condition.
# 3. Save one metadata JSON per condition.
