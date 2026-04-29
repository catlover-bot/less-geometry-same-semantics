#!/bin/bash
#SBATCH -A lang
#SBATCH -p gpu_short
#SBATCH -w elm43
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J gpucheck43
#SBATCH -o /project/nlp-work11/%u/gpucheck43_%j.out
#SBATCH -e /project/nlp-work11/%u/gpucheck43_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lgss_py310
cd ~/workspace/less-geometry-same-semantics

hostname
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
