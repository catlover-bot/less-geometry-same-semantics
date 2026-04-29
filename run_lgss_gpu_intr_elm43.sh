#!/bin/bash
#SBATCH -A is-nlp
#SBATCH -p gpu_intr
#SBATCH -w elm43
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J lgss_gpucheck
#SBATCH -o /project/nlp-work11/%u/lgss_gpucheck_%j.out
#SBATCH -e /project/nlp-work11/%u/lgss_gpucheck_%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lgss_py310
cd ~/workspace/less-geometry-same-semantics

export ARKITSCENES_ROOT=/cl/work11/$USER/ARKitScenes
export HF_HOME=/cl/work11/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export PATH="$HOME/bin:$PATH"
mkdir -p ~/bin
ln -sf "$(which curl)" ~/bin/curl.exe

echo "=== HOST ==="
hostname
echo "=== GPU ==="
nvidia-smi
echo "=== TORCH ==="
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "=== SETUP CHECK ==="
python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml

echo "=== ONE DEBUG CASE ==="
python scripts/run_main_experiments.py \
  --config configs/arkitscenes.yaml \
  --epochs 0 \
  --seeds 7 \
  --max-cases 1 \
  --output outputs/debug_one_case/results.json \
  --artifacts-dir outputs/debug_one_case
