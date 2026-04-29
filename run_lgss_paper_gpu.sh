#!/bin/bash
#SBATCH -A is-nlp
#SBATCH -p gpu_short
#SBATCH -w elm43
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J lgss_paper
#SBATCH -o /project/nlp-work11/%u/lgss_paper_%j.out
#SBATCH -e /project/nlp-work11/%u/lgss_paper_%j.out

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

echo "=== DIAGNOSTICS ==="
python scripts/run_dataset_diagnostics.py \
  --config configs/arkitscenes.yaml \
  --output-dir outputs/diagnostics/arkitscenes \
  --max-scenes 50

echo "=== PAPER PACKAGE ==="
python scripts/run_paper_package.py --plan configs/paper_plan.yaml
