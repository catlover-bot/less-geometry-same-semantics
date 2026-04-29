#!/bin/bash
#SBATCH -A lang
#SBATCH -p gpu_intr
#SBATCH -w elm43
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
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

hostname
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml
python scripts/run_dataset_diagnostics.py \
  --config configs/arkitscenes.yaml \
  --output-dir outputs/diagnostics/arkitscenes \
  --max-scenes 5
python scripts/run_paper_package.py --plan configs/paper_plan.yaml
