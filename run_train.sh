#!/bin/bash
#SBATCH --job-name=fjsp_train
#SBATCH --account=rrg-cglee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/train_seed%a_%j.out
#SBATCH --error=logs/train_seed%a_%j.err

# Trains the 1-layer_experiment models: SD1, 20x10. Only seed 0 for now
# (array range can be widened back to 0-2 later to add seeds 1/2).
# Submit with: sbatch run_train.sh

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "${REPO_ROOT}"
mkdir -p logs

module purge
module load StdEnv/2023 python/3.11.5 cuda/12.6

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="/home/grafyann/master_thesis_env/bin/python"
[[ -x "${PYTHON}" ]] || { echo "Missing ${PYTHON}" >&2; exit 1; }

SEED=${SLURM_ARRAY_TASK_ID}
N_J=20
N_M=10
DATA_SOURCE=SD1
MAX_UPDATES=1000

RUN_TAG=1-layer_experiment

echo "=== Training seed=${SEED} data_source=${DATA_SOURCE} size=${N_J}x${N_M} run_tag=${RUN_TAG} ==="
srun nvidia-smi || true

exec srun "${PYTHON}" train.py \
    --data_source "${DATA_SOURCE}" \
    --n_j "${N_J}" \
    --n_m "${N_M}" \
    --max_updates "${MAX_UPDATES}" \
    --seed_train "${SEED}" \
    --model_suffix "${RUN_TAG}_seed${SEED}" \
    --device cuda
