#!/usr/bin/env bash
# Finetune pi0.5 (LoRA) on the BDS VR_H5D sim pick datasets.
# All caches, pretrained weights and checkpoints are kept on the /mnt mount (nothing in $HOME).
#
# Usage: [CONFIG=<config_name>] bash scripts/train_bds.sh <exp_name> [extra train.py args...]
set -euo pipefail

CONFIG=${CONFIG:-pi05_bds_vfe_sim_pick_lora}
EXP_NAME=${1:?"usage: $0 <exp_name> [extra args]"}
shift

VLA_ROOT=/mnt/data/sftp/data/vla
PRETRAINED=$VLA_ROOT/vr_pretrained_checkpoints

export OPENPI_DATA_HOME=$PRETRAINED/openpi           # pi05_base params + paligemma tokenizer
export HF_HOME=$PRETRAINED/huggingface
export HF_LEROBOT_HOME=$PRETRAINED/lerobot
export JAX_COMPILATION_CACHE_DIR=$PRETRAINED/jax_cache
export XDG_CACHE_HOME=$PRETRAINED/xdg_cache
export WANDB_DIR=$VLA_ROOT/vr_checkpoints/wandb
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

cd "$(dirname "$0")/.."

# Compute normalization stats once (written to vr_checkpoints/assets/<config>/<repo_id>).
NORM_STATS=$(.venv/bin/python -c "
import openpi.training.config as c
cfg = c.get_config('$CONFIG')
print(cfg.assets_dirs / cfg.data.repo_id / 'norm_stats.json')")
if [ ! -f "$NORM_STATS" ]; then
    .venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG"
fi

.venv/bin/python scripts/train.py "$CONFIG" --exp-name "$EXP_NAME" "$@"
