#!/usr/bin/env bash
# Serve a trained BDS pi0.5 checkpoint over websocket (default port 8000).
# All caches are kept on the /mnt mount (nothing in $HOME).
#
# Usage: bash scripts/serve_bds.sh <checkpoint_step_dir> [port]
#   e.g. bash scripts/serve_bds.sh /mnt/data/sftp/data/vla/vr_checkpoints/pi05_bds_vfe_sim_pick_lora/<exp_name>/44999
set -euo pipefail

CKPT_DIR=${1:?"usage: $0 <checkpoint_step_dir> [port]"}
PORT=${2:-8000}
CONFIG=${CONFIG:-pi05_bds_vfe_sim_pick_lora}

PRETRAINED=/mnt/data/sftp/data/vla/vr_pretrained_checkpoints
export OPENPI_DATA_HOME=$PRETRAINED/openpi # paligemma tokenizer
export HF_HOME=$PRETRAINED/huggingface
export XDG_CACHE_HOME=$PRETRAINED/xdg_cache
# Inference only needs a fraction of the GPU; don't let JAX preallocate 75% of it.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

cd "$(dirname "$0")/.."
.venv/bin/python scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR"
