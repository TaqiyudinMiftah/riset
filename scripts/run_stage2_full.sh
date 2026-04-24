#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_TOTAL_PSEUDO="${TARGET_TOTAL_PSEUDO:-10000}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-emotion-reasoning}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-caers-qwen-full-${TARGET_TOTAL_PSEUDO}-$(date +%Y%m%d-%H%M%S)}"

CMD=(
  "$PYTHON_BIN" scripts/train_stage2_from_pseudo_labels.py
  --target-total-pseudo "$TARGET_TOTAL_PSEUDO"
  --run-train
  --run-eval
  --run-ablation
)

if [[ "$WANDB_ENABLE" == "true" ]]; then
  CMD+=(
    --wandb-enable
    --wandb-mode "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  CMD+=(--no-wandb-enable --wandb-mode disabled)
fi

CMD+=("$@")

echo "[run_stage2_full] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_stage2_full] TARGET_TOTAL_PSEUDO=$TARGET_TOTAL_PSEUDO"
echo "[run_stage2_full] WANDB_ENABLE=$WANDB_ENABLE"
echo "[run_stage2_full] Command: ${CMD[*]}"

"${CMD[@]}"

echo "[run_stage2_full] Done."
