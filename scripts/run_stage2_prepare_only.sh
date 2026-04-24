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

CMD=(
  "$PYTHON_BIN" scripts/train_stage2_from_pseudo_labels.py
  --target-total-pseudo "$TARGET_TOTAL_PSEUDO"
  --no-run-train
  --no-run-eval
  --no-run-ablation
  --no-wandb-enable
  --wandb-mode disabled
)

CMD+=("$@")

echo "[run_stage2_prepare_only] PROJECT_ROOT=$PROJECT_ROOT"
echo "[run_stage2_prepare_only] TARGET_TOTAL_PSEUDO=$TARGET_TOTAL_PSEUDO"
echo "[run_stage2_prepare_only] Command: ${CMD[*]}"

"${CMD[@]}"

echo "[run_stage2_prepare_only] Done."
