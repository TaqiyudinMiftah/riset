# Run Stage-2 From Existing Pseudo Labels

Panduan cepat menjalankan training stage-2 tanpa notebook.

## Prasyarat

1. Jalankan dari root repo.
2. Environment virtual aktif.
3. Dependency sudah terpasang dari requirements.

Contoh:

- cd /home/agung/riset
- source .venv/bin/activate
- pip install -r requirements.txt

## Script Siap Pakai

### 1) Prepare data saja (tanpa train/eval)

- bash scripts/run_stage2_prepare_only.sh

### 2) Train + eval (default workflow)

- export WANDB_API_KEY="isi_api_key_kamu"
- bash scripts/run_stage2_train_eval.sh

### 3) Train + eval + ablation

- export WANDB_API_KEY="isi_api_key_kamu"
- bash scripts/run_stage2_full.sh

## Opsi Override Cepat

Semua script menerima override via environment variable.

- TARGET_TOTAL_PSEUDO: default 10000
- WANDB_ENABLE: default true (untuk train/eval/full)
- WANDB_MODE: default online
- WANDB_PROJECT: default emotion-reasoning
- WANDB_ENTITY: default kosong
- WANDB_RUN_NAME: default otomatis timestamp
- PYTHON_BIN: default python

Contoh:

- TARGET_TOTAL_PSEUDO=20000 bash scripts/run_stage2_train_eval.sh
- WANDB_ENABLE=false bash scripts/run_stage2_train_eval.sh
- WANDB_MODE=offline WANDB_RUN_NAME=caers-offline-10k bash scripts/run_stage2_train_eval.sh

## Tambahan Argumen CLI

Semua argumen tambahan akan diteruskan ke script python utama.

Contoh:

- bash scripts/run_stage2_train_eval.sh --batch-size 16 --epochs 20
- bash scripts/run_stage2_full.sh --vision-lr 5e-6 --text-lr 5e-5

## Output Penting

- Data stage-2 siap train:
  - notebook_outputs/risetv1_qwen/stage1_pseudo_labels_qwen_train10k_ready.jsonl
- Config run:
  - notebook_outputs/risetv1_qwen/configs/caers_qwen_qformer.json
- Ringkasan run:
  - notebook_outputs/risetv1_qwen/stage2_train10k_summary.json
- Checkpoint/results:
  - notebook_outputs/risetv1_qwen/stage2_qformer/
