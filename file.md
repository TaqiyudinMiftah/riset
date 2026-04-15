# Laporan Hasil Pipeline

Generated at (UTC): 2026-04-10T16:43:53Z

## Ringkasan

Laporan ini merangkum hasil run terbaru dari pipeline contextual emotion recognition pada dataset CAER-S lokal.

- Dataset source: `data/caers_balanced`
- Task type: `singlelabel`
- Fusion mode (main run): `multimodal`
- Output utama diambil dari:
  - `notebook_outputs/caer_s_local_qformer/results.json`
  - `notebook_outputs/caer_s_local_qformer/ablation_summary.json`
  - `notebook_outputs/caer_s_local_qformer/history.json`

## Konfigurasi Data Run

Quick bundle yang dipakai untuk training/eval utama:

- Train: 3000
- Val: 700
- Test: 1200
- Total: 4900

Stage-1 pseudo-label summary (`notebook_outputs/stage1_pseudo_labels.jsonl`):

- Total samples: 63000
- Heuristic-style labels: 62988
- VLM-generated labels (estimasi): 12

## Hasil Evaluasi Utama

Berdasarkan `notebook_outputs/caer_s_local_qformer/results.json`:

- Test loss: 1.865820949205954e-05
- mAP: 1.0
- AUC-ROC: 1.0
- Accuracy: 1.0

## Training History (Main Run)

- Epoch 1: train_loss=0.3428848497229033, val_loss=1.860646694302142e-05, val_mAP=1.0, val_acc=1.0
- Epoch 2: train_loss=0.0008333720577217931, val_loss=1.1871122327730435e-05, val_mAP=1.0, val_acc=1.0
- Epoch 3: train_loss=1.9777156442982464e-05, val_loss=6.955704528557114e-06, val_mAP=1.0, val_acc=1.0

## Hasil Ablation

Berdasarkan `notebook_outputs/caer_s_local_qformer/ablation_summary.json`:

- Vision:
  - mAP: 0.16387892278766447
  - AUC-ROC: 0.5255017312495589
  - Accuracy: 0.14333333333333334
- Text:
  - mAP: 1.0
  - AUC-ROC: 1.0
  - Accuracy: 1.0
- Multimodal:
  - mAP: 1.0
  - AUC-ROC: 1.0
  - Accuracy: 1.0

## Lokasi Artefak Penting

- Stage-1 pseudo labels: `notebook_outputs/stage1_pseudo_labels.jsonl`
- Best checkpoint: `notebook_outputs/caer_s_local_qformer/best_local.pt`
- History: `notebook_outputs/caer_s_local_qformer/history.json`
- Results: `notebook_outputs/caer_s_local_qformer/results.json`
- Ablation: `notebook_outputs/caer_s_local_qformer/ablation_summary.json`
- Attention overlay: `notebook_outputs/caer_s_local_qformer/attention_overlay.png`

## Catatan

- Stage-1 sudah dipatch agar kompatibel saat processor tidak punya chat template.
- Pada run ini, hanya 12 sample yang memakai output VLM (sesuai konfigurasi debug/sample limit), sisanya memakai pseudo-label heuristik.
- Jika ingin laporan benchmark yang lebih representatif, jalankan Stage-1 VLM full (tanpa sample limit kecil), lalu retrain Stage-2.
