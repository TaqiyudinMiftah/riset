# Playbook Ringkas main_notebook

Dokumen ini merangkum urutan cell paling efektif untuk dua mode:
- Quick run (validasi cepat pipeline)
- Full run (eksekusi lebih serius)

Notebook acuan: main_notebook.ipynb

## A. Quick Run (disarankan untuk sanity check)

Tujuan:
- Memastikan pipeline 2-stage berjalan end-to-end
- Menghasilkan artefak utama dengan waktu lebih singkat

Urutan cell:
1. #VSC-1f8312f1
- Cek GPU dengan nvidia-smi

2. #VSC-0d10f846
- Set switch runtime (pastikan QUICK_DEBUG_MODE=True)

3. #VSC-855741a6
4. #VSC-aab6d61b
- Validasi dependency inti

5. (Opsional) #VSC-3a63c0bc lalu #VSC-a3fb53da
- Jalankan hanya jika package VLM belum siap

6. #VSC-033a8702
- Aktifkan dataset lokal CAER-S dari /home/agung/riset/caer_dataset/CAER-S

7. #VSC-fc676d1d
8. #VSC-58b61d84
- Setup dataset + integrity check

9. #VSC-3c31bab7
10. #VSC-53b45a1f
11. #VSC-d8bf95bc
- Stage 1 pseudo-label + sanity check kualitas output

12. #VSC-7c06242e
- Export dataset bundle manifest

13. #VSC-ca3f5dba
- CUDA-only guard + quick subset real data

14. #VSC-efc4190d
15. #VSC-41b0248e
- Stage 2 training + final test evaluation

16. #VSC-c99f57df
17. #VSC-fcb430b9
18. #VSC-6c8d11f8
- Ablation + attention visualization + ringkasan artefak

Artefak minimum yang harus muncul:
- notebook_outputs/stage1_pseudo_labels.jsonl
- notebook_outputs/caer_s_local_qformer/best_local.pt
- notebook_outputs/caer_s_local_qformer/results.json
- notebook_outputs/caer_s_local_qformer/attention_overlay.png

## B. Full Run (untuk eksperimen lebih serius)

Perbedaan dari quick run:
- Set QUICK_DEBUG_MODE=False di cell #VSC-0d10f846
- Naikkan LOCAL_VLM_SAMPLE_LIMIT sesuai kebutuhan di cell #VSC-0d10f846
- Pertahankan RUN_STAGE1_VLM=True dan RUN_ABLATION=True

Urutan:
- Ikuti urutan Quick Run dari awal sampai akhir
- Tambahkan blok riset dan pelaporan otomatis:
  - #VSC-38a6efa4 (research runner)
  - #VSC-4e6073f4 (auto report generation)

Artefak tambahan full run:
- notebook_outputs/research_runs/research_summary.json
- notebook_outputs/research_runs/research_report.md
- notebook_outputs/research_runs/research_report.txt

## C. Jalur Real Dataset Auto-Discovery (opsional)

Gunakan jika ingin notebook mencari annotation/image root otomatis di server:
- #VSC-ce984a89
- #VSC-0986ed9d

Catatan:
- Jika auto-discovery gagal, tetap gunakan jalur manual lewat #VSC-033a8702.

## D. Checklist Cepat Debug

Jika Stage 1 gagal:
- Pastikan transformers dan accelerate terdeteksi di #VSC-855741a6
- Jalankan patch fallback chat template di #VSC-3c31bab7

Jika Stage 2 gagal:
- Pastikan CUDA aktif di #VSC-1f8312f1 dan #VSC-ca3f5dba
- Cek integritas data di #VSC-58b61d84

Jika metrik terlihat tidak wajar:
- Validasi jumlah sampel train/val/test di output #VSC-efc4190d
- Bandingkan hasil multimodal vs ablation di #VSC-c99f57df

## E. Rekomendasi Operasional

- Untuk iterasi harian: gunakan Quick Run.
- Untuk pelaporan eksperimen: gunakan Full Run + research runner.
- Simpan dan bandingkan hasil utama dari results.json dan research_summary.json agar reproducible.
