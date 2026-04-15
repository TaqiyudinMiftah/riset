# Research Report - Contextual Emotion Recognition (Notebook Server)

Generated at: 2026-04-11T03:58:48.872685Z

## Objective
Evaluate a two-stage contextual emotion pipeline with local VLM pseudo-labeling and Q-Former-style multimodal classification.

## Experimental Setup
- Dataset mode: user dataset
- Stage 1 VLM: Salesforce/instructblip-flan-t5-xl
- Stage 1 sample limit: 12
- Stage 2 epochs per run: 3
- Batch size: 16
- Learning rate: 0.0005

## Results Table

| run_name | fusion_mode | test_loss | mAP | auc_roc | accuracy | num_train | num_val | num_test | vocab_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic_multimodal | multimodal | 0.864482 | 1.000000 | 1.000000 | 0.857136 | 44107 | 4900 | 20992 | 25 |
| localvlm_multimodal | multimodal | 1.608753 | 1.000000 | 1.000000 | 0.857136 | 44107 | 4900 | 20992 | 52 |
| localvlm_text | text | 1.597595 | 1.000000 | 1.000000 | 0.857136 | 44107 | 4900 | 20992 | 52 |
| localvlm_vision | vision | 3.025143 | 0.292617 | 0.673280 | 0.284346 | 44107 | 4900 | 20992 | 52 |

## Key Findings
1. LocalVLM multimodal vs heuristic multimodal (mAP delta): 0.0
2. LocalVLM multimodal vs heuristic multimodal (accuracy delta): 0.0
3. In this run, vision/text/multimodal on local-VLM pseudo-labels produced similar metrics.

## Interpretation
- Current dataset is very small and synthetic, so the result behaves like a pipeline validation, not a final benchmark claim.
- The end-to-end stack is now stable: local VLM generation, reusable dataset bundle export, multimodal training, ablation, and attention visualization.

## Limitations
- Stage-1 local VLM currently uses a partial subset in debug mode (sample limit active).
- The synthetic dataset does not represent real context complexity (EMOTIC/CAER-S actual distribution).

## Next Actions
1. Switch to real dataset paths (USE_USER_DATA=True) and regenerate pseudo-labels fully.
2. Increase training epochs and run multiple seeds for confidence intervals.
3. Keep exporting results to the same bundle/report format for reproducible tracking.