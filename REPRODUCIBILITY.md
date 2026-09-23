# Reproduction order

## 1. Data processing

`src/data_processing/prepare_filtered_skill_datasets.py` parses both annotated Excel
workbooks, applies the note and duplicate-ID filters, uses bracket-aware parsing
for all five label fields, and writes the software-developer, journalist, and
merged processed files.
`src/data_processing/combine_datasets.py` creates each `all_skills_list` dataset.

The scripts use repository-relative paths. The corresponding inputs are in
`data/raw_annotations/`, and regenerated outputs are written to
`data/processed/`.

## 2. Prompt-safe split

Run `src/splitting/create_prompt_safe_splits.py --seed 13`. The script minimizes
prompt-development IDs in test while maintaining 40/10/50 counts and balancing
input length and target density. The frozen output is in `data/splits/`.

## 3. Auditable prediction rules

Run `src/prediction_rules/build_prediction_rules.py`, then
use the frozen training scripts. The evidence sources, checksums, derived lists,
and all 36 training scripts are included.

## 4. Fine-tuning

The 18 BERT and 18 BART scripts are in `src/training/skill_training_scripts/`.
Each script executes 20 seeded Optuna trials, selects by exact normalized macro
F1 on validation, reinitializes the model, trains once with seed 13, and writes
auditable test predictions and HPO metadata.

## 5. Task-unadapted controls

Run `src/baselines/run_task_unadapted_baselines.py` separately for BERT and BART.
These controls never read train or validation records. BERT uses a fixed-seed
random BIO head on the pretrained encoder; BART uses the pretrained denoising
checkpoint. The central scorer reads the resulting ZIPs using the same frozen
prediction rules as the fine-tuned systems.

The 36 validated control ZIPs and their audit tables are preserved under
`results/task_unadapted_baselines/`. Run
`src/evaluation/model_comparison.py` after central scoring to
recreate the paired tables under `results/before_after/`.

## 6. Validation and all-model scoring

Run `python src/evaluation/evaluate_all_models.py` to reconstruct all
90 centrally scored model--domain--skill rows from the frozen split data,
prediction rules, neural ZIPs, and normalized LLM workbooks in this release.
Known Python dependencies are documented under `environment/`.

## 7. Inter-annotator reliability

The original reliability implementation and intermediate outputs are retained
under `reliability/` for provenance. Run `python -m unittest -v test_reliability_scorer.py`
there, then run `python reliability_checker.py`. The evaluation pipeline scorer reads the two
authoritative double-annotation workbooks, preserves commas inside square
brackets, applies exact-only deduplication, and computes global one-to-one exact
matching before token-containment matching. It does not apply prediction-only
filters to either human annotator. Outputs are written to
`reliability/results/`, including a comparison with the current
manuscript table.

## 8. Integrity

`manifests/file_manifest.csv` contains the public file size, SHA-256, and role.
Frozen integrity, scorer-reproduction, implementation-alignment, and manuscript
result reports are under `results/audits/` and `results/aggregate/`.
