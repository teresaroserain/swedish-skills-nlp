# Language Models for Skill Extraction in Swedish Job Advertisements

**Authors:** Minh Thanh Nguyen, Ngoc Buu Cat Nguyen, Duc Hong Sy Nguyen, Jonas
Harvard, Sara Ödmark, Lena-Maria Öberg

**Status:** Under submission.

This repository provides access to the data pipeline, model implementations,
frozen predictions, evaluation scripts, and inter-annotator reliability
analysis pertaining to the study on skill extraction from Swedish job
advertisements. The release encompasses experiments conducted within
software-development, journalist, and merged-domain contexts, targeting six
categories: all skills, hard skills, soft skills, distinct skills, must-have
skills, and nice-to-have skills.

Prospective users are requested to cite the associated publication should they
utilize the code, annotated data, or prompts derived from this study.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/raw_annotations/` | Source Excel annotation workbooks |
| `data/processed/` | Processed IT, journalist, and merged datasets |
| `data/prompt_development_sources/` | Prompt-development ID sources used by the split generator |
| `data/splits/` | Frozen seed-13 train, validation, and test files and manifests |
| `data/prediction_rules/` | Generated prediction rules and their source evidence |
| `data/llm_predictions/` | Retained GPT, Gemini, and Claude prediction workbooks and normalized sources |
| `src/data_processing/` | Dataset parsing and combination scripts |
| `src/splitting/` | Deterministic 40/10/50 split generator |
| `src/training/skill_training_scripts/` | Independent BERT and BART fine-tuning scripts |
| `src/baselines/` | Task-unadapted BERT and BART controls |
| `src/evaluation/` | Central scorer and before-versus-after comparison |
| `reliability/` | Inter-annotator reliability code, inputs, tests, and outputs |
| `results/` | Frozen model outputs, aggregate metrics, validation reports, and audits |

## Environment

Use Python 3.10 or newer. A CUDA-capable environment is strongly recommended
for BERT and BART runs. The final Kaggle runs utilized NVIDIA T4 GPUs. The
central evaluation and reliability calculations can run on CPU.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.in
```

`environment/requirements.in` lists direct dependencies rather than a complete
lock of the original Kaggle image. See `environment/README.md` for this
limitation.

## Data processing

The two source workbooks are already present under `data/raw_annotations/`.
The following commands rebuild the processed domain files in
`data/processed/`:

```bash
python src/data_processing/prepare_filtered_skill_datasets.py
python src/data_processing/combine_datasets.py
```

The first command applies the retained note filtering, duplicate-ID handling,
and bracket-aware parsing of all five annotation fields. It writes the IT,
journalist, and merged datasets and their skill-specific slices. The second
command builds `all_skills_list` for each domain.

Generate the deterministic seed-13 40/10/50 partitions with:

```bash
python src/splitting/create_prompt_safe_splits.py \
  --seed 13 \
  --output-dir data/splits
```

This command writes 54 JSONL partition files, 18 split manifests, and the split
audit tables. It balances input length and target density while minimizing
prompt-development IDs in the test partition. Run these commands in a clean
clone when checking reproducibility because they replace the corresponding
generated files.

Rebuild the prediction-only filtering resources with:

```bash
python src/prediction_rules/build_prediction_rules.py
```

The source evidence and checksums for the allowlist, fragment lists, and
stopword lists are under `data/prediction_rules/`.

## Fine-tuning

Every file in `src/training/skill_training_scripts/` is an independent run.
The filename identifies the model family, domain, and target. For example:

```bash
mkdir -p runs/bert_it_all_skills
cd runs/bert_it_all_skills
python ../../src/training/skill_training_scripts/train_bert_it_all_skills.py
```

For BART:

```bash
mkdir -p runs/bart_it_all_skills
cd runs/bart_it_all_skills
python ../../src/training/skill_training_scripts/train_bart_it_all_skills.py
```

Each script reads the frozen data and manifest from the repository regardless
of the current output directory. It performs 20 Optuna trials with seed 13,
uses exact normalized macro F1 as the validation objective, reinitializes the
selected model, performs final training, and writes predictions, HPO records,
hyperparameters, summaries, and the final model into the current directory.

To run all scripts sequentially from the repository root:

```bash
repository_root="$PWD"
for script in src/training/skill_training_scripts/*.py; do
  run_name="$(basename "$script" .py)"
  mkdir -p "runs/$run_name"
  (
    cd "runs/$run_name"
    python "$repository_root/$script"
  )
done
```

Run only one training process per GPU unless the available memory has been
verified. These scripts download the KBLab model checkpoints from Hugging Face
on first use.

For Kaggle, enable a T4 GPU and attach a dataset containing the repository's
`data/processed/` and `data/splits/manifests/` directories. The scripts search
`/kaggle/input` before checking the local repository.

## Task-unadapted controls

The control runner does not read train or validation records. BERT uses the
pretrained encoder with a fixed-seed random BIO classification head. BART uses
the pretrained denoising checkpoint without task fine-tuning.

Run both model families with:

```bash
python src/baselines/run_task_unadapted_baselines.py \
  --family bert \
  --device cuda \
  --output-root runs/task_unadapted_baselines

python src/baselines/run_task_unadapted_baselines.py \
  --family bart \
  --device cuda \
  --output-root runs/task_unadapted_baselines
```

Use `--domains` and `--skills` with comma-separated values to run a subset.
The accepted domains are `it`, `journal`, and `merged`. The accepted targets
are `all_skills`, `hard_skills`, `soft_skills`, `distinct_skills`,
`must_have`, and `nice_to_have`.

## Evaluation

The release already contains the validated fine-tuned and task-unadapted ZIP
archives. Recompute all metrics for BERT, BART, GPT, Gemini, and Claude with:

```bash
python src/evaluation/evaluate_all_models.py \
  --output-root results/recomputed
```

The scorer verifies the canonical test-ID order and gold labels before
producing 90 model-domain-target rows. It reports exact normalized,
boundary-relaxed containment, lexical, and semantic metrics, together with
unsupported-generation and alignment audits. Semantic matching uses
`KBLab/sentence-bert-swedish-cased` and may take several minutes on CPU.

Recreate the paired task-unadapted-versus-fine-tuned tables from the frozen
aggregate metrics with:

```bash
python src/evaluation/model_comparison.py
```

Frozen tables used for the manuscript are under `results/aggregate/` and
`results/before_after/`. Newly recomputed files are written separately under
`results/recomputed/` by the command above.

## Inter-annotator reliability

Run the reliability tests and then rebuild the reliability tables:

```bash
cd reliability
python -m unittest -v test_reliability_scorer.py
python reliability_checker.py --output-dir results
```

The implementation reads the two authoritative double-annotation workbooks,
uses bracket-aware parsing, applies exact-only deduplication, and performs
global one-to-one exact matching before contiguous-token containment matching.
Prediction-only filtering rules are not applied to human annotations.

## Frozen artifacts and verification

The scientific release contains:

- 36 fine-tuned BERT/BART output archives under `results/neural_run_zips/`.
- 36 task-unadapted control archives under
  `results/task_unadapted_baselines/output_zips/`.
- 90 centrally rescored model-domain-target rows under `results/aggregate/`.
- 10 recalculated inter-annotator domain-label rows under
  `reliability/results/`.

Read `REPRODUCIBILITY.md` for the concise pipeline order and
`DATA_AND_LICENSE_CHECKLIST.md` before public distribution.

The package contains one experimental seed, so rankings should be interpreted
descriptively. No password, API key, OAuth token, Kaggle credential, virtual
environment, or model checkpoint is intentionally included.

The repository is distributed under the existing MIT License; see
[LICENSE](LICENSE).
