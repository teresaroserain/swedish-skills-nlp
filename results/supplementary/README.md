# Supplementary evaluation results

These tables accompany the revised manuscript *Language Models for Skill
Extraction in Swedish Job Advertisements*. They report the retained evaluation
after excluding the five identified prompt-development overlap advertisements
from every test allocation in which they occurred. Predictions, gold labels,
model runs, hyperparameters, splits and scoring rules were not changed for this
release.

The main configuration table contains 324 fine-tuned BERT/BART configurations
and 162 evaluations of saved GPT/Gemini/Claude outputs. The latter are reused
outputs on different test allocations, not 162 independent API runs.

## Files

| Files | Contents |
| --- | --- |
| `all_model_configurations.csv` | All 486 evaluations, including exact and containment macro and micro F1 |
| `fine_tuning_grid.csv`, `fine_tuning_grid_summary.csv` | All 324 fine-tuning configurations and nine-configuration summaries |
| `reference_seed13.csv`, `reference_job_category_scores.csv` | Original split/training seed 13/13 comparisons after exclusions |
| `bootstrap_job_category_comparisons.csv` | 126 comparisons, with percentile intervals and both Holm correction families |
| `bootstrap_task_comparisons.csv` | 756 task-specific comparisons |
| `bert_alignment_counts.csv`, `bert_unmatched_targets.csv` | Actual training-parser and regex audit before tokenization |
| `alignment_failure_categories.csv` | Mechanical diagnostics of regex failures; not a semantic classification |
| `derived_offset_scores.csv`, `source_support_scores.csv` | Derived-offset and source-support diagnostics |
| `gold_subset_*.csv` | Shared recoverable/unique gold-only subsets, sample sizes and sensitivity scores |
| `frozen_control_*.csv` | Trained-head frozen-BERT versus full fine-tuning at all-skills seed 42/42 |
| `label_overlap.csv`, `distinct_skills_reliability.csv` | Label structure and inter-annotator agreement |
| `error_*.csv` | Automated error counts and summaries over the complete retained evaluations |
| `containment_length_*.csv` | Unchanged containment and supplementary 0.50/0.75 token-length-ratio constraints |
| `prompt_test_isolation.csv`, `excluded_prompt_development_ids.csv` | Test-allocation checks and the five known exclusions |
| `prompt_source_inventory.csv` | Ten retained occupation/label optimization sources and their checksums |
| `split_overlap.csv`, `frozen_hyperparameters.csv` | Partition overlap and frozen HPO selections |
| `execution_metadata.csv`, `llm_request_settings.csv`, `llm_retry_policies.csv`, `recorded_environments.csv` | Recoverable execution and request settings, including unrecorded fields |
| `per_advertisement_metrics.csv.gz` | 82,674 paired advertisement-level evaluations, with counts and scores; no advertisement text |
| `manuscript_table_map.csv` | Manuscript labels, table numbers in source version 9, and corresponding CSVs |
| `checksums.json` | Export/source SHA-256 hashes and any path-only sanitization |

## Metric definitions

`exact_f1`, `exact_precision` and `exact_recall` in the configuration table are
arithmetic means of per-advertisement values, giving each advertisement equal
weight. `exact_micro_f1` instead equals `2 * exact_tp / (predicted_items +
gold_items)`, using cumulative counts. The containment columns use the same
aggregation definitions with cumulative exact-first, one-to-one contiguous-token
matches. An empty gold/prediction pair has F1 zero under the retained scorer.

The six tasks are all-skills, hard skills, soft skills, distinct skills,
must-have and nice-to-have. Job-category summaries weight those six tasks
equally; they are not mention-weighted micro scores. Merged reuses the IT and
Journalism corpora and is not an independent third occupation.

The fine-tuning grid crosses three partition seeds with three training seeds
under frozen hyperparameters. Nine configurations are descriptive repeated
fits, not nine independent test samples. Prompted-model outputs are reused.

Bootstrap tables use 10,000 paired advertisement-level resamples, averaging
training seeds before comparison and stratifying Merged by source occupation.
P-values include separate seven-pair and global Holm families. Percentile
confidence intervals are not simultaneous multiplicity-adjusted intervals.
Finite resampling resolution limits global task-level significance detection.

Containment-length tables impose shorter/longer normalized token ratios of
0.50 or 0.75 in addition to contiguous-token containment. These descriptive
sensitivity results do not replace the primary metric. A separate expanded
bootstrap analysis of these constraints is not included in this release or
claimed as a manuscript result.

Derived offsets are retrospectively recovered string locations, not
expert-verified gold spans. Source-aligned token F1 omits unalignable items from
its token sets and must be read with source-absence and recovery counts.
Recoverable/unique gold-only subset scores retain full gold and predictions
for each included advertisement. They do not estimate the causal benefit of
repairing training supervision. Mechanical failure categories are mutually
exclusive checks applied in the order documented by their names; unresolved
surface mismatches are not automatically called paraphrases or implicit skills.

Error FP/P and FN/G summaries first compute cumulative ratios within each
evaluation, then average configurations/partitions and give the three job
category settings equal weight. These are not complements of macro precision
or recall, nor a manually validated taxonomy of semantic causes.

The prompt source inventory establishes the retained common optimization
sources, not complete model-specific manual-revision histories or equal
development budgets. The exclusion audit addresses identified overlap; it
does not reconstruct all historical prompt-development exposure.

## Provenance and access

Only machine-local absolute paths were reduced to filenames in selected
metadata fields; the affected columns are listed in `checksums.json`.
Evaluation counts, scores, labels, IDs and hashes were preserved. Internal
monitoring scripts, manuscript drafts, reviewer correspondence and credentials
are not part of this release. The repository's existing license is unchanged.

Download the CSVs directly from this directory. The compressed per-advertisement
file can be read using `pandas.read_csv("per_advertisement_metrics.csv.gz")`.
Table numbers can change after manuscript rearrangement; use the stable labels
in `manuscript_table_map.csv` when checking subsequent revisions.
