# Additional evaluation data

This directory publishes the frozen CSVs for Figure 5 and the 20 examples in
Table 24. Use the LaTeX labels in `manuscript_map.csv` as stable references when
manuscript numbering changes. These files supplement the original release;
they do not replace its model archives, reference scores or scoring rules.

## BERT learning curves

The CSVs in `bert_learning_curve/results/` contain all 432 configurations:
three job-category settings, six extraction settings, eight fractions of the
original training partition and three paired subset-sampling/training seeds.
The fractions are 12.5, 25, 37.5, 50, 62.5, 75, 87.5 and 100 percent. Seeds
are 13, 42 and 2026; the 100-percent points reuse 54 existing fits. Validation
and test IDs remain those of partition seed 13, with the five known
prompt-development overlap IDs excluded from scoring. HPO is not repeated.

| File | Contents |
| --- | --- |
| `run_metrics.csv` | All 432 configurations, partition sizes, exact/containment macro and micro scores, and mention counts |
| `per_ad_metrics.csv.gz` | All 73,512 advertisement-level evaluations used by the learning curves |
| `task_summary.csv` | 144 job-category/task/fraction groups, each summarizing three seeds |
| `all_skills_summary.csv` | The 24 all-skills groups shown in the upper panels of Figure 5 |
| `domain_by_seed.csv` | Equal-weight averages of the six extraction settings within each paired seed |
| `domain_summary.csv` | Three-seed summaries of those equal-weight setting averages, shown in the lower panels |
| `coverage.csv` | Available and expected seeds for every group |
| `exclusion_audit.csv` | Exclusion records for the known overlapping advertisements |

Macro scores average per-advertisement values. Micro scores pool mention
counts within each configuration. Summary columns report the arithmetic mean,
sample SD, minimum and maximum across the three paired seeds. The SD is not
a confidence interval. The seeds vary subset sampling and training together;
they are not nine independent seed combinations. Merged reuses the two
occupation corpora rather than supplying an independent third occupation.

Figure 5 uses `all_skills_summary.csv` for the upper panels and
`domain_summary.csv` for the lower panels. Select `exact_macro_f1_mean` and
`containment_macro_f1_mean` as the plotted values, with the corresponding
`*_sd` columns as one-SD error bars. Percentages refer to the original training
partition, not to the complete corpus. These CSVs report completed runs;
the original training archives are not duplicated in this directory.

## Table 24 examples

`error_examples.csv` has one FP and one FN for each occupation/model pair:
two occupations and BERT, BART, GPT, Gemini and Claude. All five annotated
labels are represented. The reference neural configuration uses split/training
seed 13/13; the LLM examples use saved outputs on the corresponding retained
test allocation, without new API calls.

`identifier` is the full advertisement ID; the table prints its eight-character
prefix. `context` is the source advertisement, while `gold` and `prediction`
are the complete saved mention lists. `phrase` identifies the illustrated
unmatched item and `role` is FP or FN. `source_excerpt_sv` and
`source_excerpt_en` contain the original excerpt and its English translation.
`prediction_excerpt` and `gold_excerpt` give the selected mentions printed in
the table. These array-valued columns are JSON strings, not semicolon lists.

FP/FN classification uses the complete mention sets under the unchanged
exact-first, one-to-one primary containment scorer. Each illustrated error
has no containment candidate in the opposite set for that advertisement and
label. The examples illustrate errors relative to recorded gold; they do not
estimate error prevalence or provide new expert adjudication. The five known
overlap IDs are absent. Local path prefixes in `result_file`, `data_file` and
`manifest` were reduced to filenames; IDs, mentions and predictions were not
changed. No credentials or private machine paths are included.

## Verification

`checksums.json` records release and source SHA-256 values. The learning-curve
CSV and gzip bytes are identical to the completed source analysis. The
20-example export adds the displayed excerpts and sanitizes only the three
path columns described above. `validation_report.json` records completeness,
summary reconciliation, ID exclusions and re-verification against the original
saved model outputs.

In the existing `../gold_subset_coverage.csv`, `original_test_ads` and
`retained_test_ads` count advertisements, not skill mentions. For example,
118/152 means 118 of 152 advertisements remain when every gold mention in an
advertisement must be recoverable. The corresponding Journalist and Merged
counts are 117/161 and 239/318. Separate `*_gold_mentions` columns count
individual mentions.
