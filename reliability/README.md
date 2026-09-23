# Inter-Annotator Agreement

This directory is the preserved implementation source for the inter-annotator
agreement table in the manuscript. It was copied from
`Split_data/check_reliability` on 11 September 2026. System files and Python
bytecode caches were excluded.

## Authoritative inputs

- `software_developer_checkreliability.xlsx`: the 32 software-developer
  advertisements shared by `review1` and `review2`. The serialized hard-skill
  lists in `software_developer_analysis` match this workbook for all 64
  reviewer records.
- `journalist_checkreliability.xlsx`: 35 nonblank advertisement IDs shared by
  both reviewer sheets. Each sheet also contains two rows with blank IDs; those
  rows are excluded before scoring.
- `software_developer_checkreliability_legacy.xlsx`: an earlier annotation
  version retained for provenance. It is not used by the current scorer.

The scorer writes SHA-256 checksums and row/ID checks to
`results/input_provenance.csv`.

## Historical implementation

`build_reviewer_summary.py`, `build_journal_summary.py`,
`eda_histograms.py`, and the historical output folders are retained so the
manuscript's existing table can be audited. The historical implementation
split every comma, including commas inside matched brackets; replaced longer
nested phrases with shorter phrases; measured many-to-many coverage; and used
BM25, embedding, and hybrid connected-component merges. It did not directly
implement the exact-first one-to-one containment method described in the
manuscript.

## Current scorer

`reliability_checker.py` applies the same comparison semantics as the model
evaluation:

1. Parse commas, semicolons, and vertical bars only outside matched square
   brackets.
2. Apply Unicode NFKC, whitespace, dash, punctuation, slash, and common
   technology-spelling normalization.
3. Remove exact normalized duplicates only. Distinct nested phrases remain.
4. Assign all exact pairs with deterministic maximum-cardinality bipartite
   matching.
5. Assign contiguous-token containment pairs only among unmatched items.
6. Aggregate matches and item counts over the overlap sample and report micro
   precision, recall, and F1.

Both annotator lists are treated symmetrically as human reference sets.
Prediction-fragment stopwords, blacklists, prefixes, and allowlists are not
applied to human annotations.

Run:

```bash
python3 reliability_checker.py
python3 -m unittest -v test_reliability_scorer.py
```

The generated files are under `results/`:

- `reliability_summary.csv`: the ten domain-label agreement rows.
- `reliability_per_record.csv`: per-advertisement counts and scores.
- `manuscript_table_rows.tex`: replacement LaTeX table rows.
- `manuscript_table_comparison.csv`: current manuscript values versus the
  recalculated values.
- `input_provenance.csv`: workbook checksums and ID validation.

Pairwise mention-set F1 is an observed-overlap measure, not a chance-corrected
agreement coefficient such as Cohen's kappa. The manuscript should therefore
name the measure explicitly and should not apply kappa interpretation bands to
these values.
