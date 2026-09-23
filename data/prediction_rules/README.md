# Auditable Prediction Rules

This directory records the evidence and deterministic procedure used to build
the four prediction-filter variables in the reported experiments. The rules
are fixed before test inference and use no test examples or test labels.

## Rule Definitions

- `PREDICTED_SKILL_ALLOWLIST`: compact expressions that occur as exact gold
  labels in the corresponding train or validation partition. These items are
  exempt from stop-word and fragment rejection. This protects valid labels
  such as `R`, `C`, `AI`, `.NET`, and `C++` without consulting the test set.
- `PREDICTED_SKILL_STOPWORDS`: the union of the official Snowball Swedish and
  English stop-word lists, minus the development-derived allowlist. A rule is
  applied only when the entire prediction is one stop word.
- `PREDICTED_SKILL_FRAGMENT_BLACKLIST`: suffixes listed by the official
  Snowball stemmers that are continuation-only tokens in both frozen KBLab
  tokenizers, occur attached to words at least five times in train and
  validation text, and occur neither as standalone development words nor as
  exact development gold labels. A rule is applied only when the entire
  prediction is one such fragment.
- `PREDICTED_SKILL_FRAGMENT_PREFIXES`: intentionally empty. The available
  external resources do not establish that a complete multi-token prediction
  is invalid merely because its first token resembles a suffix.

The generic legacy rule that rejected every one- or two-character prediction
is disabled. Short predictions are retained unless a source-backed stop-word
rule applies; annotated short development labels are explicitly allowlisted.

## Evidence Sources

- [Snowball Swedish stop-word list](https://snowballstem.org/algorithms/swedish/stop.txt)
- [Snowball English stop-word list](https://snowballstem.org/algorithms/english/stop.txt)
- [Snowball Swedish stemming algorithm](https://snowballstem.org/algorithms/swedish/stemmer.html)
- [Snowball English stemming algorithm](https://snowballstem.org/algorithms/english/stemmer.html)
- [KBLab Swedish BERT tokenizer vocabulary](https://huggingface.co/KBLab/bert-base-swedish-cased/blob/main/vocab.txt)
- [KBLab Swedish BART tokenizer artifact](https://huggingface.co/KBLab/bart-base-swedish-cased/blob/main/tokenizer.json)
- Task annotations from the frozen seed-13 train and validation JSONL files.

`source_manifest.json` records the downloaded artifact URLs, SHA-256 hashes,
file sizes, resolved model revisions, and derivation constants.
`prediction_rule_evidence.csv` gives row-level evidence for every retained or
rejected term. `prediction_rule_summary.csv` gives counts by domain and skill
type. The 18 generated JSON files are the exact rule sets embedded into both
model families.

## Evaluation Policy

`canonicalize_skills()` performs normalization-aware exact deduplication only.
It does not replace a longer skill with a shorter nested expression.

The HPO objective is exact normalized macro F1. Matching is global and
one-to-one, with all exact matches allocated before any secondary match. The
central scorer also reports contiguous-token containment, lexical-relaxed, and
semantic matches; none of these secondary matching levels contributes to the
HPO objective. Raw predictions, filtered predictions, removed predictions, and
unsupported predictions are retained in the test output for audit.

## Reproduction

Run from the repository root:

```bash
python3 src/prediction_rules/build_prediction_rules.py
```

The 36 frozen training scripts that consume these rule sets are under
`src/training/skill_training_scripts/`.
