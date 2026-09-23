# Task-unadapted controls

These controls change only supervised task adaptation:

- BERT loads `KBLab/bert-base-swedish-cased` with a three-label BIO head
  initialized under seed 13 and performs no training.
- BART loads `KBLab/bart-base-swedish-cased` and decodes without task
  fine-tuning.

Both families use the canonical test JSONL files, the same
`headline + "\n" + description` input construction, family-specific chunking,
word-boundary expansion for BERT, BART decoding limits, and frozen
train/validation-derived prediction rules used by the fine-tuned systems.
Nested skills are not collapsed. Training and validation partitions are never
read.

The runner produces prediction ZIPs only. The central evaluator recomputes
exact, containment, lexical, and semantic metrics after test-ID, gold-label,
metadata, post-processing, and unsupported-generation checks.

These outputs are task-unadapted controls, not zero-shot BERT/BART systems.
They support within-family before-versus-after task-adaptation analysis. They do
not make the comparison with prompted commercial LLMs architecture-only.
