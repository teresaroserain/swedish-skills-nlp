EVALUATION PIPELINE ALL-MODEL EVALUATION

Scope
- 36 fine-tuned BERT/BART runs and 54 aligned GPT/Gemini/Claude configurations.
- One canonical prompt-safe seed-13 40/10/50 test split per domain-skill configuration.
- Exact global one-to-one matching first, then token-containment, lexical, and semantic diagnostics.
- Exact deduplication only; no long-to-short skill replacement.

Overall mean containment macro F1 across 18 configurations
- BERT: 0.659448
- Gemini: 0.618897
- Claude: 0.595449
- GPT: 0.590053
- BART: 0.497656

Overall mean exact macro F1 across 18 configurations
- BERT: 0.440474
- Gemini: 0.390891
- GPT: 0.368674
- Claude: 0.347976
- BART: 0.333892

Domain winners under the manuscript's containment metric
- it: Gemini (0.651849)
- journal: BERT (0.687245)
- merged: BERT (0.669905)

Claim audit
- SUPPORTED: Gemini leads software developers; BERT leads journalism and merged
- SUPPORTED: BERT leads 13 of 18 domain-skill cells and prompted models lead five
- SUPPORTED: BART does not lead any domain-skill cell
- SUPPORTED: BERT leads all-skills, hard-skills, and soft-skills in every domain
- SUPPORTED: GPT does not lead an individual domain-skill cell
- SUPPORTED: Prompted-model wins are concentrated on distinct and nice-to-have skills
- SUPPORTED: The best-model distinct-skill score is the lowest best-model skill score in every domain
- SUPPORTED: The best-model nice-to-have score is the highest best-model skill score in every domain
- SUPPORTED: Distinct skills are lowest after averaging all five models within every domain
- SUPPORTED: The exact-to-containment gap is largest in journalism
- SUPPORTED: BERT outperforms BART after fine-tuning in every domain mean
- SUPPORTED: The domain winner pattern is stable across exact, containment, lexical, and semantic metrics
- SUPPORTED: Best domain-level model scores remain below the lower human containment-agreement bound of 0.714
- SUPPORTED: Fine-tuning substantially improves BERT and BART over task-unadapted controls

Fairness and reproducibility limits
- Final arithmetic scoring is common across all five systems and every run uses exact canonical test IDs.
- Exact normalized macro F1 is the HPO objective. Boundary-relaxed containment macro F1 is the primary cross-configuration reporting metric, with exact results reported alongside it; lexical and semantic scores are diagnostics.
- Five IT-all test IDs occurred in prompt development; see prompt_overlap_sensitivity.csv.
- The LLM source-text audit found 22 unique IDs (126 model-domain-ID rows) with tokenization differences, all but 0 compact-equal after removing whitespace and punctuation; see llm_source_text_mismatches.csv.
- LLM provider snapshots, raw API responses, retries, and original format compliance are not recoverable from the retained parsed workbooks.
- Supervised fine-tuning and prompted inference are different adaptation/resource regimes, not a controlled architecture comparison.
- All 36 task-unadapted controls are available and validated in results/before_after; fine-tuning improves exact and containment macro F1 in every configuration.

Files
- all_models_all_metrics.csv: primary 90-row model/domain/skill table.
- all_models_per_ad_metrics.csv: advertisement-level audit table.
- domain_model_means.csv and skill_model_means.csv: aggregate tables.
- domain_skill_means.csv: skill difficulty averaged across models within each domain.
- skill_winners_containment.csv and skill_winners_exact.csv: cell winners.
- skill_winners_all_metrics.csv, domain_winners_all_metrics.csv, and winner_counts_all_metrics.csv: winner sensitivity across all four matching levels.
- claim_audit.csv: manuscript conclusion checks.
- fairness_audit.csv: metric validity and comparability assessment.
- manuscript_method_consistency.csv: executed-code versus manuscript differences.
- neural_recalculation_audit.csv: independent reproduction of the 36 stored BERT/BART metrics.
- alignment_audit.csv: test-ID and source coverage checks.
- llm_source_text_mismatches.csv: non-token-identical LLM input-text audit.
