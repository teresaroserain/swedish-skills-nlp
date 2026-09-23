from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
FINE_TUNED_METRICS = ROOT / "results" / "aggregate" / "all_models_all_metrics.csv"
CONTROL_METRICS = (
    ROOT / "results" / "task_unadapted_baselines" / "task_unadapted_all_metrics.csv"
)
OUTPUT_ROOT = ROOT / "results" / "before_after"

FAMILIES = ("BERT", "BART")
DOMAINS = ("it", "journal", "merged")
SKILLS = (
    "all_skills",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
)
METRICS = tuple(
    f"{level}_{aggregation}_f1"
    for level in ("exact", "containment", "lexical", "semantic")
    for aggregation in ("macro", "micro")
)


def load_aligned_metrics() -> pd.DataFrame:
    fine_tuned = pd.read_csv(FINE_TUNED_METRICS, encoding="utf-8-sig")
    fine_tuned = fine_tuned[fine_tuned["model"].isin(FAMILIES)].copy()
    control = pd.read_csv(CONTROL_METRICS, encoding="utf-8-sig")
    control["family"] = control["model"].str.replace(" task-unadapted", "", regex=False)
    fine_tuned = fine_tuned.rename(columns={"model": "family"})

    keys = ["family", "domain", "skill"]
    expected = {
        (family, domain, skill)
        for family in FAMILIES
        for domain in DOMAINS
        for skill in SKILLS
    }
    for name, frame in (("fine-tuned", fine_tuned), ("control", control)):
        actual = set(frame[keys].itertuples(index=False, name=None))
        if len(frame) != 36 or actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise ValueError(
                f"{name} metrics are incomplete: rows={len(frame)}, "
                f"missing={missing}, extra={extra}"
            )
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} metrics contain duplicate configuration keys")

    selected = [*keys, "test_rows", *METRICS]
    for frame in (fine_tuned, control):
        frame["filtered_item_rate"] = (
            frame["filtered_items"]
            .div(frame["raw_prediction_items"].replace(0, pd.NA))
            .fillna(0.0)
        )
    selected.extend(("unsupported_item_rate", "filtered_item_rate"))

    aligned = control[selected].merge(
        fine_tuned[selected],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_control", "_finetuned"),
        indicator=True,
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Control and fine-tuned configuration keys do not align")
    if not aligned["test_rows_control"].eq(aligned["test_rows_finetuned"]).all():
        raise ValueError("Control and fine-tuned test-row counts differ")

    output = aligned[keys].copy()
    output["test_rows"] = aligned["test_rows_control"].astype(int)
    for metric in (*METRICS, "unsupported_item_rate", "filtered_item_rate"):
        output[f"control_{metric}"] = aligned[f"{metric}_control"]
        output[f"finetuned_{metric}"] = aligned[f"{metric}_finetuned"]
        output[f"delta_{metric}"] = (
            aligned[f"{metric}_finetuned"] - aligned[f"{metric}_control"]
        )
    return output.sort_values(keys, kind="stable").reset_index(drop=True)


def aggregate(frame: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    value_columns = [
        column
        for column in frame.columns
        if column.startswith(("control_", "finetuned_", "delta_"))
    ]
    return (
        frame.groupby(group_by, sort=False, as_index=False)[value_columns]
        .mean()
        .sort_values(group_by, kind="stable")
        .reset_index(drop=True)
    )


def build_summary(
    configuration: pd.DataFrame,
    domain: pd.DataFrame,
    overall: pd.DataFrame,
) -> str:
    lines = [
        "EVALUATION PIPELINE TASK-UNADAPTED CONTROL VERSUS FINE-TUNED SUMMARY",
        "",
        "Primary metric: exact normalized macro F1.",
        "Controls and fine-tuned outputs use identical test IDs, gold labels,",
        "prediction rules, and the same centralized scorer.",
        "",
        "Overall mean across 18 domain-skill configurations per family:",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"- {row.family}: control={row.control_exact_macro_f1:.6f}, "
            f"fine-tuned={row.finetuned_exact_macro_f1:.6f}, "
            f"delta={row.delta_exact_macro_f1:+.6f}"
        )
    lines.extend(("", "Mean exact normalized macro F1 by domain:"))
    for row in domain.itertuples(index=False):
        lines.append(
            f"- {row.family}/{row.domain}: "
            f"control={row.control_exact_macro_f1:.6f}, "
            f"fine-tuned={row.finetuned_exact_macro_f1:.6f}, "
            f"delta={row.delta_exact_macro_f1:+.6f}"
        )
    improved = int(configuration["delta_exact_macro_f1"].gt(0).sum())
    unchanged = int(configuration["delta_exact_macro_f1"].eq(0).sum())
    lines.extend(
        (
            "",
            f"Fine-tuning improved exact macro F1 in {improved}/36 configurations; "
            f"{unchanged} were unchanged.",
            "",
            "Interpretation note:",
            "These are task-unadapted controls, not architecture-matched zero-shot",
            "systems. BERT uses a fixed random BIO classification head; BART uses",
            "the pretrained denoising checkpoint without task-specific updates.",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configuration = load_aligned_metrics()
    domain = aggregate(configuration, ["family", "domain"])
    overall = aggregate(configuration, ["family"])

    configuration.to_csv(
        OUTPUT_ROOT / "before_after_by_configuration.csv",
        index=False,
        encoding="utf-8-sig",
    )
    domain.to_csv(
        OUTPUT_ROOT / "before_after_domain_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall.to_csv(
        OUTPUT_ROOT / "before_after_overall_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_ROOT / "BEFORE_AFTER_SUMMARY.txt").write_text(
        build_summary(configuration, domain, overall), encoding="utf-8"
    )
    LOGGER.info("Wrote aligned before-after results to %s", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
