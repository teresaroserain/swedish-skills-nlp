from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReviewLabel = Literal["review1", "review2"]
MetricMode = Literal["base", "merge"]

CHECK_RELIABILITY_DIR = Path(__file__).resolve().parent
JOURNAL_DIR = CHECK_RELIABILITY_DIR / "journalist_analysis"
OUTPUT_PATH = JOURNAL_DIR / "journalist.csv"

LABEL_TYPES: tuple[str, ...] = (
    "hard_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
    "soft_skills",
)

METHOD_ORDER: tuple[str, ...] = (
    "exact_only",
    "bm25_only",
    "embed_only",
    "hybrid",
)

BASE_METHOD_METRICS: Mapping[str, str] = {
    "exact_only": "exact_ratio",
    "bm25_only": "bm25_ratio",
    "embed_only": "embed_ratio",
    "hybrid": "hybrid_ratio",
}

MERGE_METHOD_METRICS: Mapping[str, str] = {
    "exact_only": "merge_exact_ratio",
    "bm25_only": "merge_bm25_ratio",
    "embed_only": "merge_embed_ratio",
    "hybrid": "merge_hybrid_ratio",
}

BASE_PER_ROW_COLUMNS: Mapping[str, tuple[str, str, str, str]] = {
    "exact_ratio": ("exact_ratio_p1", "exact_ratio_p2", "p1_count", "p2_count"),
    "bm25_ratio": ("bm25_ratio_p1", "bm25_ratio_p2", "p1_count", "p2_count"),
    "embed_ratio": ("embed_ratio_p1", "embed_ratio_p2", "p1_count", "p2_count"),
    "hybrid_ratio": ("hybrid_ratio_p1", "hybrid_ratio_p2", "p1_count", "p2_count"),
}

MERGE_PER_ROW_COLUMNS: Mapping[str, tuple[str, str, str, str]] = {
    "merge_exact_ratio": (
        "ratio_p1_merge_exact",
        "ratio_p2_merge_exact",
        "p1_merge_count_exact",
        "p2_merge_count_exact",
    ),
    "merge_bm25_ratio": (
        "ratio_p1_merge_bm25",
        "ratio_p2_merge_bm25",
        "p1_merge_count_bm25",
        "p2_merge_count_bm25",
    ),
    "merge_embed_ratio": (
        "ratio_p1_merge_embed",
        "ratio_p2_merge_embed",
        "p1_merge_count_embed",
        "p2_merge_count_embed",
    ),
    "merge_hybrid_ratio": (
        "ratio_p1_merge_hybrid",
        "ratio_p2_merge_hybrid",
        "p1_merge_count_hybrid",
        "p2_merge_count_hybrid",
    ),
}

OUTPUT_COLUMNS: tuple[str, ...] = (
    "label_type",
    "method",
    "overlap_rows",
    "exact_first",
    "lexical_guard",
    "bm25_threshold",
    "embed_threshold",
    "hybrid_threshold",
    "alpha_embed",
    "macro_precision_avg",
    "macro_recall_avg",
    "macro_f1_avg",
    "micro_exact",
    "micro_extra",
    "micro_matched_total",
    "micro_total_p1_items",
    "micro_total_p2_items",
    "micro_precision",
    "micro_recall",
    "micro_f1",
)


@dataclass(frozen=True)
class ReviewMetric:
    avg: float
    total_items: int
    overlap_total: int


@dataclass(frozen=True)
class MethodMetric:
    precision: ReviewMetric
    recall: ReviewMetric


@dataclass(frozen=True)
class LabelMeta:
    overlap_rows: int
    lexical_guard: str
    bm25_threshold: str
    embed_threshold: str
    hybrid_threshold: str
    alpha_embed: str


def main() -> None:
    rows = build_summary_rows(JOURNAL_DIR, metric_mode="base")
    write_rows(OUTPUT_PATH, rows)
    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")


def build_summary_rows(
    journal_dir: Path,
    *,
    metric_mode: MetricMode = "base",
) -> list[dict[str, str]]:
    method_metrics = (
        BASE_METHOD_METRICS if metric_mode == "base" else MERGE_METHOD_METRICS
    )

    output_rows: list[dict[str, str]] = []
    for label_type in LABEL_TYPES:
        label_dir = journal_dir / label_type
        avg_path = label_dir / "avg_by_label.csv"
        metrics = read_avg_by_label(avg_path, label_type=label_type)
        meta = read_label_meta(label_dir)

        exact_metric = metrics[method_metrics["exact_only"]]
        exact_total = exact_metric.precision.overlap_total

        for method in METHOD_ORDER:
            metric_name = method_metrics[method]
            metric = metrics[metric_name]
            macro_precision = metric.precision.avg
            macro_recall = metric.recall.avg
            macro_f1 = read_macro_f1(label_dir, metric_name, metric_mode)
            if macro_f1 is None:
                macro_f1 = f1_score(macro_precision, macro_recall)

            micro_precision = divide(
                metric.precision.overlap_total,
                metric.precision.total_items,
            )
            micro_recall = divide(
                metric.recall.overlap_total,
                metric.recall.total_items,
            )
            micro_f1 = f1_score(micro_precision, micro_recall)

            matched_total = metric.precision.overlap_total

            output_rows.append(
                {
                    "label_type": label_type,
                    "method": method,
                    "overlap_rows": str(meta.overlap_rows),
                    "exact_first": "True",
                    "lexical_guard": meta.lexical_guard,
                    "bm25_threshold": meta.bm25_threshold,
                    "embed_threshold": meta.embed_threshold,
                    "hybrid_threshold": meta.hybrid_threshold,
                    "alpha_embed": meta.alpha_embed,
                    "macro_precision_avg": number_to_text(macro_precision),
                    "macro_recall_avg": number_to_text(macro_recall),
                    "macro_f1_avg": number_to_text(macro_f1),
                    "micro_exact": str(exact_total),
                    "micro_extra": str(max(matched_total - exact_total, 0)),
                    "micro_matched_total": str(matched_total),
                    "micro_total_p1_items": str(metric.precision.total_items),
                    "micro_total_p2_items": str(metric.recall.total_items),
                    "micro_precision": number_to_text(micro_precision),
                    "micro_recall": number_to_text(micro_recall),
                    "micro_f1": number_to_text(micro_f1),
                }
            )

    return output_rows


def read_avg_by_label(path: Path, *, label_type: str) -> dict[str, MethodMetric]:
    rows = read_csv_rows(path)
    grouped: dict[str, dict[ReviewLabel, ReviewMetric]] = {}

    for row in rows:
        row_label_type = row.get("label_type", "").strip()
        if row_label_type and row_label_type != label_type:
            raise ValueError(
                f"{path} has label_type={row_label_type!r}, expected {label_type!r}"
            )

        metric = required_value(row, "metric", path)
        review_label = parse_review_label(required_value(row, "label", path), path)
        grouped.setdefault(metric, {})[review_label] = ReviewMetric(
            avg=parse_float(required_value(row, "avg", path), path),
            total_items=parse_int(required_value(row, "total_items", path), path),
            overlap_total=parse_int(required_value(row, "overlap_total", path), path),
        )

    output: dict[str, MethodMetric] = {}
    for metric, by_label in grouped.items():
        if "review1" not in by_label or "review2" not in by_label:
            raise ValueError(f"{path} is missing review1/review2 rows for {metric}")
        output[metric] = MethodMetric(
            precision=by_label["review1"],
            recall=by_label["review2"],
        )

    required_metrics = set(BASE_METHOD_METRICS.values()) | set(
        MERGE_METHOD_METRICS.values()
    )
    missing_metrics = sorted(required_metrics - set(output))
    if missing_metrics:
        raise ValueError(f"{path} is missing metrics: {', '.join(missing_metrics)}")

    return output


def read_label_meta(label_dir: Path) -> LabelMeta:
    summary_path = label_dir / "summary.csv"
    if summary_path.is_file():
        row = first_row(summary_path)
        return LabelMeta(
            overlap_rows=parse_int(
                required_value(row, "overlap_rows", summary_path),
                summary_path,
            ),
            lexical_guard=required_value(row, "lexical_guard", summary_path),
            bm25_threshold=required_value(row, "bm25_threshold", summary_path),
            embed_threshold=required_value(row, "embed_threshold", summary_path),
            hybrid_threshold=required_value(row, "hybrid_threshold", summary_path),
            alpha_embed=required_value(row, "alpha", summary_path),
        )

    per_row_path = label_dir / "per_row.csv"
    return LabelMeta(
        overlap_rows=len(read_csv_rows(per_row_path)) if per_row_path.is_file() else 0,
        lexical_guard="",
        bm25_threshold="",
        embed_threshold="",
        hybrid_threshold="",
        alpha_embed="",
    )


def read_macro_f1(
    label_dir: Path,
    metric_name: str,
    metric_mode: MetricMode,
) -> float | None:
    per_row_path = label_dir / "per_row.csv"
    if not per_row_path.is_file():
        return None

    column_map = (
        BASE_PER_ROW_COLUMNS if metric_mode == "base" else MERGE_PER_ROW_COLUMNS
    )
    if metric_name not in column_map:
        return None

    precision_col, recall_col, p1_count_col, p2_count_col = column_map[metric_name]
    f1_values: list[float] = []
    for row in read_csv_rows(per_row_path):
        p1_count = parse_int(row.get(p1_count_col, "0"), per_row_path)
        p2_count = parse_int(row.get(p2_count_col, "0"), per_row_path)
        if p1_count == 0 and p2_count == 0:
            continue

        precision = parse_float(row.get(precision_col, "0"), per_row_path)
        recall = parse_float(row.get(recall_col, "0"), per_row_path)
        f1_values.append(f1_score(precision, recall))

    if not f1_values:
        return None

    return sum(f1_values) / len(f1_values)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows[0]


def write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_review_label(value: str, path: Path) -> ReviewLabel:
    if value == "review1" or value == "review2":
        return value
    raise ValueError(f"{path} has unsupported review label: {value!r}")


def required_value(row: Mapping[str, str], field: str, path: Path) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"{path} is missing a value for {field!r}")
    return value.strip()


def parse_float(value: str, path: Path) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{path} has invalid float value: {value!r}") from error


def parse_int(value: str, path: Path) -> int:
    try:
        return int(float(value))
    except ValueError as error:
        raise ValueError(f"{path} has invalid integer value: {value!r}") from error


def divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def number_to_text(value: float) -> str:
    return str(value)


if __name__ == "__main__":
    main()
