from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Literal

ReviewLabel = Literal["review1", "review2"]

CHECK_RELIABILITY_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = CHECK_RELIABILITY_DIR / "software_developer_analysis"
DEFAULT_OUTPUT_PATH = DEFAULT_ROOT_DIR / "software_developer_reviewer_form.csv"

LABEL_TYPES: tuple[str, ...] = (
    "hard_skills",
    "soft_skills",
    "must_have",
    "nice_to_have",
    "distinct_skills",
)

LABEL_DISPLAY_NAMES: Mapping[str, str] = {
    "hard_skills": "Hard skill",
    "soft_skills": "Soft skill",
    "must_have": "Must have skill",
    "nice_to_have": "Nice to have skill",
    "distinct_skills": "Distinct skill",
}

METRIC_CHOICES: tuple[str, ...] = (
    "exact_ratio",
    "bm25_ratio",
    "embed_ratio",
    "hybrid_ratio",
    "merge_exact_ratio",
    "merge_bm25_ratio",
    "merge_embed_ratio",
    "merge_hybrid_ratio",
)

OUTPUT_COLUMNS: tuple[str, ...] = ("Cột 1", "Reviewer 1", "Reviewer 2", "Ghi chú")

NOTES: tuple[str, str, str] = (
    "Total skill count là tập các skill (đã deduplicate)",
    "total overlap là tập skill overlap (đã deduplicate)",
    "Avg overlap ratio là trung bình tỉ lệ overlap trên toàn bộ sample",
)


@dataclass(frozen=True)
class ReviewerResult:
    avg: float
    total_items: int
    overlap_total: int


@dataclass(frozen=True)
class LabelResult:
    reviewer_1: ReviewerResult
    reviewer_2: ReviewerResult


@dataclass(frozen=True)
class CliArgs:
    root_dir: Path
    output_path: Path
    metric: str
    decimal_places: int


def main() -> None:
    args = parse_args()
    rows = build_rows(
        root_dir=args.root_dir,
        metric=args.metric,
        decimal_places=args.decimal_places,
    )
    write_rows(args.output_path, rows)
    print(f"Saved {len(rows)} rows to {args.output_path}")


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description="Build reviewer summary CSV in the screenshot-style format."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Folder containing label subfolders with avg_by_label.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_CHOICES,
        default="merge_hybrid_ratio",
        help="Metric row to aggregate from avg_by_label.csv.",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=2,
        help="Decimal places for avg overlap ratios.",
    )
    namespace = parser.parse_args()

    if namespace.decimal_places < 0:
        raise ValueError("--decimal-places must be zero or greater")

    return CliArgs(
        root_dir=namespace.root_dir,
        output_path=namespace.output,
        metric=namespace.metric,
        decimal_places=namespace.decimal_places,
    )


def build_rows(
    *,
    root_dir: Path,
    metric: str,
    decimal_places: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for label_index, label_type in enumerate(LABEL_TYPES):
        label_dir = root_dir / label_type
        result = read_label_result(label_dir / "avg_by_label.csv", metric=metric)
        display_name = LABEL_DISPLAY_NAMES[label_type]
        notes = NOTES if label_index == 0 else ("", "", "")

        rows.extend(
            [
                {
                    "Cột 1": f"{display_name} avg overlap ratio",
                    "Reviewer 1": format_ratio(
                        result.reviewer_1.avg,
                        decimal_places=decimal_places,
                    ),
                    "Reviewer 2": format_ratio(
                        result.reviewer_2.avg,
                        decimal_places=decimal_places,
                    ),
                    "Ghi chú": notes[0],
                },
                {
                    "Cột 1": f"{display_name} total skill count",
                    "Reviewer 1": str(result.reviewer_1.total_items),
                    "Reviewer 2": str(result.reviewer_2.total_items),
                    "Ghi chú": notes[1],
                },
                {
                    "Cột 1": f"{display_name} total overlap",
                    "Reviewer 1": str(result.reviewer_1.overlap_total),
                    "Reviewer 2": str(result.reviewer_2.overlap_total),
                    "Ghi chú": notes[2],
                },
            ]
        )

        if label_index < len(LABEL_TYPES) - 1:
            rows.append(empty_row())

    return rows


def read_label_result(path: Path, *, metric: str) -> LabelResult:
    rows = read_csv_rows(path)
    by_reviewer: dict[ReviewLabel, ReviewerResult] = {}

    for row in rows:
        if row.get("metric", "").strip() != metric:
            continue

        review_label = parse_review_label(required_value(row, "label", path), path)
        by_reviewer[review_label] = ReviewerResult(
            avg=parse_float(required_value(row, "avg", path), path),
            total_items=parse_int(required_value(row, "total_items", path), path),
            overlap_total=parse_int(required_value(row, "overlap_total", path), path),
        )

    if "review1" not in by_reviewer or "review2" not in by_reviewer:
        raise ValueError(f"{path} is missing review1/review2 rows for {metric}")

    return LabelResult(
        reviewer_1=by_reviewer["review1"],
        reviewer_2=by_reviewer["review2"],
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


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


def format_ratio(value: float, *, decimal_places: int) -> str:
    quantizer = Decimal(1).scaleb(-decimal_places)
    decimal_value = Decimal(str(value)).quantize(quantizer, rounding=ROUND_DOWN)
    return format(decimal_value, "f").rstrip("0").rstrip(".")


def empty_row() -> dict[str, str]:
    return {column: "" for column in OUTPUT_COLUMNS}


if __name__ == "__main__":
    main()
