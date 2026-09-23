from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent

SCORER_VERSION = "exact_first_matching"
SHEET_1 = "review1"
SHEET_2 = "review2"
ID_COLUMN = "id"

LABEL_COLUMNS: tuple[str, ...] = (
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
)

DOMAIN_DISPLAY = {
    "software_developers": "Software developers",
    "journalists": "Journalists",
}

LABEL_DISPLAY = {
    "hard_skills": "Hard skills",
    "soft_skills": "Soft skills",
    "distinct_skills": "Distinct skills",
    "must_have": "Must-have skills",
    "nice_to_have": "Nice-to-have skills",
}

XML_NAMESPACES = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office_rel": (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ),
}


@dataclass(frozen=True)
class WorkbookConfig:
    domain: str
    path: Path


@dataclass(frozen=True)
class WorkbookAudit:
    domain: str
    workbook: str
    sha256: str
    reviewer1_rows: int
    reviewer2_rows: int
    reviewer1_blank_id_rows: int
    reviewer2_blank_id_rows: int
    valid_overlap_ids: int
    exact_id_set_match: bool


@dataclass(frozen=True)
class MatchCounts:
    annotator1_count: int
    annotator2_count: int
    exact_tp: int
    containment_extra_tp: int


@dataclass(frozen=True)
class SummaryCounts:
    records: int
    annotator1_count: int
    annotator2_count: int
    exact_tp: int
    containment_extra_tp: int


@dataclass(frozen=True)
class CliArgs:
    output_dir: Path
    manuscript_tex: Path | None


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate pairwise inter-annotator mention-set agreement with the "
            "evaluation pipeline normalization and exact-first one-to-one matcher."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Destination for per-record, summary, provenance, and LaTeX outputs.",
    )
    parser.add_argument(
        "--manuscript-tex",
        type=Path,
        default=default_manuscript_path(),
        help=(
            "Current manuscript source used to build a table-value comparison. "
            "The comparison is omitted when the file does not exist."
        ),
    )
    namespace = parser.parse_args()
    manuscript = namespace.manuscript_tex
    if manuscript is not None and not manuscript.is_file():
        LOGGER.warning(
            "Manuscript source not found; comparison will be omitted: %s", manuscript
        )
        manuscript = None
    return CliArgs(output_dir=namespace.output_dir, manuscript_tex=manuscript)


def default_manuscript_path() -> Path | None:
    candidates = (ROOT.parent / "paper" / "source_snapshot" / "cas-sc-template.tex",)
    return next((path for path in candidates if path.is_file()), None)


def workbook_configs() -> tuple[WorkbookConfig, ...]:
    return (
        WorkbookConfig(
            domain="software_developers",
            path=ROOT / "software_developer_checkreliability.xlsx",
        ),
        WorkbookConfig(
            domain="journalists",
            path=ROOT / "journalist_checkreliability.xlsx",
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def shared_strings(archive: ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read(path))
    return [
        "".join(text.text or "" for text in item.iterfind(".//main:t", XML_NAMESPACES))
        for item in root.findall("main:si", XML_NAMESPACES)
    ]


def sheet_archive_path(archive: ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships
    }
    sheet = next(
        (
            item
            for item in workbook.findall("main:sheets/main:sheet", XML_NAMESPACES)
            if item.attrib.get("name") == sheet_name
        ),
        None,
    )
    if sheet is None:
        raise ValueError(f"Workbook is missing sheet {sheet_name!r}")
    relationship_id = sheet.attrib[f"{{{XML_NAMESPACES['office_rel']}}}id"]
    target = targets[relationship_id]
    if target.startswith("/"):
        return target.lstrip("/")
    return (PurePosixPath("xl") / target).as_posix()


def read_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        strings = shared_strings(archive)
        sheet_path = sheet_archive_path(archive, sheet_name)
        root = ElementTree.fromstring(archive.read(sheet_path))

    matrix: list[list[str]] = []
    for row in root.findall("main:sheetData/main:row", XML_NAMESPACES):
        values: dict[int, str] = {}
        for cell in row.findall("main:c", XML_NAMESPACES):
            reference = cell.attrib.get("r", "")
            index = column_index(reference)
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(
                    text.text or ""
                    for text in cell.iterfind(".//main:t", XML_NAMESPACES)
                )
            else:
                value_node = cell.find("main:v", XML_NAMESPACES)
                raw_value = "" if value_node is None else value_node.text or ""
                if cell_type == "s" and raw_value:
                    value = strings[int(raw_value)]
                else:
                    value = raw_value
            values[index] = value
        if values:
            matrix.append([values.get(index, "") for index in range(max(values) + 1)])

    if not matrix:
        raise ValueError(f"Sheet {sheet_name!r} in {path} is empty")
    headers = [header.strip() for header in matrix[0]]
    return [
        {
            header: row[index] if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        for row in matrix[1:]
    ]


def index_nonblank_ids(
    rows: Sequence[Mapping[str, str]],
    *,
    workbook: Path,
    sheet_name: str,
) -> tuple[dict[str, Mapping[str, str]], int]:
    indexed: dict[str, Mapping[str, str]] = {}
    blank_rows = 0
    for row in rows:
        record_id = str(row.get(ID_COLUMN, "")).strip()
        if not record_id:
            blank_rows += 1
            continue
        if record_id in indexed:
            raise ValueError(
                f"Duplicate nonblank ID {record_id!r} in {workbook.name}/{sheet_name}"
            )
        indexed[record_id] = row
    return indexed, blank_rows


def normalize_annotation_cell(value: str) -> str:
    text = str(value or "")
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    text = text.replace('"', "")
    return re.sub(r"\s+", " ", text).strip()


def split_bracketed_delimiters(value: str) -> list[str]:
    text = normalize_annotation_cell(value)
    if not text:
        return []

    matched_open: set[int] = set()
    matched_close: set[int] = set()
    stack: list[int] = []
    for index, character in enumerate(text):
        if character == "[":
            stack.append(index)
        elif character == "]" and stack:
            matched_open.add(stack.pop())
            matched_close.add(index)

    output: list[str] = []
    buffer: list[str] = []
    depth = 0
    for index, character in enumerate(text):
        if character == "[":
            if index in matched_open:
                depth += 1
            continue
        if character == "]":
            if index in matched_close:
                depth = max(depth - 1, 0)
            continue
        if depth == 0 and character in {",", ";", "|"}:
            item = "".join(buffer).strip()
            if item:
                output.append(item)
            buffer = []
        else:
            buffer.append(character)

    final_item = "".join(buffer).strip()
    if final_item:
        output.append(final_item)
    return [re.sub(r"\s+", " ", item).strip() for item in output if item.strip()]


def norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    return " ".join(text.strip().lower().split())


def normalize_skill_surface(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    text = " ".join(text.strip().split())
    text = re.sub(r"^[\-*\u2022\u00b7]+\s*", "", text)
    text = text.strip(" \t\r\n,;:|")
    text = re.sub(r"^[\"'`\u00b4\u2018\u2019\u201c\u201d\(\[\{]+", "", text)
    text = re.sub(r"[\"'`\u00b4\u2018\u2019\u201c\u201d\)\]\}]+$", "", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*#\s*", "#", text)
    text = re.sub(r"\bC\s*\+\s*\+", "C++", text, flags=re.IGNORECASE)
    text = re.sub(r"\bF\s*#", "F#", text, flags=re.IGNORECASE)
    text = re.sub(r"\bC\s*#", "C#", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCI\s*/\s*CD\b", "CI/CD", text, flags=re.IGNORECASE)
    text = re.sub(r"\.\s+(?=[A-Za-z])", ".", text)
    text = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = " ".join(text.split())

    while text.endswith(".") and norm_key(text) != ".net":
        text = text[:-1].rstrip()
    if norm_key(text) == "net":
        return ".NET"
    return text


def unique_normalized_items(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_skill_surface(value)
        key = norm_key(normalized)
        if key and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def tokenize_for_match(value: str) -> list[str]:
    text = re.sub(r"[^\w]+", " ", norm_key(value), flags=re.UNICODE)
    text = " ".join(text.split())
    return text.split() if text else []


def is_contiguous_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    length = len(short)
    return any(
        list(long[index : index + length]) == list(short)
        for index in range(len(long) - length + 1)
    )


def maximum_cardinality_pairs(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    eligible: Callable[[int, int], bool],
) -> list[tuple[int, int]]:
    adjacency = {
        left: [right for right in right_indices if eligible(left, right)]
        for left in left_indices
    }
    right_to_left: dict[int, int] = {}

    def augment(left: int, visited: set[int]) -> bool:
        for right in adjacency[left]:
            if right in visited:
                continue
            visited.add(right)
            previous_left = right_to_left.get(right)
            if previous_left is None or augment(previous_left, visited):
                right_to_left[right] = left
                return True
        return False

    for left in left_indices:
        augment(left, set())
    return sorted((left, right) for right, left in right_to_left.items())


def match_counts(annotator1: Sequence[str], annotator2: Sequence[str]) -> MatchCounts:
    left_items = unique_normalized_items(annotator1)
    right_items = unique_normalized_items(annotator2)
    left_keys = [norm_key(item) for item in left_items]
    right_keys = [norm_key(item) for item in right_items]
    left_tokens = [tokenize_for_match(item) for item in left_items]
    right_tokens = [tokenize_for_match(item) for item in right_items]

    remaining_left = list(range(len(left_items)))
    remaining_right = list(range(len(right_items)))
    exact_pairs = maximum_cardinality_pairs(
        remaining_left,
        remaining_right,
        lambda left, right: left_keys[left] == right_keys[right],
    )
    used_left = {left for left, _ in exact_pairs}
    used_right = {right for _, right in exact_pairs}
    remaining_left = [index for index in remaining_left if index not in used_left]
    remaining_right = [index for index in remaining_right if index not in used_right]

    containment_pairs = maximum_cardinality_pairs(
        remaining_left,
        remaining_right,
        lambda left, right: (
            is_contiguous_subsequence(left_tokens[left], right_tokens[right])
            or is_contiguous_subsequence(right_tokens[right], left_tokens[left])
        ),
    )
    return MatchCounts(
        annotator1_count=len(left_items),
        annotator2_count=len(right_items),
        exact_tp=len(exact_pairs),
        containment_extra_tp=len(containment_pairs),
    )


def prf(true_positives: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_workbook(
    config: WorkbookConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], WorkbookAudit]:
    reviewer1_rows = read_sheet(config.path, SHEET_1)
    reviewer2_rows = read_sheet(config.path, SHEET_2)
    reviewer1, reviewer1_blank = index_nonblank_ids(
        reviewer1_rows,
        workbook=config.path,
        sheet_name=SHEET_1,
    )
    reviewer2, reviewer2_blank = index_nonblank_ids(
        reviewer2_rows,
        workbook=config.path,
        sheet_name=SHEET_2,
    )
    if set(reviewer1) != set(reviewer2):
        missing_from_reviewer1 = sorted(set(reviewer2) - set(reviewer1))
        missing_from_reviewer2 = sorted(set(reviewer1) - set(reviewer2))
        raise ValueError(
            f"ID sets differ for {config.domain}: "
            f"missing from reviewer1={missing_from_reviewer1}; "
            f"missing from reviewer2={missing_from_reviewer2}"
        )

    record_ids = sorted(reviewer1)
    per_record: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for label in LABEL_COLUMNS:
        label_counts: list[MatchCounts] = []
        for record_id in record_ids:
            left = split_bracketed_delimiters(reviewer1[record_id].get(label, ""))
            right = split_bracketed_delimiters(reviewer2[record_id].get(label, ""))
            counts = match_counts(left, right)
            label_counts.append(counts)
            exact_precision, exact_recall, exact_f1 = prf(
                counts.exact_tp,
                counts.annotator1_count,
                counts.annotator2_count,
            )
            containment_tp = counts.exact_tp + counts.containment_extra_tp
            containment_precision, containment_recall, containment_f1 = prf(
                containment_tp,
                counts.annotator1_count,
                counts.annotator2_count,
            )
            per_record.append(
                {
                    "scorer_version": SCORER_VERSION,
                    "domain": config.domain,
                    "record_id": record_id,
                    "label": label,
                    "annotator1_count": counts.annotator1_count,
                    "annotator2_count": counts.annotator2_count,
                    "exact_tp": counts.exact_tp,
                    "containment_extra_tp": counts.containment_extra_tp,
                    "exact_precision": exact_precision,
                    "exact_recall": exact_recall,
                    "exact_f1": exact_f1,
                    "containment_precision": containment_precision,
                    "containment_recall": containment_recall,
                    "containment_f1": containment_f1,
                }
            )

        totals = SummaryCounts(
            records=len(record_ids),
            annotator1_count=sum(item.annotator1_count for item in label_counts),
            annotator2_count=sum(item.annotator2_count for item in label_counts),
            exact_tp=sum(item.exact_tp for item in label_counts),
            containment_extra_tp=sum(
                item.containment_extra_tp for item in label_counts
            ),
        )
        exact_precision, exact_recall, exact_f1 = prf(
            totals.exact_tp,
            totals.annotator1_count,
            totals.annotator2_count,
        )
        containment_tp = totals.exact_tp + totals.containment_extra_tp
        containment_precision, containment_recall, containment_f1 = prf(
            containment_tp,
            totals.annotator1_count,
            totals.annotator2_count,
        )
        summary.append(
            {
                "scorer_version": SCORER_VERSION,
                "domain": config.domain,
                "domain_display": DOMAIN_DISPLAY[config.domain],
                "label": label,
                "label_display": LABEL_DISPLAY[label],
                "records": totals.records,
                "annotator1_count": totals.annotator1_count,
                "annotator2_count": totals.annotator2_count,
                "exact_tp": totals.exact_tp,
                "containment_extra_tp": totals.containment_extra_tp,
                "exact_precision": exact_precision,
                "exact_recall": exact_recall,
                "exact_micro_f1": exact_f1,
                "containment_precision": containment_precision,
                "containment_recall": containment_recall,
                "containment_micro_f1": containment_f1,
            }
        )

    audit = WorkbookAudit(
        domain=config.domain,
        workbook=config.path.name,
        sha256=sha256(config.path),
        reviewer1_rows=len(reviewer1_rows),
        reviewer2_rows=len(reviewer2_rows),
        reviewer1_blank_id_rows=reviewer1_blank,
        reviewer2_blank_id_rows=reviewer2_blank,
        valid_overlap_ids=len(record_ids),
        exact_id_set_match=set(reviewer1) == set(reviewer2),
    )
    return per_record, summary, audit


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manuscript_rows(
    path: Path,
    summary_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "% Generated by reliability/reliability_checker.py.",
        "% Values are micro F1 over exact-first one-to-one mention matches.",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['domain_display']} & {row['label_display']} & {row['records']} & "
            f"{float(row['exact_micro_f1']):.3f} & "
            f"{float(row['containment_micro_f1']):.3f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_simple_oursrev(text: str) -> str:
    pattern = re.compile(r"\\OURSREV\{([^{}]*)\}\{([^{}]*)\}")
    previous = ""
    while previous != text:
        previous = text
        text = pattern.sub(lambda match: match.group(2), text)
    return text


def parse_manuscript_agreement_table(path: Path) -> list[dict[str, object]]:
    pattern = re.compile(
        r"^\s*(Software developers|Journalists)\s*&\s*"
        r"(Hard skills|Soft skills|Distinct skills|Must-have skills|Nice-to-have skills)"
        r"\s*&\s*(\d+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*\\\\"
    )
    rows: list[dict[str, object]] = []
    manuscript = resolve_simple_oursrev(path.read_text(encoding="utf-8"))
    for line in manuscript.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        domain_display, label_display, records, exact, containment = match.groups()
        rows.append(
            {
                "domain_display": domain_display,
                "label_display": label_display,
                "manuscript_records": int(records),
                "manuscript_exact_micro_f1": float(exact),
                "manuscript_containment_micro_f1": float(containment),
            }
        )
    if len(rows) != 10:
        raise ValueError(
            f"Expected 10 manuscript agreement rows in {path}, found {len(rows)}"
        )
    return rows


def compare_with_manuscript(
    manuscript_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    current = {
        (str(row["domain_display"]), str(row["label_display"])): row
        for row in manuscript_rows
    }
    output: list[dict[str, object]] = []
    for row in summary_rows:
        key = (str(row["domain_display"]), str(row["label_display"]))
        old = current[key]
        recalculated_exact = float(row["exact_micro_f1"])
        recalculated_containment = float(row["containment_micro_f1"])
        manuscript_exact = float(old["manuscript_exact_micro_f1"])
        manuscript_containment = float(old["manuscript_containment_micro_f1"])
        output.append(
            {
                "domain": row["domain"],
                "label": row["label"],
                "manuscript_records": old["manuscript_records"],
                "recalculated_records": row["records"],
                "manuscript_exact_micro_f1": manuscript_exact,
                "recalculated_exact_micro_f1": recalculated_exact,
                "exact_delta": recalculated_exact - manuscript_exact,
                "manuscript_containment_micro_f1": manuscript_containment,
                "recalculated_containment_micro_f1": recalculated_containment,
                "containment_delta": recalculated_containment - manuscript_containment,
            }
        )
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    per_record_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    audits: list[WorkbookAudit] = []

    for config in workbook_configs():
        per_record, summary, audit = score_workbook(config)
        per_record_rows.extend(per_record)
        summary_rows.extend(summary)
        audits.append(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reliability_per_record.csv", per_record_rows)
    write_csv(args.output_dir / "reliability_summary.csv", summary_rows)
    write_csv(
        args.output_dir / "input_provenance.csv",
        [audit.__dict__ for audit in audits],
    )
    write_manuscript_rows(args.output_dir / "manuscript_table_rows.tex", summary_rows)

    if args.manuscript_tex is not None:
        manuscript_rows = parse_manuscript_agreement_table(args.manuscript_tex)
        comparison = compare_with_manuscript(manuscript_rows, summary_rows)
        write_csv(args.output_dir / "manuscript_table_comparison.csv", comparison)

    LOGGER.info(
        "Wrote %d per-record rows and %d summary rows to %s",
        len(per_record_rows),
        len(summary_rows),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
