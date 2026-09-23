from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

Domain = Literal["it", "journal", "merged"]
Skill = Literal[
    "all_skills",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
]
Split = Literal["train", "validation", "test", "excluded"]
JsonRecord = dict[str, object]

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "splits"
DEFAULT_SEED = 13

DOMAINS: tuple[Domain, ...] = ("it", "journal", "merged")
SKILLS: tuple[Skill, ...] = (
    "all_skills",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
)
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.40,
    "validation": 0.10,
    "test": 0.50,
}

DOMAIN_DIRS: dict[Domain, str] = {
    "it": "IT",
    "journal": "journal",
    "merged": "merged",
}
PROMPT_SKILLS: tuple[Skill, ...] = (
    "distinct_skills",
    "hard_skills",
    "must_have",
    "nice_to_have",
    "soft_skills",
)
COMBINED_SKILL_LIST_FIELDS: dict[Skill, str] = {
    "all_skills": "all_skills_list",
    "hard_skills": "hard_skills_list",
    "soft_skills": "soft_skills_list",
    "distinct_skills": "distinct_skills_list",
    "must_have": "must_have_list",
    "nice_to_have": "nice_to_have_list",
}


@dataclass(frozen=True)
class SplitCounts:
    train: int
    validation: int
    test: int

    @property
    def assigned(self) -> int:
        return self.train + self.validation + self.test


@dataclass(frozen=True)
class PromptSource:
    domain: Literal["it", "journal"]
    skill: Skill
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic 40/10/50 splits that minimize prompt-development "
            "ID overlap in test and make it zero whenever mathematically feasible. "
            "IDs are independent samples; text-based grouping is intentionally not applied."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=30_000,
        help="Maximum deterministic swap attempts per configuration.",
    )
    return parser.parse_args()


def read_json_records(path: Path) -> list[JsonRecord]:
    with path.open(encoding="utf-8") as file:
        try:
            payload = json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            records: list[JsonRecord] = []
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Expected an object at {path}:{line_number}, got {type(value)}"
                    )
                records.append(value)
            return records

    if isinstance(payload, list):
        return [value for value in payload if isinstance(value, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [value for value in data if isinstance(value, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def data_path(domain: Domain, skill: Skill) -> Path:
    domain_dir = DATA_ROOT / "processed" / DOMAIN_DIRS[domain]
    if skill == "all_skills":
        return domain_dir / "dataset_clean_merged_5skills.jsonl"
    return domain_dir / "by_skill_type" / f"skills_{skill}.jsonl"


def prompt_source(domain: Literal["it", "journal"], skill: Skill) -> PromptSource:
    if skill == "all_skills":
        raise ValueError("all_skills uses the union of the five prompt sources")
    path = DATA_ROOT / "prompt_development_sources" / domain / f"{skill}.csv"
    return PromptSource(domain=domain, skill=skill, path=path)


def csv_ids(path: Path) -> tuple[set[str], int]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = {
        str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()
    }
    return ids, len(rows)


def prompt_ids_for(
    domain: Domain, skill: Skill
) -> tuple[set[str], list[PromptSource], list[dict[str, object]]]:
    source_domains: tuple[Literal["it", "journal"], ...]
    source_domains = ("it", "journal") if domain == "merged" else (domain,)
    source_skills = tuple(PROMPT_SKILLS) if skill == "all_skills" else (skill,)

    identifiers: set[str] = set()
    sources: list[PromptSource] = []
    audit_rows: list[dict[str, object]] = []
    for source_domain in source_domains:
        for source_skill in source_skills:
            source = prompt_source(source_domain, source_skill)
            source_ids, row_count = csv_ids(source.path)
            identifiers.update(source_ids)
            sources.append(source)
            audit_rows.append(
                {
                    "requested_domain": domain,
                    "requested_skill": skill,
                    "source_domain": source_domain,
                    "source_skill": source_skill,
                    "source_file": str(source.path.relative_to(REPO_ROOT)),
                    "source_rows": row_count,
                    "source_unique_ids": len(source_ids),
                }
            )
    return identifiers, sources, audit_rows


def record_id(record: JsonRecord) -> str:
    identifier = str(record.get("id", "")).strip()
    if not identifier:
        raise ValueError("Every dataset record must have a non-empty id")
    return identifier


def ensure_unique_ids(records: Sequence[JsonRecord], path: Path) -> None:
    ids = [record_id(record) for record in records]
    duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"Duplicate IDs in {path}: {preview}")


def as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def word_count(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return len(re.findall(r"\w+", value, flags=re.UNICODE))


def numeric_features(
    records: Sequence[JsonRecord],
    skill: Skill,
    id_domains: dict[str, Literal["it", "journal"]],
) -> tuple[np.ndarray, list[str]]:
    target_field = (
        COMBINED_SKILL_LIST_FIELDS["all_skills"]
        if skill == "all_skills"
        else "skills_list"
    )
    invalid_target_ids = [
        record_id(record)
        for record in records
        if not isinstance(record.get(target_field), list)
    ]
    if invalid_target_ids:
        raise ValueError(
            f"Expected list field {target_field!r} for skill={skill!r}; "
            f"invalid IDs: {invalid_target_ids[:5]}"
        )
    values: list[list[float]] = []
    names = ["input_words", "target_items", "target_words"]
    if skill == "all_skills":
        names.extend(f"{name}_items" for name in PROMPT_SKILLS)
    if id_domains:
        names.append("is_journal")

    for record in records:
        target = as_string_list(record.get(target_field))
        text = f"{record.get('headline', '')} {record.get('description', '')}"
        row = [
            float(word_count(text)),
            float(len(target)),
            float(sum(word_count(item) for item in target)),
        ]
        if skill == "all_skills":
            row.extend(
                float(len(as_string_list(record.get(COMBINED_SKILL_LIST_FIELDS[name]))))
                for name in PROMPT_SKILLS
            )
        if id_domains:
            row.append(float(id_domains.get(record_id(record)) == "journal"))
        values.append(row)
    return np.asarray(values, dtype=np.float64), names


def balance_features(
    numeric: np.ndarray, names: Sequence[str], quantiles: int = 4
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    feature_names: list[str] = []
    for column_index, name in enumerate(names):
        values = numeric[:, column_index]
        maximum = float(values.max(initial=0.0))
        if maximum > 0:
            columns.append(values / maximum)
            feature_names.append(f"{name}_scaled")

        unique = np.unique(values)
        if len(unique) <= 1:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, quantiles + 1)))
        if len(edges) <= 2:
            bins = (values > edges[0]).astype(int)
            bin_count = int(bins.max(initial=0)) + 1
        else:
            bins = np.digitize(values, edges[1:-1], right=True)
            bin_count = len(edges) - 1
        for bin_index in range(bin_count):
            columns.append((bins == bin_index).astype(np.float64))
            feature_names.append(f"{name}_q{bin_index + 1}")

    if not columns:
        return np.ones((len(numeric), 1), dtype=np.float64), ["constant"]
    return np.column_stack(columns), feature_names


def target_counts(size: int) -> SplitCounts:
    train = round(size * SPLIT_RATIOS["train"])
    validation = round(size * SPLIT_RATIOS["validation"])
    test = size - train - validation
    return SplitCounts(train=train, validation=validation, test=test)


def imbalance_score(actual: np.ndarray, target: np.ndarray, total: np.ndarray) -> float:
    weights = 1.0 / np.maximum(total, 1.0)
    return float(np.sum(np.square(actual - target) * weights))


def choose_subset(
    candidates: np.ndarray,
    count: int,
    features: np.ndarray,
    target: np.ndarray,
    total: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == len(candidates):
        return candidates.copy()
    if count < 0 or count > len(candidates):
        raise ValueError(
            f"Cannot choose {count} items from {len(candidates)} candidates"
        )

    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    selected = shuffled[:count].copy()
    unselected = shuffled[count:].copy()
    selected_sum = features[selected].sum(axis=0)
    score = imbalance_score(selected_sum, target, total)
    stale = 0

    for _ in range(iterations):
        selected_position = int(rng.integers(len(selected)))
        unselected_position = int(rng.integers(len(unselected)))
        remove_index = selected[selected_position]
        add_index = unselected[unselected_position]
        candidate_sum = selected_sum - features[remove_index] + features[add_index]
        candidate_score = imbalance_score(candidate_sum, target, total)
        if candidate_score + 1e-12 < score:
            selected[selected_position] = add_index
            unselected[unselected_position] = remove_index
            selected_sum = candidate_sum
            score = candidate_score
            stale = 0
        else:
            stale += 1
            if stale >= 5_000:
                break
    return np.sort(selected)


def optimize_assignments(
    features: np.ndarray,
    prompt_mask: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, SplitCounts]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(features), dtype=np.int64)
    prompt_indices = all_indices[prompt_mask]

    excluded = np.empty(0, dtype=np.int64)

    retained_mask = np.ones(len(features), dtype=bool)
    retained_mask[excluded] = False
    retained = all_indices[retained_mask]
    retained_prompt_mask = prompt_mask & retained_mask
    clean_retained = all_indices[retained_mask & ~prompt_mask]
    counts = target_counts(len(retained))
    total = features[retained].sum(axis=0)

    required_prompt_test = max(0, counts.test - len(clean_retained))
    if required_prompt_test:
        clean_test = clean_retained.copy()
        prompt_target = total * SPLIT_RATIOS["test"] - features[clean_test].sum(axis=0)
        prompt_test = choose_subset(
            prompt_indices,
            required_prompt_test,
            features,
            prompt_target,
            total,
            rng,
            iterations,
        )
        test = np.sort(np.concatenate((clean_test, prompt_test)))
    else:
        test = choose_subset(
            clean_retained,
            counts.test,
            features,
            total * SPLIT_RATIOS["test"],
            total,
            rng,
            iterations,
        )
    test_mask = np.zeros(len(features), dtype=bool)
    test_mask[test] = True
    non_test = all_indices[retained_mask & ~test_mask]
    validation = choose_subset(
        non_test,
        counts.validation,
        features,
        total * SPLIT_RATIOS["validation"],
        total,
        rng,
        iterations,
    )

    assignments = np.full(len(features), "train", dtype=object)
    assignments[validation] = "validation"
    assignments[test] = "test"
    assignments[excluded] = "excluded"
    actual_prompt_test = int(np.sum((assignments == "test") & retained_prompt_mask))
    if actual_prompt_test != required_prompt_test:
        raise AssertionError(
            "Test split does not contain the minimum required prompt-development IDs"
        )
    return assignments, counts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Sequence[JsonRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def domain_lookup() -> dict[str, Literal["it", "journal"]]:
    lookup: dict[str, Literal["it", "journal"]] = {}
    for domain in ("it", "journal"):
        records = read_json_records(data_path(domain, "all_skills"))
        for record in records:
            lookup[record_id(record)] = domain
    return lookup


def split_one(
    domain: Domain,
    skill: Skill,
    seed: int,
    output_dir: Path,
    iterations: int,
    id_domains: dict[str, Literal["it", "journal"]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    source_path = data_path(domain, skill)
    records = read_json_records(source_path)
    ensure_unique_ids(records, source_path)
    identifiers = [record_id(record) for record in records]
    prompt_ids, prompt_sources, prompt_audit = prompt_ids_for(domain, skill)
    current_prompt_ids = prompt_ids.intersection(identifiers)
    prompt_mask = np.asarray(
        [identifier in current_prompt_ids for identifier in identifiers], dtype=bool
    )

    numeric, numeric_names = numeric_features(
        records, skill, id_domains if domain == "merged" else {}
    )
    features, _ = balance_features(numeric, numeric_names)
    assignments, counts = optimize_assignments(features, prompt_mask, seed, iterations)

    config_name = f"{domain}_{skill}"
    manifest_rows: list[dict[str, object]] = []
    for source_index, (identifier, split, is_prompt) in enumerate(
        zip(identifiers, assignments, prompt_mask), start=1
    ):
        manifest_rows.append(
            {
                "id": identifier,
                "split": str(split),
                "seed": seed,
                "domain": domain,
                "skill": skill,
                "source_row": source_index,
                "is_prompt_development": bool(is_prompt),
            }
        )

    manifest_path = (
        output_dir / "manifests" / f"{config_name}_split_manifest_seed{seed}.csv"
    )
    write_csv(manifest_path, manifest_rows)
    dataset_dir = output_dir / "datasets" / domain / skill
    for split in ("train", "validation", "test", "excluded"):
        split_records = [
            record
            for record, assignment in zip(records, assignments)
            if assignment == split
        ]
        split_path = dataset_dir / f"{config_name}_{split}_seed{seed}.jsonl"
        if split_records or split != "excluded":
            write_jsonl(
                split_path,
                split_records,
            )
        else:
            split_path.unlink(missing_ok=True)

    distribution_rows: list[dict[str, object]] = []
    for split in ("all", "train", "validation", "test", "excluded"):
        mask = assignments != "excluded" if split == "all" else assignments == split
        if not np.any(mask):
            continue
        for column_index, feature_name in enumerate(numeric_names):
            values = numeric[mask, column_index]
            distribution_rows.append(
                {
                    "domain": domain,
                    "skill": skill,
                    "split": split,
                    "feature": feature_name,
                    "n": int(mask.sum()),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    assigned_count = int(np.sum(assignments != "excluded"))
    minimum_prompt_ids_in_test = max(
        0,
        counts.test - (len(records) - len(current_prompt_ids)),
    )
    prompt_ids_in_test = int(np.sum((assignments == "test") & prompt_mask))
    split_summary = {
        "domain": domain,
        "skill": skill,
        "source_rows": len(records),
        "assigned_rows": assigned_count,
        "excluded_rows": int(np.sum(assignments == "excluded")),
        "train_rows": int(np.sum(assignments == "train")),
        "validation_rows": int(np.sum(assignments == "validation")),
        "test_rows": int(np.sum(assignments == "test")),
        "train_ratio": int(np.sum(assignments == "train")) / assigned_count,
        "validation_ratio": int(np.sum(assignments == "validation")) / assigned_count,
        "test_ratio": int(np.sum(assignments == "test")) / assigned_count,
        "prompt_source_ids": len(prompt_ids),
        "prompt_ids_in_dataset": len(current_prompt_ids),
        "prompt_ids_in_test": prompt_ids_in_test,
        "minimum_prompt_ids_required_in_test": minimum_prompt_ids_in_test,
        "prompt_ids_excluded": int(np.sum((assignments == "excluded") & prompt_mask)),
        "manifest": str(manifest_path.relative_to(output_dir)),
        "input_file": str(source_path.relative_to(REPO_ROOT)),
        "input_sha256": sha256(source_path),
        "seed": seed,
        "status": ("PASS_MINIMAL_PROMPT_OVERLAP" if prompt_ids_in_test else "PASS"),
    }
    if (
        split_summary["train_rows"] != counts.train
        or split_summary["validation_rows"] != counts.validation
        or split_summary["test_rows"] != counts.test
        or prompt_ids_in_test != minimum_prompt_ids_in_test
    ):
        raise AssertionError(f"Split validation failed for {config_name}")

    for row in prompt_audit:
        row["ids_present_in_requested_dataset"] = len(
            csv_ids(REPO_ROOT / str(row["source_file"]))[0].intersection(identifiers)
        )
    LOGGER.info(
        "%s: train=%d validation=%d test=%d excluded=%d prompt_in_test=%d",
        config_name,
        split_summary["train_rows"],
        split_summary["validation_rows"],
        split_summary["test_rows"],
        split_summary["excluded_rows"],
        prompt_ids_in_test,
    )
    return split_summary, distribution_rows, prompt_audit


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    id_domains = domain_lookup()

    summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    prompt_sources: list[dict[str, object]] = []
    for domain in DOMAINS:
        for skill in SKILLS:
            summary, distribution, source_rows = split_one(
                domain,
                skill,
                args.seed,
                output_dir,
                args.search_iterations,
                id_domains,
            )
            summaries.append(summary)
            distributions.extend(distribution)
            prompt_sources.extend(source_rows)

    write_csv(output_dir / f"split_summary_seed{args.seed}.csv", summaries)
    write_csv(
        output_dir / f"split_distribution_audit_seed{args.seed}.csv", distributions
    )
    write_csv(output_dir / f"prompt_source_audit_seed{args.seed}.csv", prompt_sources)
    LOGGER.info("Saved prompt-safe splits to %s", output_dir)


if __name__ == "__main__":
    main()
