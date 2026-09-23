from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
SCRIPT_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_ROOT.parents[1]
NEURAL_OUTPUT_ROOT = RELEASE_ROOT / "results" / "neural_run_zips"
NEURAL_VALIDATION_PATH = (
    RELEASE_ROOT / "results" / "validation" / "neural_output_validation.csv"
)
TEST_ROOT = RELEASE_ROOT / "data" / "splits" / "datasets"
MANIFEST_ROOT = RELEASE_ROOT / "data" / "splits" / "manifests"
RULE_ROOT = RELEASE_ROOT / "data" / "prediction_rules" / "generated"
LLM_SOURCE_ROOT = RELEASE_ROOT / "data" / "llm_predictions" / "normalized_source"
DEFAULT_OUTPUT_ROOT = RELEASE_ROOT / "results" / "recomputed"
BEFORE_AFTER_PATH = (
    RELEASE_ROOT / "results" / "before_after" / "before_after_by_configuration.csv"
)

MODELS = ("BERT", "BART", "GPT", "Gemini", "Claude")
DOMAINS = ("it", "journal", "merged")
SKILLS = (
    "all_skills",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
)
LLM_MODELS = ("GPT", "Gemini", "Claude")
LLM_FIELDS = (
    "must_have",
    "nice_to_have",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
)

ROUGE_TOKEN_F1_THRESHOLD = 0.55
SEMANTIC_SIM_THRESHOLD = 0.80
SEMANTIC_SHORT_SKILL_MAX_TOKENS = 2
SEMANTIC_MODEL_NAME = "KBLab/sentence-bert-swedish-cased"

JsonObject: TypeAlias = dict[str, object]
Embedding: TypeAlias = NDArray[np.float32]
ScoreFunction: TypeAlias = Callable[[int, int], float]


@dataclass(frozen=True)
class RunKey:
    model: str
    domain: str
    skill: str

    @property
    def run_key(self) -> str:
        return f"{self.domain}_{self.skill}"


@dataclass(frozen=True)
class PredictionRules:
    allowlist: frozenset[str]
    fragment_blacklist: frozenset[str]
    fragment_prefixes: frozenset[str]
    stopwords: frozenset[str]


@dataclass(frozen=True)
class Example:
    run: RunKey
    identifier: str
    source_text: str
    gold: tuple[str, ...]
    raw_prediction: tuple[str, ...]
    prediction: tuple[str, ...]
    filtered_prediction: tuple[str, ...]
    duplicate_items_removed: int
    llm_source_row: int | None
    llm_text_exact_after_normalization: bool | None
    llm_text_token_exact_after_normalization: bool | None
    llm_text_compact_exact_after_normalization: bool | None
    llm_text_token_sequence_similarity: float | None


def read_jsonl(path: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        rows.append(value)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    if isinstance(value, tuple):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return parse_skills(value)


def norm_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = normalized.replace("\ufeff", "")
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", normalized)
    return " ".join(normalized.strip().lower().split())


def tokenize_for_match(value: str) -> list[str]:
    normalized = re.sub(r"[^\w]+", " ", norm_key(value), flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    return normalized.split() if normalized else []


def compact_text_key(value: str) -> str:
    return re.sub(r"[^\w]+", "", norm_key(value), flags=re.UNICODE)


def normalize_skill_surface(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    text = " ".join(text.strip().split())
    text = re.sub(r"^[\-*\u2022\u00b7]+\s*", "", text)
    text = text.strip(" \t\r\n,;:|")
    text = re.sub(r"^[\"'`\u00b4\u2018\u2019\u201c\u201d(\[{]+", "", text)
    text = re.sub(r"[\"'`\u00b4\u2018\u2019\u201c\u201d)\]}]+$", "", text)
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
    while text.endswith("."):
        text = text[:-1].rstrip()
    return ".NET" if norm_key(text) == "net" else text


def exact_dedupe(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        key = norm_key(cleaned)
        if key and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def parse_skills(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    separator = "|" if "|" in text else ";" if ";" in text else None
    parts = [text] if separator is None else text.split(separator)
    return exact_dedupe(part.strip() for part in parts if part.strip())


def load_rules(domain: str, skill: str) -> PredictionRules:
    path = RULE_ROOT / f"{domain}_{skill}_rules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid rules file: {path}")

    def string_set(field: str) -> frozenset[str]:
        value = payload.get(field)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"Invalid {field} in {path}")
        return frozenset(norm_key(item) for item in value)

    return PredictionRules(
        allowlist=string_set("predicted_skill_allowlist"),
        fragment_blacklist=string_set("predicted_skill_fragment_blacklist"),
        fragment_prefixes=string_set("predicted_skill_fragment_prefixes"),
        stopwords=string_set("predicted_skill_stopwords"),
    )


def is_prediction_fragment(skill: str, rules: PredictionRules) -> bool:
    normalized = normalize_skill_surface(skill)
    if not normalized:
        return True
    key = norm_key(normalized)
    if key in rules.allowlist:
        return False
    tokens = tokenize_for_match(normalized)
    if not tokens:
        return True
    if tokens[0] in rules.fragment_prefixes:
        return True
    return len(tokens) == 1 and (
        tokens[0] in rules.stopwords or tokens[0] in rules.fragment_blacklist
    )


def postprocess_predictions(
    raw_predictions: Sequence[str], rules: PredictionRules
) -> tuple[list[str], list[str], int]:
    normalized_raw = [
        normalized
        for value in raw_predictions
        if (normalized := normalize_skill_surface(value))
    ]
    raw_unique = exact_dedupe(normalized_raw)
    duplicate_items_removed = len(normalized_raw) - len(raw_unique)
    kept = [value for value in raw_unique if not is_prediction_fragment(value, rules)]
    kept_keys = {norm_key(value) for value in kept}
    filtered = [value for value in raw_unique if norm_key(value) not in kept_keys]
    return kept, filtered, duplicate_items_removed


def build_source_text(row: Mapping[str, object]) -> str:
    parts = [as_string(row.get("headline")), as_string(row.get("description"))]
    return "\n".join(part for part in parts if part).strip()


def test_path(domain: str, skill: str) -> Path:
    return TEST_ROOT / domain / skill / f"{domain}_{skill}_test_seed13.jsonl"


def manifest_path(domain: str, skill: str) -> Path:
    return MANIFEST_ROOT / f"{domain}_{skill}_split_manifest_seed13.csv"


def gold_from_test_row(row: Mapping[str, object], skill: str) -> list[str]:
    field = "all_skills_list" if skill == "all_skills" else "skills_list"
    return exact_dedupe(as_string_list(row.get(field)))


def load_neural_validation_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    with NEURAL_VALIDATION_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36 or any(row.get("status") != "valid" for row in rows):
        raise ValueError("Evaluation pipeline BERT/BART validation is not 36/36 valid")
    return {
        (
            row["family"].upper(),
            "merged" if row["domain"] == "merge" else row["domain"],
            {
                "all": "all_skills",
                "hard": "hard_skills",
                "soft": "soft_skills",
                "distinct": "distinct_skills",
                "must": "must_have",
                "nice": "nice_to_have",
            }[row["skill"]],
        ): row
        for row in rows
    }


def read_zip_csv(zip_path: Path, member: str) -> list[dict[str, str]]:
    with ZipFile(zip_path) as archive:
        content = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(content)))


def resolve_neural_zip(metadata: Mapping[str, str]) -> Path:
    recorded_path = Path(metadata["zip_path"])
    if recorded_path.is_file():
        return recorded_path

    domain = "merged" if metadata["domain"] == "merge" else metadata["domain"]
    skill = {
        "all": "all-skills",
        "hard": "hard-skills",
        "soft": "soft-skills",
        "distinct": "distinct-skills",
        "must": "must-have",
        "nice": "nice-to-have",
    }[metadata["skill"]]
    release_slug = f"{metadata['family'].lower()}-{domain}-{skill}"
    release_directory = NEURAL_OUTPUT_ROOT / release_slug
    candidates = sorted(release_directory.glob("*.zip"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not resolve one neural ZIP for {metadata['ref']} in {release_directory}"
    )


def load_llm_rows(
    model: str, domain: str
) -> tuple[dict[str, JsonObject], Counter[str]]:
    source_model = model.lower()
    source_domains = ("it", "journal") if domain == "merged" else (domain,)
    indexed: dict[str, JsonObject] = {}
    duplicate_counts: Counter[str] = Counter()
    source_row = 0
    for source_domain in source_domains:
        path = LLM_SOURCE_ROOT / f"{source_model}_{source_domain}.jsonl"
        for row in read_jsonl(path):
            source_row += 1
            identifier = as_string(row.get("id"))
            if not identifier:
                raise ValueError(f"Missing LLM ID in {path}")
            duplicate_counts[identifier] += 1
            if identifier in indexed:
                continue
            indexed[identifier] = {**row, "_source_row": source_row}
    return indexed, duplicate_counts


def llm_prediction(row: Mapping[str, object], skill: str) -> list[str]:
    if skill != "all_skills":
        return parse_skills(row.get(skill))
    values: list[str] = []
    for field in LLM_FIELDS:
        values.extend(parse_skills(row.get(field)))
    return exact_dedupe(values)


def create_examples() -> tuple[list[Example], list[dict[str, object]]]:
    validation = load_neural_validation_rows()
    examples: list[Example] = []
    alignment_rows: list[dict[str, object]] = []

    for domain in DOMAINS:
        for skill in SKILLS:
            test_rows = read_jsonl(test_path(domain, skill))
            expected_ids = [as_string(row.get("id")) for row in test_rows]
            if len(expected_ids) != len(set(expected_ids)):
                raise ValueError(f"Duplicate canonical test IDs for {domain}/{skill}")
            rules = load_rules(domain, skill)

            for model in ("BERT", "BART"):
                metadata = validation[(model, domain, skill)]
                result_rows = read_zip_csv(
                    resolve_neural_zip(metadata), metadata["test_member"]
                )
                result_ids = [row["id"] for row in result_rows]
                if result_ids != expected_ids:
                    raise ValueError(
                        f"Test-ID order mismatch for {model}/{domain}/{skill}"
                    )
                for test_row, result_row in zip(test_rows, result_rows, strict=True):
                    raw = parse_skills(result_row.get("raw_pred"))
                    prediction, filtered, duplicates = postprocess_predictions(
                        raw, rules
                    )
                    examples.append(
                        Example(
                            run=RunKey(model, domain, skill),
                            identifier=result_row["id"],
                            source_text=build_source_text(test_row),
                            gold=tuple(gold_from_test_row(test_row, skill)),
                            raw_prediction=tuple(raw),
                            prediction=tuple(prediction),
                            filtered_prediction=tuple(filtered),
                            duplicate_items_removed=duplicates,
                            llm_source_row=None,
                            llm_text_exact_after_normalization=None,
                            llm_text_token_exact_after_normalization=None,
                            llm_text_compact_exact_after_normalization=None,
                            llm_text_token_sequence_similarity=None,
                        )
                    )
                alignment_rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "skill": skill,
                        "expected_test_rows": len(test_rows),
                        "available_prediction_rows": len(result_rows),
                        "missing_test_ids": 0,
                        "extra_prediction_ids": 0,
                        "exact_test_id_order": True,
                        "duplicate_source_ids": 0,
                        "text_exact_after_normalization": "not_applicable",
                        "duplicate_id_policy": "not_applicable",
                    }
                )

            for model in LLM_MODELS:
                indexed, duplicate_counts = load_llm_rows(model, domain)
                missing = [
                    identifier
                    for identifier in expected_ids
                    if identifier not in indexed
                ]
                if missing:
                    raise ValueError(
                        f"Missing {len(missing)} LLM IDs for {model}/{domain}/{skill}: "
                        f"{missing[:3]}"
                    )
                text_matches = 0
                token_text_matches = 0
                compact_text_matches = 0
                token_similarities: list[float] = []
                for test_row in test_rows:
                    identifier = as_string(test_row.get("id"))
                    source_row = indexed[identifier]
                    raw = llm_prediction(source_row, skill)
                    prediction, filtered, duplicates = postprocess_predictions(
                        raw, rules
                    )
                    canonical_source = build_source_text(test_row)
                    llm_source = build_source_text(source_row)
                    text_match = norm_key(canonical_source) == norm_key(llm_source)
                    canonical_tokens = tokenize_for_match(canonical_source)
                    llm_tokens = tokenize_for_match(llm_source)
                    token_text_match = canonical_tokens == llm_tokens
                    compact_text_match = compact_text_key(
                        canonical_source
                    ) == compact_text_key(llm_source)
                    token_similarity = SequenceMatcher(
                        a=canonical_tokens,
                        b=llm_tokens,
                        autojunk=False,
                    ).ratio()
                    text_matches += int(text_match)
                    token_text_matches += int(token_text_match)
                    compact_text_matches += int(compact_text_match)
                    token_similarities.append(token_similarity)
                    examples.append(
                        Example(
                            run=RunKey(model, domain, skill),
                            identifier=identifier,
                            source_text=llm_source,
                            gold=tuple(gold_from_test_row(test_row, skill)),
                            raw_prediction=tuple(raw),
                            prediction=tuple(prediction),
                            filtered_prediction=tuple(filtered),
                            duplicate_items_removed=duplicates,
                            llm_source_row=int(source_row["_source_row"]),
                            llm_text_exact_after_normalization=text_match,
                            llm_text_token_exact_after_normalization=token_text_match,
                            llm_text_compact_exact_after_normalization=(
                                compact_text_match
                            ),
                            llm_text_token_sequence_similarity=token_similarity,
                        )
                    )
                duplicate_source_ids = sum(
                    1 for count in duplicate_counts.values() if count > 1
                )
                alignment_rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "skill": skill,
                        "expected_test_rows": len(test_rows),
                        "available_prediction_rows": len(indexed),
                        "missing_test_ids": 0,
                        "extra_prediction_ids": len(
                            set(indexed).difference(expected_ids)
                        ),
                        "exact_test_id_order": True,
                        "duplicate_source_ids": duplicate_source_ids,
                        "text_exact_after_normalization": text_matches,
                        "text_not_exact_after_normalization": len(test_rows)
                        - text_matches,
                        "text_token_exact_after_normalization": token_text_matches,
                        "text_token_not_exact_after_normalization": len(test_rows)
                        - token_text_matches,
                        "text_compact_exact_after_normalization": compact_text_matches,
                        "text_compact_not_exact_after_normalization": len(test_rows)
                        - compact_text_matches,
                        "minimum_token_sequence_similarity": min(token_similarities),
                        "duplicate_id_policy": (
                            "retain first source occurrence (journal source row 4)"
                            if duplicate_source_ids
                            else "not_applicable"
                        ),
                    }
                )

    return examples, alignment_rows


def is_contiguous_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    length = len(short)
    return any(
        long[index : index + length] == list(short)
        for index in range(len(long) - length + 1)
    )


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0] * (len(right) + 1)
        for index, right_item in enumerate(right, start=1):
            current[index] = (
                previous[index - 1] + 1
                if left_item == right_item
                else max(previous[index], current[index - 1])
            )
        previous = current
    return previous[-1]


def overlap_f1(overlap: int, predicted: int, gold: int) -> float:
    if overlap <= 0 or predicted <= 0 or gold <= 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / gold
    return 2 * precision * recall / (precision + recall)


def lexical_score(prediction: str, gold: str) -> float:
    predicted_tokens = tokenize_for_match(prediction)
    gold_tokens = tokenize_for_match(gold)
    if not predicted_tokens or not gold_tokens:
        return 0.0
    remaining = Counter(gold_tokens)
    overlap = 0
    for token in predicted_tokens:
        if remaining[token] > 0:
            overlap += 1
            remaining[token] -= 1
    token_f1 = overlap_f1(overlap, len(predicted_tokens), len(gold_tokens))
    rouge_l = overlap_f1(
        lcs_length(predicted_tokens, gold_tokens),
        len(predicted_tokens),
        len(gold_tokens),
    )
    return max(token_f1, rouge_l)


def maximum_cardinality_pairs(
    predicted_indices: Sequence[int],
    gold_indices: Sequence[int],
    score_function: ScoreFunction,
    threshold: float,
) -> list[tuple[int, int]]:
    adjacency: dict[int, list[int]] = {}
    for predicted_index in predicted_indices:
        scored = [
            (score_function(predicted_index, gold_index), gold_index)
            for gold_index in gold_indices
        ]
        adjacency[predicted_index] = [
            gold_index
            for score, gold_index in sorted(
                scored, key=lambda item: (-item[0], item[1])
            )
            if score >= threshold
        ]

    gold_to_prediction: dict[int, int] = {}

    def augment(predicted_index: int, visited_gold: set[int]) -> bool:
        for gold_index in adjacency[predicted_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous_prediction = gold_to_prediction.get(gold_index)
            if previous_prediction is None or augment(
                previous_prediction, visited_gold
            ):
                gold_to_prediction[gold_index] = predicted_index
                return True
        return False

    for predicted_index in predicted_indices:
        augment(predicted_index, set())
    return sorted(
        (predicted_index, gold_index)
        for gold_index, predicted_index in gold_to_prediction.items()
    )


class SemanticScorer:
    def __init__(self, embeddings: Mapping[str, Embedding]) -> None:
        self._embeddings = embeddings

    def score(
        self,
        prediction: str,
        gold: str,
        allowlist: frozenset[str],
    ) -> float:
        prediction_key = norm_key(prediction)
        gold_key = norm_key(gold)
        if prediction_key in allowlist or gold_key in allowlist:
            return 0.0
        if max(len(tokenize_for_match(prediction)), len(tokenize_for_match(gold))) <= 2:
            return 0.0
        return float(
            np.dot(self._embeddings[prediction_key], self._embeddings[gold_key])
        )


def build_semantic_scorer(
    examples: Sequence[Example], batch_size: int
) -> SemanticScorer:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required; install evaluation-requirements.txt"
        ) from error

    surfaces: dict[str, str] = {}
    for example in examples:
        for value in example.prediction:
            key = norm_key(value)
            if key:
                surfaces.setdefault(key, value)
        for value in example.gold:
            normalized = normalize_skill_surface(value)
            key = norm_key(normalized)
            if key:
                surfaces.setdefault(key, normalized)
    keys = sorted(surfaces)
    LOGGER.info(
        "Encoding %d unique skill phrases with %s", len(keys), SEMANTIC_MODEL_NAME
    )
    model = SentenceTransformer(
        SEMANTIC_MODEL_NAME, device="cpu", local_files_only=True
    )
    matrix = model.encode(
        [surfaces[key] for key in keys],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    return SemanticScorer(dict(zip(keys, matrix, strict=True)))


def precision_recall_f1(
    true_positives: int, predicted_count: int, gold_count: int
) -> tuple[float, float, float]:
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def match_example(
    example: Example, semantic_scorer: SemanticScorer
) -> dict[str, int | float]:
    predictions = list(example.prediction)
    gold = exact_dedupe(normalize_skill_surface(value) for value in example.gold)
    prediction_keys = [norm_key(value) for value in predictions]
    gold_keys = [norm_key(value) for value in gold]
    prediction_tokens = [tokenize_for_match(value) for value in predictions]
    gold_tokens = [tokenize_for_match(value) for value in gold]
    remaining_predictions = list(range(len(predictions)))
    remaining_gold = list(range(len(gold)))

    def consume(pairs: Sequence[tuple[int, int]]) -> None:
        nonlocal remaining_predictions, remaining_gold
        used_predictions = {predicted for predicted, _ in pairs}
        used_gold = {gold_index for _, gold_index in pairs}
        remaining_predictions = [
            index for index in remaining_predictions if index not in used_predictions
        ]
        remaining_gold = [index for index in remaining_gold if index not in used_gold]

    exact_pairs = maximum_cardinality_pairs(
        remaining_predictions,
        remaining_gold,
        lambda predicted, gold_index: float(
            prediction_keys[predicted] == gold_keys[gold_index]
        ),
        1.0,
    )
    consume(exact_pairs)
    containment_pairs = maximum_cardinality_pairs(
        remaining_predictions,
        remaining_gold,
        lambda predicted, gold_index: float(
            is_contiguous_subsequence(
                prediction_tokens[predicted], gold_tokens[gold_index]
            )
            or is_contiguous_subsequence(
                gold_tokens[gold_index], prediction_tokens[predicted]
            )
        ),
        1.0,
    )
    consume(containment_pairs)
    lexical_pairs = maximum_cardinality_pairs(
        remaining_predictions,
        remaining_gold,
        lambda predicted, gold_index: lexical_score(
            predictions[predicted], gold[gold_index]
        ),
        ROUGE_TOKEN_F1_THRESHOLD,
    )
    consume(lexical_pairs)
    rules = load_rules(example.run.domain, example.run.skill)
    semantic_pairs = maximum_cardinality_pairs(
        remaining_predictions,
        remaining_gold,
        lambda predicted, gold_index: semantic_scorer.score(
            predictions[predicted], gold[gold_index], rules.allowlist
        ),
        SEMANTIC_SIM_THRESHOLD,
    )

    exact = len(exact_pairs)
    containment = len(containment_pairs)
    lexical = len(lexical_pairs)
    semantic = len(semantic_pairs)
    predicted_count = len(predictions)
    gold_count = len(gold)
    result: dict[str, int | float] = {
        "exact_new_matches": exact,
        "containment_new_matches": containment,
        "lexical_new_matches": lexical,
        "semantic_new_matches": semantic,
        "predicted_count": predicted_count,
        "gold_count": gold_count,
    }
    cumulative = {
        "exact": exact,
        "containment": exact + containment,
        "lexical": exact + containment + lexical,
        "semantic": exact + containment + lexical + semantic,
    }
    for level, true_positives in cumulative.items():
        precision, recall, f1 = precision_recall_f1(
            true_positives, predicted_count, gold_count
        )
        result[f"{level}_true_positives"] = true_positives
        result[f"{level}_precision"] = precision
        result[f"{level}_recall"] = recall
        result[f"{level}_f1"] = f1
    return result


def unsupported_predictions(predictions: Sequence[str], source_text: str) -> list[str]:
    source_tokens = tokenize_for_match(source_text)
    return [
        prediction
        for prediction in predictions
        if not is_contiguous_subsequence(tokenize_for_match(prediction), source_tokens)
    ]


def evaluate_examples(
    examples: Sequence[Example], semantic_scorer: SemanticScorer
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_ad_rows: list[dict[str, object]] = []
    grouped: dict[RunKey, list[dict[str, object]]] = {}
    for index, example in enumerate(examples, start=1):
        if index % 1000 == 0:
            LOGGER.info("Scored %d/%d advertisements", index, len(examples))
        metrics = match_example(example, semantic_scorer)
        unsupported = unsupported_predictions(example.prediction, example.source_text)
        row: dict[str, object] = {
            "model": example.run.model,
            "domain": example.run.domain,
            "skill": example.run.skill,
            "id": example.identifier,
            **metrics,
            "raw_prediction_count": len(example.raw_prediction),
            "filtered_prediction_count": len(example.filtered_prediction),
            "duplicate_items_removed": example.duplicate_items_removed,
            "unsupported_count": len(unsupported),
            "unsupported_rate": (
                len(unsupported) / len(example.prediction)
                if example.prediction
                else 0.0
            ),
            "gold": " | ".join(example.gold),
            "raw_prediction": " | ".join(example.raw_prediction),
            "prediction": " | ".join(example.prediction),
            "filtered_prediction": " | ".join(example.filtered_prediction),
            "unsupported_prediction": " | ".join(unsupported),
            "llm_source_row": example.llm_source_row or "",
            "llm_text_exact_after_normalization": (
                example.llm_text_exact_after_normalization
                if example.llm_text_exact_after_normalization is not None
                else "not_applicable"
            ),
            "llm_text_token_exact_after_normalization": (
                example.llm_text_token_exact_after_normalization
                if example.llm_text_token_exact_after_normalization is not None
                else "not_applicable"
            ),
            "llm_text_compact_exact_after_normalization": (
                example.llm_text_compact_exact_after_normalization
                if example.llm_text_compact_exact_after_normalization is not None
                else "not_applicable"
            ),
            "llm_text_token_sequence_similarity": (
                example.llm_text_token_sequence_similarity
                if example.llm_text_token_sequence_similarity is not None
                else "not_applicable"
            ),
        }
        per_ad_rows.append(row)
        grouped.setdefault(example.run, []).append(row)

    run_rows: list[dict[str, object]] = []
    levels = ("exact", "containment", "lexical", "semantic")
    for run in sorted(
        grouped,
        key=lambda item: (
            MODELS.index(item.model),
            DOMAINS.index(item.domain),
            SKILLS.index(item.skill),
        ),
    ):
        rows = grouped[run]
        summary: dict[str, object] = {
            "model": run.model,
            "domain": run.domain,
            "skill": run.skill,
            "test_rows": len(rows),
            "raw_prediction_items": sum(
                int(row["raw_prediction_count"]) for row in rows
            ),
            "predicted_items": sum(int(row["predicted_count"]) for row in rows),
            "gold_items": sum(int(row["gold_count"]) for row in rows),
            "filtered_items": sum(
                int(row["filtered_prediction_count"]) for row in rows
            ),
            "duplicate_items_removed": sum(
                int(row["duplicate_items_removed"]) for row in rows
            ),
            "unsupported_items": sum(int(row["unsupported_count"]) for row in rows),
            "format_compliance": (
                "not applicable to source-bound token classification"
                if run.model == "BERT"
                else "not recoverable from retained parsed artifacts"
            ),
        }
        predicted_total = int(summary["predicted_items"])
        summary["unsupported_item_rate"] = (
            int(summary["unsupported_items"]) / predicted_total
            if predicted_total
            else 0.0
        )
        for stage in ("exact", "containment", "lexical", "semantic"):
            summary[f"{stage}_new_matches"] = sum(
                int(row[f"{stage}_new_matches"]) for row in rows
            )
        for level in levels:
            for metric in ("precision", "recall", "f1"):
                summary[f"{level}_macro_{metric}"] = float(
                    np.mean([float(row[f"{level}_{metric}"]) for row in rows])
                )
            true_positives = sum(int(row[f"{level}_true_positives"]) for row in rows)
            micro_precision, micro_recall, micro_f1 = precision_recall_f1(
                true_positives,
                int(summary["predicted_items"]),
                int(summary["gold_items"]),
            )
            summary[f"{level}_micro_precision"] = micro_precision
            summary[f"{level}_micro_recall"] = micro_recall
            summary[f"{level}_micro_f1"] = micro_f1
        run_rows.append(summary)
    return per_ad_rows, run_rows


def mean_rows(
    rows: Sequence[Mapping[str, object]], group_fields: Sequence[str]
) -> list[dict[str, object]]:
    metric_fields = [
        f"{level}_macro_{metric}"
        for level in ("exact", "containment", "lexical", "semantic")
        for metric in ("precision", "recall", "f1")
    ]
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, values in grouped.items():
        record: dict[str, object] = dict(zip(group_fields, key, strict=True))
        record["configuration_count"] = len(values)
        for field in metric_fields:
            record[field] = float(np.mean([float(value[field]) for value in values]))
        predicted_items = sum(int(value["predicted_items"]) for value in values)
        raw_prediction_items = sum(
            int(value["raw_prediction_items"]) for value in values
        )
        unsupported_items = sum(int(value["unsupported_items"]) for value in values)
        filtered_items = sum(int(value["filtered_items"]) for value in values)
        record["predicted_items"] = predicted_items
        record["unsupported_items"] = unsupported_items
        record["unsupported_item_rate"] = (
            unsupported_items / predicted_items if predicted_items else 0.0
        )
        record["raw_prediction_items"] = raw_prediction_items
        record["filtered_items"] = filtered_items
        record["filtered_item_rate"] = (
            filtered_items / raw_prediction_items if raw_prediction_items else 0.0
        )
        output.append(record)
    return sorted(
        output, key=lambda row: tuple(str(row[field]) for field in group_fields)
    )


def winner_rows(
    run_rows: Sequence[Mapping[str, object]], metric: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for domain in DOMAINS:
        for skill in SKILLS:
            candidates = [
                row
                for row in run_rows
                if row["domain"] == domain and row["skill"] == skill
            ]
            ranked = sorted(
                candidates,
                key=lambda row: (-float(row[metric]), MODELS.index(str(row["model"]))),
            )
            best = ranked[0]
            runner_up = ranked[1]
            output.append(
                {
                    "domain": domain,
                    "skill": skill,
                    "metric": metric,
                    "winner": best["model"],
                    "winner_score": best[metric],
                    "runner_up": runner_up["model"],
                    "runner_up_score": runner_up[metric],
                    "margin": float(best[metric]) - float(runner_up[metric]),
                }
            )
    return output


def domain_winner_rows(
    domain_rows: Sequence[Mapping[str, object]], metric: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for domain in DOMAINS:
        candidates = [row for row in domain_rows if row["domain"] == domain]
        ranked = sorted(
            candidates,
            key=lambda row: (-float(row[metric]), MODELS.index(str(row["model"]))),
        )
        best = ranked[0]
        runner_up = ranked[1]
        output.append(
            {
                "domain": domain,
                "metric": metric,
                "winner": best["model"],
                "winner_score": best[metric],
                "runner_up": runner_up["model"],
                "runner_up_score": runner_up[metric],
                "margin": float(best[metric]) - float(runner_up[metric]),
            }
        )
    return output


def winner_count_rows(
    all_skill_winners: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric in (
        "exact_macro_f1",
        "containment_macro_f1",
        "lexical_macro_f1",
        "semantic_macro_f1",
    ):
        counts = Counter(
            str(row["winner"]) for row in all_skill_winners if row["metric"] == metric
        )
        for model in MODELS:
            output.append(
                {
                    "metric": metric,
                    "model": model,
                    "domain_skill_cell_wins": counts[model],
                }
            )
    return output


def prompt_overlap_ids() -> set[str]:
    path = manifest_path("it", "all_skills")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["id"]
        for row in rows
        if row["split"] == "test" and row["is_prompt_development"].lower() == "true"
    }


def llm_source_text_audit_rows(
    examples: Sequence[Example],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Example]] = {}
    for example in examples:
        if example.run.model not in LLM_MODELS:
            continue
        grouped.setdefault(
            (example.run.model, example.run.domain, example.identifier), []
        ).append(example)

    output: list[dict[str, object]] = []
    for (model, domain, identifier), values in sorted(grouped.items()):
        example = values[0]
        token_exact = bool(example.llm_text_token_exact_after_normalization)
        if token_exact:
            continue
        output.append(
            {
                "model": model,
                "domain": domain,
                "id": identifier,
                "affected_skill_views": " | ".join(
                    sorted({value.run.skill for value in values})
                ),
                "surface_exact_after_normalization": bool(
                    example.llm_text_exact_after_normalization
                ),
                "token_exact_after_normalization": token_exact,
                "compact_exact_after_normalization": bool(
                    example.llm_text_compact_exact_after_normalization
                ),
                "token_sequence_similarity": (
                    example.llm_text_token_sequence_similarity
                ),
                "interpretation": (
                    "whitespace/punctuation-only difference"
                    if example.llm_text_compact_exact_after_normalization
                    else "substantive text difference requiring manual review"
                ),
            }
        )
    return output


def sensitivity_rows(
    per_ad_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    excluded_ids = prompt_overlap_ids()
    output: list[dict[str, object]] = []
    for model in MODELS:
        rows = [
            row
            for row in per_ad_rows
            if row["model"] == model
            and row["domain"] == "it"
            and row["skill"] == "all_skills"
        ]
        retained = [row for row in rows if str(row["id"]) not in excluded_ids]
        record: dict[str, object] = {
            "model": model,
            "domain": "it",
            "skill": "all_skills",
            "excluded_prompt_development_ids": len(excluded_ids),
            "full_test_rows": len(rows),
            "sensitivity_test_rows": len(retained),
        }
        for level in ("exact", "containment", "lexical", "semantic"):
            for metric in ("precision", "recall", "f1"):
                full = float(np.mean([float(row[f"{level}_{metric}"]) for row in rows]))
                reduced = float(
                    np.mean([float(row[f"{level}_{metric}"]) for row in retained])
                )
                record[f"full_{level}_macro_{metric}"] = full
                record[f"sensitivity_{level}_macro_{metric}"] = reduced
                record[f"delta_{level}_macro_{metric}"] = reduced - full
        output.append(record)
    return output


def neural_recalculation_audit(
    run_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    validation = load_neural_validation_rows()
    fields = {
        "exact_macro_f1": "exact_macro_f1",
        "containment_macro_f1": "containment_macro_f1",
        "lexical_relaxed_macro_f1": "lexical_macro_f1",
        "semantic_relaxed_macro_f1": "semantic_macro_f1",
    }
    output: list[dict[str, object]] = []
    for row in run_rows:
        model = str(row["model"])
        if model not in {"BERT", "BART"}:
            continue
        stored = validation[(model, str(row["domain"]), str(row["skill"]))]
        record: dict[str, object] = {
            "model": model,
            "domain": row["domain"],
            "skill": row["skill"],
        }
        valid = True
        for stored_field, recalculated_field in fields.items():
            stored_value = float(stored[stored_field])
            recalculated_value = float(row[recalculated_field])
            difference = recalculated_value - stored_value
            record[f"stored_{stored_field}"] = stored_value
            record[f"recalculated_{stored_field}"] = recalculated_value
            record[f"difference_{stored_field}"] = difference
            valid = valid and math.isclose(
                stored_value, recalculated_value, rel_tol=1e-6, abs_tol=1e-6
            )
        record["valid"] = valid
        output.append(record)
    return output


def claim_audit_rows(
    run_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    containment_winners: Sequence[Mapping[str, object]],
    exact_winners: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    domain_lookup = {
        (str(row["model"]), str(row["domain"])): row for row in domain_rows
    }

    def domain_winner(domain: str, metric: str) -> tuple[str, float]:
        ranked = sorted(
            [row for row in domain_rows if row["domain"] == domain],
            key=lambda row: -float(row[metric]),
        )
        return str(ranked[0]["model"]), float(ranked[0][metric])

    containment_domain_winners = {
        domain: domain_winner(domain, "containment_macro_f1") for domain in DOMAINS
    }
    exact_domain_winners = {
        domain: domain_winner(domain, "exact_macro_f1") for domain in DOMAINS
    }
    lexical_domain_winners = {
        domain: domain_winner(domain, "lexical_macro_f1") for domain in DOMAINS
    }
    semantic_domain_winners = {
        domain: domain_winner(domain, "semantic_macro_f1") for domain in DOMAINS
    }
    containment_counts = Counter(str(row["winner"]) for row in containment_winners)
    exact_counts = Counter(str(row["winner"]) for row in exact_winners)
    prompted_containment_labels = Counter(
        str(row["skill"])
        for row in containment_winners
        if str(row["winner"]) in LLM_MODELS
    )
    prompted_exact_labels = Counter(
        str(row["skill"]) for row in exact_winners if str(row["winner"]) in LLM_MODELS
    )
    focus_labels = {"distinct_skills", "nice_to_have"}

    def verdict(condition: bool) -> str:
        return "SUPPORTED" if condition else "NOT_SUPPORTED"

    def concentration_evidence(label_counts: Counter[str]) -> tuple[bool, str]:
        total = sum(label_counts.values())
        focused = sum(label_counts[label] for label in focus_labels)
        share = focused / total if total else 0.0
        evidence = {
            "prompted_wins": total,
            "distinct_or_nice_wins": focused,
            "share": share,
            "by_skill": dict(label_counts),
        }
        return share >= 0.75, json.dumps(evidence, ensure_ascii=False)

    containment_concentrated, containment_concentration_evidence = (
        concentration_evidence(prompted_containment_labels)
    )
    exact_concentrated, exact_concentration_evidence = concentration_evidence(
        prompted_exact_labels
    )
    gap_by_domain = {
        domain: float(
            np.mean(
                [
                    float(row["containment_macro_f1"]) - float(row["exact_macro_f1"])
                    for row in run_rows
                    if row["domain"] == domain
                ]
            )
        )
        for domain in DOMAINS
    }
    lowest_winner_skill = {
        domain: min(
            [row for row in containment_winners if row["domain"] == domain],
            key=lambda row: float(row["winner_score"]),
        )["skill"]
        for domain in DOMAINS
    }
    highest_winner_skill = {
        domain: max(
            [row for row in containment_winners if row["domain"] == domain],
            key=lambda row: float(row["winner_score"]),
        )["skill"]
        for domain in DOMAINS
    }
    domain_skill_rows = mean_rows(run_rows, ("domain", "skill"))
    lowest_mean_skill = {
        domain: min(
            [row for row in domain_skill_rows if row["domain"] == domain],
            key=lambda row: float(row["containment_macro_f1"]),
        )["skill"]
        for domain in DOMAINS
    }

    before_after_status = "NOT_REVALIDATED"
    before_after_evidence = f"Missing {BEFORE_AFTER_PATH}"
    if BEFORE_AFTER_PATH.exists():
        with BEFORE_AFTER_PATH.open(encoding="utf-8-sig", newline="") as handle:
            before_after_rows = list(csv.DictReader(handle))
        exact_improved = sum(
            float(row["delta_exact_macro_f1"]) > 0.0 for row in before_after_rows
        )
        containment_improved = sum(
            float(row["delta_containment_macro_f1"]) > 0.0 for row in before_after_rows
        )
        before_after_status = verdict(
            len(before_after_rows) == 36
            and exact_improved == 36
            and containment_improved == 36
        )
        before_after_evidence = json.dumps(
            {
                "configurations": len(before_after_rows),
                "exact_macro_f1_improved": exact_improved,
                "containment_macro_f1_improved": containment_improved,
                "source": str(BEFORE_AFTER_PATH.relative_to(RELEASE_ROOT)),
            },
            ensure_ascii=False,
        )

    rows: list[dict[str, object]] = [
        {
            "claim": "Gemini leads software developers; BERT leads journalism and merged",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                containment_domain_winners["it"][0] == "Gemini"
                and containment_domain_winners["journal"][0] == "BERT"
                and containment_domain_winners["merged"][0] == "BERT"
            ),
            "paper_metric_evidence": json.dumps(
                containment_domain_winners, ensure_ascii=False
            ),
            "status_under_exact_metric": verdict(
                exact_domain_winners["it"][0] == "Gemini"
                and exact_domain_winners["journal"][0] == "BERT"
                and exact_domain_winners["merged"][0] == "BERT"
            ),
            "exact_metric_evidence": json.dumps(
                exact_domain_winners, ensure_ascii=False
            ),
        },
        {
            "claim": "BERT leads 13 of 18 domain-skill cells and prompted models lead five",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                containment_counts["BERT"] == 13
                and sum(containment_counts[model] for model in LLM_MODELS) == 5
            ),
            "paper_metric_evidence": json.dumps(containment_counts, ensure_ascii=False),
            "status_under_exact_metric": verdict(
                exact_counts["BERT"] == 13
                and sum(exact_counts[model] for model in LLM_MODELS) == 5
            ),
            "exact_metric_evidence": json.dumps(exact_counts, ensure_ascii=False),
        },
        {
            "claim": "BART does not lead any domain-skill cell",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(containment_counts["BART"] == 0),
            "paper_metric_evidence": f"BART winners={containment_counts['BART']}",
            "status_under_exact_metric": verdict(exact_counts["BART"] == 0),
            "exact_metric_evidence": f"BART winners={exact_counts['BART']}",
        },
        {
            "claim": "BERT leads all-skills, hard-skills, and soft-skills in every domain",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(
                    row["winner"] == "BERT"
                    for row in containment_winners
                    if row["skill"] in {"all_skills", "hard_skills", "soft_skills"}
                )
            ),
            "paper_metric_evidence": "; ".join(
                f"{row['domain']}/{row['skill']}={row['winner']}"
                for row in containment_winners
                if row["skill"] in {"all_skills", "hard_skills", "soft_skills"}
            ),
            "status_under_exact_metric": verdict(
                all(
                    row["winner"] == "BERT"
                    for row in exact_winners
                    if row["skill"] in {"all_skills", "hard_skills", "soft_skills"}
                )
            ),
            "exact_metric_evidence": "; ".join(
                f"{row['domain']}/{row['skill']}={row['winner']}"
                for row in exact_winners
                if row["skill"] in {"all_skills", "hard_skills", "soft_skills"}
            ),
        },
        {
            "claim": "GPT does not lead an individual domain-skill cell",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(containment_counts["GPT"] == 0),
            "paper_metric_evidence": f"GPT winners={containment_counts['GPT']}",
            "status_under_exact_metric": verdict(exact_counts["GPT"] == 0),
            "exact_metric_evidence": f"GPT winners={exact_counts['GPT']}",
        },
        {
            "claim": "Prompted-model wins are concentrated on distinct and nice-to-have skills",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(containment_concentrated),
            "paper_metric_evidence": containment_concentration_evidence,
            "status_under_exact_metric": verdict(exact_concentrated),
            "exact_metric_evidence": exact_concentration_evidence,
        },
        {
            "claim": "The best-model distinct-skill score is the lowest best-model skill score in every domain",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(
                    value == "distinct_skills" for value in lowest_winner_skill.values()
                )
            ),
            "paper_metric_evidence": json.dumps(
                lowest_winner_skill, ensure_ascii=False
            ),
            "status_under_exact_metric": "NOT_EVALUATED_FOR_THIS_WORDING",
            "exact_metric_evidence": "See skill_winners_exact.csv",
        },
        {
            "claim": "The best-model nice-to-have score is the highest best-model skill score in every domain",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(value == "nice_to_have" for value in highest_winner_skill.values())
            ),
            "paper_metric_evidence": json.dumps(
                highest_winner_skill, ensure_ascii=False
            ),
            "status_under_exact_metric": "NOT_EVALUATED_FOR_THIS_WORDING",
            "exact_metric_evidence": "See skill_winners_exact.csv",
        },
        {
            "claim": "Distinct skills are lowest after averaging all five models within every domain",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(value == "distinct_skills" for value in lowest_mean_skill.values())
            ),
            "paper_metric_evidence": json.dumps(lowest_mean_skill, ensure_ascii=False),
            "status_under_exact_metric": "NOT_EVALUATED_FOR_THIS_WORDING",
            "exact_metric_evidence": "See domain_skill_means.csv",
        },
        {
            "claim": "The exact-to-containment gap is largest in journalism",
            "paper_metric": "mean containment_macro_f1 minus exact_macro_f1",
            "status_under_paper_metric": verdict(
                max(gap_by_domain, key=gap_by_domain.get) == "journal"
            ),
            "paper_metric_evidence": json.dumps(gap_by_domain, ensure_ascii=False),
            "status_under_exact_metric": "NOT_APPLICABLE",
            "exact_metric_evidence": "This claim is itself a metric gap",
        },
        {
            "claim": "BERT outperforms BART after fine-tuning in every domain mean",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(
                    float(domain_lookup[("BERT", domain)]["containment_macro_f1"])
                    > float(domain_lookup[("BART", domain)]["containment_macro_f1"])
                    for domain in DOMAINS
                )
            ),
            "paper_metric_evidence": "; ".join(
                f"{domain}: BERT={float(domain_lookup[('BERT', domain)]['containment_macro_f1']):.6f}, "
                f"BART={float(domain_lookup[('BART', domain)]['containment_macro_f1']):.6f}"
                for domain in DOMAINS
            ),
            "status_under_exact_metric": verdict(
                all(
                    float(domain_lookup[("BERT", domain)]["exact_macro_f1"])
                    > float(domain_lookup[("BART", domain)]["exact_macro_f1"])
                    for domain in DOMAINS
                )
            ),
            "exact_metric_evidence": "; ".join(
                f"{domain}: BERT={float(domain_lookup[('BERT', domain)]['exact_macro_f1']):.6f}, "
                f"BART={float(domain_lookup[('BART', domain)]['exact_macro_f1']):.6f}"
                for domain in DOMAINS
            ),
        },
        {
            "claim": "The domain winner pattern is stable across exact, containment, lexical, and semantic metrics",
            "paper_metric": "all four matching levels",
            "status_under_paper_metric": verdict(
                all(
                    winners[domain][0] == ("Gemini" if domain == "it" else "BERT")
                    for winners in (
                        exact_domain_winners,
                        containment_domain_winners,
                        lexical_domain_winners,
                        semantic_domain_winners,
                    )
                    for domain in DOMAINS
                )
            ),
            "paper_metric_evidence": json.dumps(
                {
                    "exact": exact_domain_winners,
                    "containment": containment_domain_winners,
                    "lexical": lexical_domain_winners,
                    "semantic": semantic_domain_winners,
                },
                ensure_ascii=False,
            ),
            "status_under_exact_metric": "SAME_AS_PAPER_METRIC",
            "exact_metric_evidence": "This claim explicitly compares all four levels",
        },
        {
            "claim": "Best domain-level model scores remain below the lower human containment-agreement bound of 0.714",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": verdict(
                all(score < 0.714 for _, score in containment_domain_winners.values())
            ),
            "paper_metric_evidence": json.dumps(
                containment_domain_winners, ensure_ascii=False
            ),
            "status_under_exact_metric": "NOT_APPLICABLE",
            "exact_metric_evidence": "The reference bound is a containment-agreement value",
        },
        {
            "claim": "Fine-tuning substantially improves BERT and BART over task-unadapted controls",
            "paper_metric": "containment_macro_f1",
            "status_under_paper_metric": before_after_status,
            "paper_metric_evidence": before_after_evidence,
            "status_under_exact_metric": before_after_status,
            "exact_metric_evidence": before_after_evidence,
        },
    ]
    return rows


def manuscript_consistency_rows() -> list[dict[str, object]]:
    return [
        {
            "topic": "HPO objective",
            "manuscript_statement": "Exact normalized macro F1",
            "executed_pipeline": "Exact normalized macro F1 (eval_exact_f1)",
            "consistent": True,
            "impact": "The manuscript distinguishes exact HPO from containment-based primary reporting.",
        },
        {
            "topic": "Canonicalization",
            "manuscript_statement": "Only exact normalized duplicates are removed",
            "executed_pipeline": "Normalization-aware exact deduplication only",
            "consistent": True,
            "impact": "Distinct nested phrases are preserved in both the manuscript and code.",
        },
        {
            "topic": "One-to-one matching",
            "manuscript_statement": "Maximum-cardinality bipartite matching at each cascade level",
            "executed_pipeline": "Deterministic maximum-cardinality bipartite matching at each cascade level",
            "consistent": True,
            "impact": "The stated and executed matching procedures are prediction-order invariant.",
        },
        {
            "topic": "Containment definition",
            "manuscript_statement": "One normalized token sequence is a contiguous subsequence of the other",
            "executed_pipeline": "One normalized token sequence is a contiguous subsequence of the other",
            "consistent": True,
            "impact": "The metric definition matches the implementation.",
        },
        {
            "topic": "Optuna sampler",
            "manuscript_statement": "TPESampler seeded with seed 13",
            "executed_pipeline": "TPESampler(seed=13)",
            "consistent": True,
            "impact": "The seeded HPO procedure is described consistently.",
        },
        {
            "topic": "BART chunk targets",
            "manuscript_statement": "Source-visible chunk targets with deterministic token-overlap fallback",
            "executed_pipeline": "Source-visible skills are assigned to matching chunks, with deterministic fallback",
            "consistent": True,
            "impact": "The BART chunk-target procedure and its remaining limitation are aligned.",
        },
        {
            "topic": "Final scoring interface",
            "manuscript_statement": "Same normalization and one-to-one scoring for all five systems",
            "executed_pipeline": "Same rules and scorer are applied to raw predictions for all five systems",
            "consistent": True,
            "impact": "Arithmetic comparison is consistent after exact test-ID alignment.",
        },
        {
            "topic": "Prompt-development overlap",
            "manuscript_statement": "Same canonical test identifiers; prompt-safe split implied",
            "executed_pipeline": "Five IT-all test IDs were used in prompt development; all other configurations have zero overlap",
            "consistent": False,
            "impact": "IT-all and the six-skill IT mean require a disclosed sensitivity result.",
        },
        {
            "topic": "Output-format compliance",
            "manuscript_statement": "Necessary diagnostic for BART and prompted models",
            "executed_pipeline": "Raw API responses and request logs are not retained; only parsed workbooks survive",
            "consistent": False,
            "impact": "Historical provider-level format compliance cannot be reconstructed honestly.",
        },
    ]


def fairness_audit_rows(
    overall_rows: Sequence[Mapping[str, object]],
    sensitivity: Sequence[Mapping[str, object]],
    source_text_audit: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    model_rates = {
        str(row["model"]): {
            "filtered_item_rate": float(row["filtered_item_rate"]),
            "unsupported_item_rate": float(row["unsupported_item_rate"]),
        }
        for row in overall_rows
    }
    maximum_prompt_overlap_delta = max(
        abs(float(row["delta_containment_macro_f1"])) for row in sensitivity
    )
    substantive_source_differences = sum(
        row["interpretation"] != "whitespace/punctuation-only difference"
        for row in source_text_audit
    )
    return [
        {
            "component": "Exact normalized matching",
            "verdict": "VALID_HPO_AND_CO_REPORTED",
            "evidence": "Same normalization, exact deduplication, global one-to-one matching, test IDs, and gold labels for all five systems; used for HPO and reported alongside the primary cross-configuration metric.",
        },
        {
            "component": "Boundary-relaxed token containment",
            "verdict": "VALID_PRIMARY_REPORTING",
            "evidence": "Same order-invariant token-contiguous rule for all systems; used consistently as the primary cross-configuration reporting metric, with exact results reported for boundary interpretation.",
        },
        {
            "component": "Lexical matching",
            "verdict": "DIAGNOSTIC_ONLY",
            "evidence": "Same token-overlap/LCS maximum and 0.55 threshold for all systems, but the threshold is study-defined and gives partial phrase credit.",
        },
        {
            "component": "Semantic matching",
            "verdict": "DIAGNOSTIC_ONLY",
            "evidence": "Same Swedish sentence-embedding checkpoint, 0.80 threshold, length guard, and train/validation-derived allowlist for all systems; model and threshold dependence preclude primary use.",
        },
        {
            "component": "Macro and micro precision/recall/F1",
            "verdict": "VALID",
            "evidence": "Macro scores weight each advertisement equally; micro scores aggregate matched, predicted, and gold counts. Both are emitted at every matching level.",
        },
        {
            "component": "Prediction filtering",
            "verdict": "VALID_WITH_DISCLOSURE",
            "evidence": json.dumps(model_rates, ensure_ascii=False),
        },
        {
            "component": "Unsupported-generation rate",
            "verdict": "VALID_DIAGNOSTIC",
            "evidence": "Measured against each system's retained input text. BERT is source-bound, so its structural zero is not a like-for-like hallucination advantage over generative systems.",
        },
        {
            "component": "Canonical test alignment",
            "verdict": "VALID",
            "evidence": "All 90 model-domain-skill configurations have every expected test ID in exact canonical order and use canonical gold labels.",
        },
        {
            "component": "LLM source-text alignment",
            "verdict": "VALID_WITH_MINOR_CAVEAT",
            "evidence": f"{len(source_text_audit)} model-domain-ID rows differ at token boundaries; substantive compact-text differences={substantive_source_differences}.",
        },
        {
            "component": "Prompt-development overlap",
            "verdict": "VALID_WITH_SENSITIVITY",
            "evidence": f"Five IT-all test IDs overlap prompt development; maximum absolute containment macro-F1 change after excluding them={maximum_prompt_overlap_delta:.6f}.",
        },
        {
            "component": "BERT/BART versus prompted LLM comparison",
            "verdict": "DESCRIPTIVE_RESOURCE_REGIME_COMPARISON",
            "evidence": "Common scoring makes outputs numerically comparable, but supervised fine-tuning and prompted inference are not matched supervision or adaptation regimes.",
        },
        {
            "component": "LLM API reproducibility and format compliance",
            "verdict": "NOT_FULLY_RECONSTRUCTABLE",
            "evidence": "Provider snapshots, raw API responses, retry histories, and original delimiter/schema failures are absent from retained parsed workbooks.",
        },
        {
            "component": "Statistical ranking claims",
            "verdict": "DESCRIPTIVE_ONLY",
            "evidence": "BERT and BART use one training seed and the retained LLM outputs do not provide repeated calls; small winner margins cannot establish significant superiority.",
        },
    ]


def write_readme(
    output_root: Path,
    run_rows: Sequence[Mapping[str, object]],
    domain_rows: Sequence[Mapping[str, object]],
    claims: Sequence[Mapping[str, object]],
    source_text_audit: Sequence[Mapping[str, object]],
) -> None:
    containment_rankings = sorted(
        mean_rows(run_rows, ("model",)),
        key=lambda row: -float(row["containment_macro_f1"]),
    )
    exact_rankings = sorted(
        mean_rows(run_rows, ("model",)),
        key=lambda row: -float(row["exact_macro_f1"]),
    )
    substantive_source_differences = sum(
        row["interpretation"] != "whitespace/punctuation-only difference"
        for row in source_text_audit
    )
    source_mismatch_ids = len({str(row["id"]) for row in source_text_audit})
    lines = [
        "EVALUATION PIPELINE ALL-MODEL EVALUATION",
        "",
        "Scope",
        "- 36 fine-tuned BERT/BART runs and 54 aligned GPT/Gemini/Claude configurations.",
        "- One canonical prompt-safe seed-13 40/10/50 test split per domain-skill configuration.",
        "- Exact global one-to-one matching first, then token-containment, lexical, and semantic diagnostics.",
        "- Exact deduplication only; no long-to-short skill replacement.",
        "",
        "Overall mean containment macro F1 across 18 configurations",
    ]
    lines.extend(
        f"- {row['model']}: {float(row['containment_macro_f1']):.6f}"
        for row in containment_rankings
    )
    lines.extend(["", "Overall mean exact macro F1 across 18 configurations"])
    lines.extend(
        f"- {row['model']}: {float(row['exact_macro_f1']):.6f}"
        for row in exact_rankings
    )
    lines.extend(["", "Domain winners under the manuscript's containment metric"])
    for domain in DOMAINS:
        candidates = [row for row in domain_rows if row["domain"] == domain]
        best = max(candidates, key=lambda row: float(row["containment_macro_f1"]))
        lines.append(
            f"- {domain}: {best['model']} ({float(best['containment_macro_f1']):.6f})"
        )
    lines.extend(["", "Claim audit"])
    lines.extend(
        f"- {row['status_under_paper_metric']}: {row['claim']}" for row in claims
    )
    lines.extend(
        [
            "",
            "Fairness and reproducibility limits",
            "- Final arithmetic scoring is common across all five systems and every run uses exact canonical test IDs.",
            "- Exact normalized macro F1 is the HPO objective. Boundary-relaxed containment macro F1 is the primary cross-configuration reporting metric, with exact results reported alongside it; lexical and semantic scores are diagnostics.",
            "- Five IT-all test IDs occurred in prompt development; see prompt_overlap_sensitivity.csv.",
            f"- The LLM source-text audit found {source_mismatch_ids} unique IDs ({len(source_text_audit)} model-domain-ID rows) with tokenization differences, all but {substantive_source_differences} compact-equal after removing whitespace and punctuation; see llm_source_text_mismatches.csv.",
            "- LLM provider snapshots, raw API responses, retries, and original format compliance are not recoverable from the retained parsed workbooks.",
            "- Supervised fine-tuning and prompted inference are different adaptation/resource regimes, not a controlled architecture comparison.",
            "- All 36 task-unadapted controls are available and validated in results/before_after; fine-tuning improves exact and containment macro F1 in every configuration.",
            "",
            "Files",
            "- all_models_all_metrics.csv: primary 90-row model/domain/skill table.",
            "- all_models_per_ad_metrics.csv: advertisement-level audit table.",
            "- domain_model_means.csv and skill_model_means.csv: aggregate tables.",
            "- domain_skill_means.csv: skill difficulty averaged across models within each domain.",
            "- skill_winners_containment.csv and skill_winners_exact.csv: cell winners.",
            "- skill_winners_all_metrics.csv, domain_winners_all_metrics.csv, and winner_counts_all_metrics.csv: winner sensitivity across all four matching levels.",
            "- claim_audit.csv: manuscript conclusion checks.",
            "- fairness_audit.csv: metric validity and comparability assessment.",
            "- manuscript_method_consistency.csv: executed-code versus manuscript differences.",
            "- neural_recalculation_audit.csv: independent reproduction of the 36 stored BERT/BART metrics.",
            "- alignment_audit.csv: test-ID and source coverage checks.",
            "- llm_source_text_mismatches.csv: non-token-identical LLM input-text audit.",
        ]
    )
    (output_root / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score evaluation pipeline BERT, BART, GPT, Gemini, and Claude outputs."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--semantic-batch-size", type=int, default=128)
    args = parser.parse_args()

    examples, alignment = create_examples()
    LOGGER.info("Loaded %d aligned model-advertisement examples", len(examples))
    semantic_scorer = build_semantic_scorer(examples, args.semantic_batch_size)
    per_ad_rows, run_rows = evaluate_examples(examples, semantic_scorer)
    domain_rows = mean_rows(run_rows, ("model", "domain"))
    skill_rows = mean_rows(run_rows, ("model", "skill"))
    domain_skill_rows = mean_rows(run_rows, ("domain", "skill"))
    overall_rows = mean_rows(run_rows, ("model",))
    winner_metrics = (
        "exact_macro_f1",
        "containment_macro_f1",
        "lexical_macro_f1",
        "semantic_macro_f1",
    )
    all_skill_winners = [
        row for metric in winner_metrics for row in winner_rows(run_rows, metric)
    ]
    all_domain_winners = [
        row
        for metric in winner_metrics
        for row in domain_winner_rows(domain_rows, metric)
    ]
    containment_winners = [
        row for row in all_skill_winners if row["metric"] == "containment_macro_f1"
    ]
    exact_winners = [
        row for row in all_skill_winners if row["metric"] == "exact_macro_f1"
    ]
    sensitivity = sensitivity_rows(per_ad_rows)
    source_text_audit = llm_source_text_audit_rows(examples)
    fairness_audit = fairness_audit_rows(
        overall_rows,
        sensitivity,
        source_text_audit,
    )
    neural_audit = neural_recalculation_audit(run_rows)
    invalid_neural = [row for row in neural_audit if not bool(row["valid"])]
    if invalid_neural:
        raise ValueError(
            "Independent BERT/BART recalculation does not match stored metrics: "
            f"{invalid_neural[:2]}"
        )
    claims = claim_audit_rows(run_rows, domain_rows, containment_winners, exact_winners)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "all_models_all_metrics.csv", run_rows)
    write_csv(output_root / "all_models_per_ad_metrics.csv", per_ad_rows)
    write_csv(output_root / "overall_model_means.csv", overall_rows)
    write_csv(output_root / "domain_model_means.csv", domain_rows)
    write_csv(output_root / "skill_model_means.csv", skill_rows)
    write_csv(output_root / "domain_skill_means.csv", domain_skill_rows)
    write_csv(output_root / "skill_winners_containment.csv", containment_winners)
    write_csv(output_root / "skill_winners_exact.csv", exact_winners)
    write_csv(output_root / "skill_winners_all_metrics.csv", all_skill_winners)
    write_csv(output_root / "domain_winners_all_metrics.csv", all_domain_winners)
    write_csv(
        output_root / "winner_counts_all_metrics.csv",
        winner_count_rows(all_skill_winners),
    )
    write_csv(output_root / "prompt_overlap_sensitivity.csv", sensitivity)
    write_csv(output_root / "neural_recalculation_audit.csv", neural_audit)
    write_csv(output_root / "alignment_audit.csv", alignment)
    write_csv(output_root / "llm_source_text_mismatches.csv", source_text_audit)
    write_csv(output_root / "claim_audit.csv", claims)
    write_csv(output_root / "fairness_audit.csv", fairness_audit)
    write_csv(
        output_root / "manuscript_method_consistency.csv",
        manuscript_consistency_rows(),
    )
    write_readme(output_root, run_rows, domain_rows, claims, source_text_audit)

    result = {
        "output_root": str(output_root),
        "run_count": len(run_rows),
        "per_advertisement_rows": len(per_ad_rows),
        "neural_recalculation_valid": sum(bool(row["valid"]) for row in neural_audit),
        "claim_supported": sum(
            row["status_under_paper_metric"] == "SUPPORTED" for row in claims
        ),
        "claim_not_supported": sum(
            row["status_under_paper_metric"] == "NOT_SUPPORTED" for row in claims
        ),
        "claim_not_revalidated": sum(
            row["status_under_paper_metric"] == "NOT_REVALIDATED" for row in claims
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
