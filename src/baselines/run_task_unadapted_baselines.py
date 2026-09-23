from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import re
import shutil
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "task_unadapted_baselines" / "generated"
DEFAULT_RULE_ROOT = ROOT / "data" / "prediction_rules" / "generated"
DEFAULT_TEST_ROOT = ROOT / "data" / "splits" / "datasets"

SEED = 13
BERT_MODEL_NAME = "KBLab/bert-base-swedish-cased"
BART_MODEL_NAME = "KBLab/bart-base-swedish-cased"

BERT_MAX_LEN = 512
BERT_DOC_STRIDE = 128
BART_MAX_SOURCE_LEN = 1024
BART_CHUNK_MAX_TOKENS = 1000
BART_CHUNK_OVERLAP = 160
BART_MAX_NEW_TOKENS = 512
BART_NUM_BEAMS = 6
BART_NO_REPEAT_NGRAM_SIZE = 3

DOMAINS = ("it", "journal", "merged")
SKILLS = (
    "all_skills",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
    "must_have",
    "nice_to_have",
)
LABELS = ("O", "B-SKILL", "I-SKILL")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


@dataclass(frozen=True)
class PredictionRules:
    allowlist: frozenset[str]
    fragment_blacklist: frozenset[str]
    fragment_prefixes: frozenset[str]
    stopwords: frozenset[str]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class BaselineMetadata:
    schema_version: str
    family: str
    domain: str
    skill: str
    run_name: str
    seed: int
    model_name: str
    model_revision: str | None
    finetuned: bool
    uses_training_partition: bool
    uses_validation_partition: bool
    baseline_definition: str
    test_file: str
    test_sha256: str
    test_rows: int
    prediction_rules_file: str
    prediction_rules_sha256: str
    input_builder: str
    prediction_postprocessing: str
    scoring_location: str
    decoding: dict[str, int | bool]
    classifier_head_sha256: str | None
    python_version: str
    torch_version: str
    transformers_version: str


class Predictor(Protocol):
    model_revision: str | None
    classifier_head_sha256: str | None

    def predict_raw(self, text: str) -> list[str]: ...


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def norm_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
    normalized = normalized.replace("\ufeff", "")
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", normalized)
    return " ".join(normalized.strip().lower().split())


def tokenize_for_match(value: str) -> list[str]:
    normalized = re.sub(r"[^\w]+", " ", norm_key(value), flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    return normalized.split() if normalized else []


def is_contiguous_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    if not short or len(short) > len(long):
        return False
    width = len(short)
    return any(
        list(long[index : index + width]) == list(short)
        for index in range(len(long) - width + 1)
    )


def normalize_skill_surface(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
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
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        key = norm_key(cleaned)
        if key and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def parse_skills(value: object) -> list[str]:
    if isinstance(value, list):
        return exact_dedupe(str(item) for item in value if str(item).strip())
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    separator = "|" if "|" in text else ";" if ";" in text else None
    parts = [text] if separator is None else text.split(separator)
    return exact_dedupe(part.strip() for part in parts if part.strip())


def load_rules(path: Path) -> PredictionRules:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid prediction rules: {path}")

    def string_set(field: str) -> frozenset[str]:
        values = payload.get(field)
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            raise ValueError(f"Invalid {field} in {path}")
        return frozenset(norm_key(item) for item in values)

    return PredictionRules(
        allowlist=string_set("predicted_skill_allowlist"),
        fragment_blacklist=string_set("predicted_skill_fragment_blacklist"),
        fragment_prefixes=string_set("predicted_skill_fragment_prefixes"),
        stopwords=string_set("predicted_skill_stopwords"),
        source_path=path,
        sha256=sha256_file(path),
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
    duplicates_removed = len(normalized_raw) - len(raw_unique)
    kept = [value for value in raw_unique if not is_prediction_fragment(value, rules)]
    kept_keys = {norm_key(value) for value in kept}
    filtered = [value for value in raw_unique if norm_key(value) not in kept_keys]
    return kept, filtered, duplicates_removed


def build_source_text(row: Mapping[str, object]) -> str:
    values = (row.get("headline"), row.get("description"))
    return "\n".join(
        value.strip() for value in values if isinstance(value, str) and value.strip()
    )


def gold_items(row: Mapping[str, object], skill: str) -> list[str]:
    field = "all_skills_list" if skill == "all_skills" else "skills_list"
    return parse_skills(row.get(field))


def unsupported_predictions(predictions: Sequence[str], source_text: str) -> list[str]:
    source_tokens = tokenize_for_match(source_text)
    return [
        prediction
        for prediction in predictions
        if not is_contiguous_subsequence(tokenize_for_match(prediction), source_tokens)
    ]


def expand_span_to_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def clean_span_text(value: str) -> str:
    text = re.sub(r"^[\-*\u2022\u00b7]+\s*", "", value.strip())
    return " ".join(text.strip(" \t\r\n,.;:|").split())


def spans_from_predictions(
    offsets: Sequence[tuple[int, int]], predictions: Sequence[int]
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for (start, end), label in zip(offsets, predictions, strict=True):
        if start == 0 and end == 0:
            continue
        if label == LABEL2ID["B-SKILL"]:
            if current is not None:
                spans.append(current)
            current = (start, end)
        elif label == LABEL2ID["I-SKILL"]:
            current = (
                (start, end) if current is None else (current[0], max(current[1], end))
            )
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    return spans


def merge_overlapping_spans(
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (item[0], item[1]))
    output = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = output[-1]
        if start <= previous_end:
            output[-1] = (previous_start, max(previous_end, end))
        else:
            output.append((start, end))
    return output


class BertTaskUnadaptedPredictor:
    def __init__(self, device: str) -> None:
        import torch
        import transformers
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        set_all_seeds(SEED)
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(device)
        self.model.eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        buffer = io.BytesIO()
        torch.save(self.model.classifier.state_dict(), buffer)
        self.classifier_head_sha256 = sha256_bytes(buffer.getvalue())

    def predict_raw(self, text: str) -> list[str]:
        if not text.strip():
            return []
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=BERT_MAX_LEN,
            stride=BERT_DOC_STRIDE,
            return_overflowing_tokens=True,
            padding=False,
        )
        spans: list[tuple[int, int]] = []
        with self.torch.no_grad():
            for input_ids, attention_mask, offsets in zip(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["offset_mapping"],
                strict=True,
            ):
                tensor_ids = self.torch.tensor([input_ids], device=self.device)
                tensor_attention = self.torch.tensor(
                    [attention_mask], device=self.device
                )
                logits = self.model(
                    input_ids=tensor_ids, attention_mask=tensor_attention
                ).logits
                predictions = self.torch.argmax(logits, dim=-1)[0].tolist()
                spans.extend(spans_from_predictions(offsets, predictions))

        extracted: list[str] = []
        for start, end in merge_overlapping_spans(spans):
            if 0 <= start < end <= len(text):
                start, end = expand_span_to_word_boundaries(text, start, end)
                value = clean_span_text(text[start:end])
                if value:
                    extracted.append(value)
        return exact_dedupe(extracted)


class BartTaskUnadaptedPredictor:
    def __init__(self, device: str) -> None:
        import torch
        import transformers
        from transformers import AutoTokenizer, BartForConditionalGeneration

        set_all_seeds(SEED)
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_NAME, use_fast=True)
        self.model = BartForConditionalGeneration.from_pretrained(BART_MODEL_NAME).to(
            device
        )
        self.model.eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        self.classifier_head_sha256 = None

    def chunk_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= BART_CHUNK_MAX_TOKENS:
            return [text.strip()]
        chunks: list[str] = []
        step = max(1, BART_CHUNK_MAX_TOKENS - BART_CHUNK_OVERLAP)
        for start in range(0, len(token_ids), step):
            piece = token_ids[start : start + BART_CHUNK_MAX_TOKENS]
            if not piece:
                break
            chunks.append(
                self.tokenizer.decode(piece, skip_special_tokens=True).strip()
            )
            if start + BART_CHUNK_MAX_TOKENS >= len(token_ids):
                break
        return chunks

    def generate_chunk(self, text: str) -> list[str]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=BART_MAX_SOURCE_LEN,
        )
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=BART_MAX_NEW_TOKENS,
                num_beams=BART_NUM_BEAMS,
                no_repeat_ngram_size=BART_NO_REPEAT_NGRAM_SIZE,
                use_cache=False,
            )
        return parse_skills(
            self.tokenizer.decode(generated[0], skip_special_tokens=True)
        )

    def predict_raw(self, text: str) -> list[str]:
        values: list[str] = []
        for chunk in self.chunk_text(text):
            values.extend(self.generate_chunk(chunk))
        return exact_dedupe(values)


def resolve_test_path(test_root: Path, domain: str, skill: str) -> Path:
    filename = f"{domain}_{skill}_test_seed{SEED}.jsonl"
    direct = test_root / domain / skill / filename
    if direct.is_file():
        return direct
    search_root = Path("/kaggle/input")
    if search_root.is_dir():
        candidates = [
            path
            for path in search_root.rglob(filename)
            if "prompt_safe_splits_40_10_50" in path.parts
            and "datasets" in path.parts
            and domain in path.parts
            and skill in path.parts
        ]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError(f"Ambiguous canonical test file {filename}: {candidates}")
    raise FileNotFoundError(f"Cannot locate canonical test file {filename}")


def resolve_rules_path(rule_root: Path, domain: str, skill: str) -> Path:
    direct = rule_root / f"{domain}_{skill}_rules.json"
    if direct.is_file():
        return direct
    raise FileNotFoundError(direct)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        rows.append(value)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_name(family: str, domain: str, skill: str) -> str:
    return f"{family}_{domain}_{skill}_task_unadapted"


def decoding_metadata(family: str) -> dict[str, int | bool]:
    if family == "bert":
        return {"max_length": BERT_MAX_LEN, "document_stride": BERT_DOC_STRIDE}
    return {
        "max_source_length": BART_MAX_SOURCE_LEN,
        "chunk_max_tokens": BART_CHUNK_MAX_TOKENS,
        "chunk_overlap": BART_CHUNK_OVERLAP,
        "max_new_tokens": BART_MAX_NEW_TOKENS,
        "num_beams": BART_NUM_BEAMS,
        "no_repeat_ngram_size": BART_NO_REPEAT_NGRAM_SIZE,
        "use_cache": False,
    }


def build_metadata(
    family: str,
    domain: str,
    skill: str,
    predictor: Predictor,
    test_path: Path,
    rules: PredictionRules,
    test_rows: int,
) -> BaselineMetadata:
    import torch

    return BaselineMetadata(
        schema_version="task-unadapted-baseline",
        family=family,
        domain=domain,
        skill=skill,
        run_name=run_name(family, domain, skill),
        seed=SEED,
        model_name=BERT_MODEL_NAME if family == "bert" else BART_MODEL_NAME,
        model_revision=predictor.model_revision,
        finetuned=False,
        uses_training_partition=False,
        uses_validation_partition=False,
        baseline_definition=(
            "Pretrained encoder plus a fixed-seed randomly initialized BIO head, "
            "with no task training"
            if family == "bert"
            else "Pretrained denoising checkpoint decoded without task fine-tuning"
        ),
        test_file=str(test_path),
        test_sha256=sha256_file(test_path),
        test_rows=test_rows,
        prediction_rules_file=str(rules.source_path),
        prediction_rules_sha256=rules.sha256,
        input_builder="headline.strip() + '\\n' + description.strip(), omitting empty fields",
        prediction_postprocessing=(
            "Evaluation pipeline surface normalization, exact deduplication, and frozen "
            "train/validation-derived fragment rules; no nested-skill collapse"
        ),
        scoring_location=(
            "Metrics are recomputed after download by the central evaluation pipeline scorer"
        ),
        decoding=decoding_metadata(family),
        classifier_head_sha256=predictor.classifier_head_sha256,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        transformers_version=str(getattr(predictor, "transformers_version", "unknown")),
    )


def write_run_zip(
    family: str,
    domain: str,
    skill: str,
    predictor: Predictor,
    raw_cache: dict[str, list[str]],
    test_root: Path,
    rule_root: Path,
    output_root: Path,
) -> Path:
    test_path = resolve_test_path(test_root, domain, skill)
    test_rows = read_jsonl(test_path)
    identifiers = [str(row.get("id", "")).strip() for row in test_rows]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Invalid canonical test IDs in {test_path}")
    rules = load_rules(resolve_rules_path(rule_root, domain, skill))

    rows: list[dict[str, object]] = []
    for index, row in enumerate(test_rows, start=1):
        source_text = build_source_text(row)
        cache_key = sha256_bytes(source_text.encode("utf-8"))
        if cache_key not in raw_cache:
            raw_cache[cache_key] = predictor.predict_raw(source_text)
        raw_prediction = raw_cache[cache_key]
        prediction, filtered, duplicates_removed = postprocess_predictions(
            raw_prediction, rules
        )
        unsupported = unsupported_predictions(prediction, source_text)
        gold = gold_items(row, skill)
        rows.append(
            {
                "id": identifiers[index - 1],
                "gold": " | ".join(gold),
                "gold_json": json.dumps(gold, ensure_ascii=False),
                "raw_pred": " | ".join(raw_prediction),
                "raw_pred_json": json.dumps(raw_prediction, ensure_ascii=False),
                "pred": " | ".join(prediction),
                "pred_json": json.dumps(prediction, ensure_ascii=False),
                "filtered_out_pred": " | ".join(filtered),
                "filtered_out_pred_json": json.dumps(filtered, ensure_ascii=False),
                "filtered_out_count": len(filtered),
                "duplicate_items_removed": duplicates_removed,
                "unsupported_pred": " | ".join(unsupported),
                "unsupported_pred_json": json.dumps(unsupported, ensure_ascii=False),
                "unsupported_count": len(unsupported),
                "unsupported_rate": (
                    len(unsupported) / len(prediction) if prediction else 0.0
                ),
            }
        )
        if index % 25 == 0 or index == len(test_rows):
            LOGGER.info(
                "%s/%s/%s: %d/%d rows",
                family,
                domain,
                skill,
                index,
                len(test_rows),
            )

    name = run_name(family, domain, skill)
    temporary = output_root / f".{name}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    result_path = temporary / f"{name}_test_results_seed{SEED}.csv"
    metadata_path = temporary / f"{name}_metadata_seed{SEED}.json"
    write_csv(result_path, rows)
    metadata = build_metadata(
        family,
        domain,
        skill,
        predictor,
        test_path,
        rules,
        len(test_rows),
    )
    metadata_path.write_text(
        json.dumps(asdict(metadata), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    zip_path = output_root / f"{name}_outputs.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_archive:
        zip_archive.write(result_path, result_path.name)
        zip_archive.write(metadata_path, metadata_path.name)
    shutil.rmtree(temporary)
    return zip_path


def make_predictor(family: str, device: str) -> Predictor:
    if family == "bert":
        return BertTaskUnadaptedPredictor(device)
    if family == "bart":
        return BartTaskUnadaptedPredictor(device)
    raise ValueError(f"Unsupported family: {family}")


def parse_csv_values(value: str, allowed: Sequence[str]) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid values {invalid}; allowed={list(allowed)}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("bert", "bart"))
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--skills", default=",".join(SKILLS))
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--rule-root", type=Path, default=DEFAULT_RULE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    domains = parse_csv_values(args.domains, DOMAINS)
    skills = parse_csv_values(args.skills, SKILLS)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictor = make_predictor(args.family, device)
    raw_cache: dict[str, list[str]] = {}
    written: list[str] = []
    for domain in domains:
        for skill in skills:
            output = write_run_zip(
                args.family,
                domain,
                skill,
                predictor,
                raw_cache,
                args.test_root.resolve(),
                args.rule_root.resolve(),
                output_root,
            )
            written.append(str(output))
            LOGGER.info("Wrote %s", output)
    print(
        json.dumps(
            {
                "family": args.family,
                "domains": domains,
                "skills": skills,
                "zip_outputs": written,
                "cached_unique_inputs": len(raw_cache),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
