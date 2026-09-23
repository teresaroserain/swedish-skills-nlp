from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import nltk
from nltk.stem.snowball import EnglishStemmer, SwedishStemmer


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RULE_ROOT = ROOT / "data" / "prediction_rules"
SOURCE_ROOT = RULE_ROOT / "sources"
OUTPUT_ROOT = RULE_ROOT / "generated"
SPLIT_ROOT = ROOT / "data" / "splits" / "datasets"

SOURCE_URLS = {
    "snowball_swedish_stopwords.txt": (
        "https://snowballstem.org/algorithms/swedish/stop.txt"
    ),
    "snowball_english_stopwords.txt": (
        "https://snowballstem.org/algorithms/english/stop.txt"
    ),
    "snowball_swedish_stemmer.html": (
        "https://snowballstem.org/algorithms/swedish/stemmer.html"
    ),
    "snowball_english_stemmer.html": (
        "https://snowballstem.org/algorithms/english/stemmer.html"
    ),
    "kblab_bert_vocab.txt": (
        "https://huggingface.co/KBLab/bert-base-swedish-cased/resolve/main/vocab.txt"
    ),
    "kblab_bart_tokenizer.json": (
        "https://huggingface.co/KBLab/bart-base-swedish-cased/resolve/main/tokenizer.json"
    ),
    "kblab_bert_model_api.json": (
        "https://huggingface.co/api/models/KBLab/bert-base-swedish-cased"
    ),
    "kblab_bart_model_api.json": (
        "https://huggingface.co/api/models/KBLab/bart-base-swedish-cased"
    ),
}

DOMAINS = ("it", "journal", "merged")
SKILLS = (
    "all_skills",
    "hard_skills",
    "soft_skills",
    "must_have",
    "nice_to_have",
    "distinct_skills",
)
DEVELOPMENT_SPLITS = ("train", "validation")
MIN_ATTACHED_SUFFIX_COUNT = 5
MIN_FRAGMENT_LENGTH = 3


class JsonRow(TypedDict, total=False):
    id: str
    headline: str
    description: str
    all_skills_list: list[str]
    skills_list: list[str]


class RuleSet(TypedDict):
    run_key: str
    evidence_partitions: list[str]
    predicted_skill_allowlist: list[str]
    predicted_skill_fragment_blacklist: list[str]
    predicted_skill_fragment_prefixes: list[str]
    predicted_skill_stopwords: list[str]
    fragment_prefix_policy: str


@dataclass(frozen=True)
class DevelopmentEvidence:
    train_gold: Counter[str]
    validation_gold: Counter[str]
    surface_by_key: dict[str, str]
    words: Counter[str]

    @property
    def all_gold(self) -> Counter[str]:
        return self.train_gold + self.validation_gold


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    return " ".join(text.strip().lower().split())


def tokenize_for_match(value: str) -> list[str]:
    normalized = re.sub(r"[^\w]+", " ", norm_key(value), flags=re.UNICODE)
    return normalized.split()


def read_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_path(domain: str, skill: str, split: str) -> Path:
    return SPLIT_ROOT / domain / skill / f"{domain}_{skill}_{split}_seed13.jsonl"


def gold_field(skill: str) -> str:
    return "all_skills_list" if skill == "all_skills" else "skills_list"


def collect_development_evidence(domain: str, skill: str) -> DevelopmentEvidence:
    gold_counts: dict[str, Counter[str]] = {
        split: Counter() for split in DEVELOPMENT_SPLITS
    }
    surface_by_key: dict[str, str] = {}
    words: Counter[str] = Counter()
    field = gold_field(skill)

    for split in DEVELOPMENT_SPLITS:
        for row in read_jsonl(split_path(domain, skill, split)):
            text = f"{row.get('headline', '')}\n{row.get('description', '')}"
            words.update(re.findall(r"[^\W\d_]+", norm_key(text), flags=re.UNICODE))
            values = row.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    continue
                key = norm_key(value)
                if not key:
                    continue
                gold_counts[split][key] += 1
                surface_by_key.setdefault(key, value.strip())

    return DevelopmentEvidence(
        train_gold=gold_counts["train"],
        validation_gold=gold_counts["validation"],
        surface_by_key=surface_by_key,
        words=words,
    )


def parse_snowball_stopwords(path: Path) -> set[str]:
    values: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            continue
        token = line.split("|", maxsplit=1)[0].strip().split(maxsplit=1)[0]
        if token:
            values.add(norm_key(token))
    return values


def snowball_suffixes() -> set[str]:
    suffixes: set[str] = set()
    for stemmer_class in (SwedishStemmer, EnglishStemmer):
        for name in dir(stemmer_class):
            value = getattr(stemmer_class, name)
            if "suffixes" in name and isinstance(value, tuple):
                suffixes.update(norm_key(item) for item in value)
    return suffixes


def load_tokenizer_evidence() -> tuple[set[str], set[str]]:
    bert_vocab = set(
        (SOURCE_ROOT / "kblab_bert_vocab.txt").read_text(encoding="utf-8").splitlines()
    )
    bart_data = json.loads(
        (SOURCE_ROOT / "kblab_bart_tokenizer.json").read_text(encoding="utf-8")
    )
    bart_vocab = set(bart_data["model"]["vocab"])
    return bert_vocab, bart_vocab


def is_compact_annotated_expression(surface: str, stopwords: set[str]) -> bool:
    key = norm_key(surface)
    tokens = tokenize_for_match(surface)
    if len(tokens) != 1:
        return False
    compact = re.sub(r"\s+", "", surface)
    has_non_letter = any(not character.isalpha() for character in compact)
    return (
        key in stopwords
        or len(tokens[0]) <= 2
        or (len(compact) <= 12 and has_non_letter)
    )


def attached_suffix_count(words: Counter[str], suffix: str) -> int:
    return sum(
        count
        for word, count in words.items()
        if len(word) >= len(suffix) + 2 and word.endswith(suffix)
    )


def build_rules(
    domain: str,
    skill: str,
    external_stopwords: set[str],
    suffixes: set[str],
    bert_vocab: set[str],
    bart_vocab: set[str],
) -> tuple[RuleSet, list[dict[str, str | int | bool]]]:
    evidence = collect_development_evidence(domain, skill)
    all_gold = evidence.all_gold
    allowlist = {
        key
        for key in all_gold
        if is_compact_annotated_expression(
            evidence.surface_by_key[key], external_stopwords
        )
    }

    fragment_candidates: dict[str, int] = {}
    for suffix in suffixes:
        if len(suffix) < MIN_FRAGMENT_LENGTH or not suffix.isalpha():
            continue
        bert_continuation_only = (
            f"##{suffix}" in bert_vocab and suffix not in bert_vocab
        )
        bart_continuation_only = suffix in bart_vocab and f"Ġ{suffix}" not in bart_vocab
        if not bert_continuation_only or not bart_continuation_only:
            continue
        attached_count = attached_suffix_count(evidence.words, suffix)
        if (
            attached_count >= MIN_ATTACHED_SUFFIX_COUNT
            and evidence.words[suffix] == 0
            and all_gold[suffix] == 0
        ):
            fragment_candidates[suffix] = attached_count

    fragment_blacklist = set(fragment_candidates) - allowlist
    stopwords = external_stopwords - allowlist
    run_key = f"{domain}_{skill}"
    rule_set: RuleSet = {
        "run_key": run_key,
        "evidence_partitions": ["train", "validation"],
        "predicted_skill_allowlist": sorted(allowlist),
        "predicted_skill_fragment_blacklist": sorted(fragment_blacklist),
        "predicted_skill_fragment_prefixes": [],
        "predicted_skill_stopwords": sorted(stopwords),
        "fragment_prefix_policy": (
            "Disabled: external suffix resources do not establish that a complete "
            "multi-token prediction beginning with a suffix-like token is invalid."
        ),
    }

    rows: list[dict[str, str | int | bool]] = []
    for item in sorted(allowlist):
        rows.append(
            {
                "run_key": run_key,
                "rule_type": "PREDICTED_SKILL_ALLOWLIST",
                "item": item,
                "decision": "keep",
                "evidence": "exact annotated gold in development data",
                "train_gold_count": evidence.train_gold[item],
                "validation_gold_count": evidence.validation_gold[item],
                "development_standalone_count": evidence.words[item],
                "development_attached_suffix_count": 0,
                "external_resource": "task gold annotations",
                "bert_continuation_only": False,
                "bart_continuation_only": False,
            }
        )
    for item in sorted(fragment_blacklist):
        rows.append(
            {
                "run_key": run_key,
                "rule_type": "PREDICTED_SKILL_FRAGMENT_BLACKLIST",
                "item": item,
                "decision": "remove standalone prediction",
                "evidence": (
                    "Snowball suffix; continuation-only in both frozen tokenizers; "
                    "attached >=5 times and never standalone or gold in development data"
                ),
                "train_gold_count": 0,
                "validation_gold_count": 0,
                "development_standalone_count": 0,
                "development_attached_suffix_count": fragment_candidates[item],
                "external_resource": "Snowball stemmers and KBLab tokenizer artifacts",
                "bert_continuation_only": True,
                "bart_continuation_only": True,
            }
        )
    for item in sorted(stopwords):
        rows.append(
            {
                "run_key": run_key,
                "rule_type": "PREDICTED_SKILL_STOPWORDS",
                "item": item,
                "decision": "remove standalone prediction",
                "evidence": "official Snowball Swedish or English stop-word list",
                "train_gold_count": evidence.train_gold[item],
                "validation_gold_count": evidence.validation_gold[item],
                "development_standalone_count": evidence.words[item],
                "development_attached_suffix_count": 0,
                "external_resource": "Snowball stop-word lists",
                "bert_continuation_only": False,
                "bart_continuation_only": False,
            }
        )
    return rule_set, rows


def write_csv(path: Path, rows: list[dict[str, str | int | bool]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_manifest() -> dict[str, object]:
    bert_api = json.loads(
        (SOURCE_ROOT / "kblab_bert_model_api.json").read_text(encoding="utf-8")
    )
    bart_api = json.loads(
        (SOURCE_ROOT / "kblab_bart_model_api.json").read_text(encoding="utf-8")
    )
    files = []
    for filename, url in SOURCE_URLS.items():
        path = SOURCE_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(
            {
                "filename": filename,
                "url": url,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "source_files": files,
        "resolved_model_revisions": {
            "KBLab/bert-base-swedish-cased": bert_api["sha"],
            "KBLab/bart-base-swedish-cased": bart_api["sha"],
        },
        "nltk_version_used_to_read_snowball_suffix_tables": nltk.__version__,
        "derivation_policy": {
            "development_partitions": ["train", "validation"],
            "test_data_used": False,
            "minimum_attached_suffix_count": MIN_ATTACHED_SUFFIX_COUNT,
            "minimum_fragment_length": MIN_FRAGMENT_LENGTH,
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    swedish_stopwords = parse_snowball_stopwords(
        SOURCE_ROOT / "snowball_swedish_stopwords.txt"
    )
    english_stopwords = parse_snowball_stopwords(
        SOURCE_ROOT / "snowball_english_stopwords.txt"
    )
    external_stopwords = swedish_stopwords | english_stopwords
    suffixes = snowball_suffixes()
    bert_vocab, bart_vocab = load_tokenizer_evidence()

    all_evidence_rows: list[dict[str, str | int | bool]] = []
    summary_rows: list[dict[str, str | int | bool]] = []
    for domain in DOMAINS:
        for skill in SKILLS:
            rules, evidence_rows = build_rules(
                domain,
                skill,
                external_stopwords,
                suffixes,
                bert_vocab,
                bart_vocab,
            )
            output_path = OUTPUT_ROOT / f"{domain}_{skill}_rules.json"
            output_path.write_text(
                json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            all_evidence_rows.extend(evidence_rows)
            summary_rows.append(
                {
                    "run_key": rules["run_key"],
                    "allowlist_count": len(rules["predicted_skill_allowlist"]),
                    "fragment_blacklist_count": len(
                        rules["predicted_skill_fragment_blacklist"]
                    ),
                    "fragment_prefix_count": len(
                        rules["predicted_skill_fragment_prefixes"]
                    ),
                    "stopword_count": len(rules["predicted_skill_stopwords"]),
                    "test_data_used": False,
                }
            )

    write_csv(RULE_ROOT / "prediction_rule_evidence.csv", all_evidence_rows)
    write_csv(RULE_ROOT / "prediction_rule_summary.csv", summary_rows)
    (RULE_ROOT / "source_manifest.json").write_text(
        json.dumps(source_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Generated %d rule sets and %d evidence rows in %s",
        len(summary_rows),
        len(all_evidence_rows),
        RULE_ROOT,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
