import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import inspect
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def trainer_tokenizer_kwargs(tok: Any) -> Dict[str, Any]:
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tok}
    if "tokenizer" in params:
        return {"tokenizer": tok}
    return {}


BIO_CLASS_WEIGHT_MAX = 10.0


class WeightedTokenClassificationTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.detach().clone()

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            ignore_index=-100,
        )
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return (loss, outputs) if return_outputs else loss


def compute_bio_class_weights(
    dataset: Dataset,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, float | int]]]:
    counts = torch.zeros(len(LABEL2ID), dtype=torch.long)
    for label_sequence in dataset["labels"]:
        labels = torch.as_tensor(label_sequence, dtype=torch.long)
        labels = labels[labels != -100]
        if labels.numel():
            counts += torch.bincount(labels, minlength=len(LABEL2ID))

    if torch.any(counts == 0):
        missing = [ID2LABEL[index] for index, count in enumerate(counts) if count == 0]
        raise RuntimeError(f"Cannot compute BIO class weights; no labels for {missing}")

    outside_count = counts[LABEL2ID["O"]].float()
    weights = torch.sqrt(outside_count / counts.float())
    weights = torch.clamp(weights, min=1.0, max=BIO_CLASS_WEIGHT_MAX)
    weights[LABEL2ID["O"]] = 1.0

    details = {
        ID2LABEL[index]: {
            "count": int(counts[index].item()),
            "weight": float(weights[index].item()),
        }
        for index in range(len(counts))
    }
    return weights, details


INPUT_JSON_CANDIDATES = [
    "data/processed/IT/by_skill_type/skills_hard_skills.jsonl",
]
SPLIT_MANIFEST_CANDIDATES = [
    "data/splits/manifests/it_hard_skills_split_manifest_seed13.csv",
]
EXPECTED_INPUT_DIRECTORY = "IT"


TRAIN_SKILLS_FIELD = "skills_list"
TEST_GOLD_LIST_FIELD = "skills_list"
RUN_NAME = "bert_it_hard_skills"

MODEL_NAME = "KBLab/bert-base-swedish-cased"


MAX_LEN = 512
DOC_STRIDE = 128


EPOCHS = 3
LR = 3e-5
BATCH_SIZE = 8
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01

BASE_TRAIN_DEFAULTS = dict(
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    weight_decay=WEIGHT_DECAY,
)


DO_HPO = True
HPO_TRIALS = 20

ROUGE_TOKEN_F1_THRESHOLD = 0.55
SEMANTIC_SIM_THRESHOLD = 0.80
SEMANTIC_SHORT_SKILL_MAX_TOKENS = 2
SEMANTIC_MODEL_NAME = "KBLab/sentence-bert-swedish-cased"
SEMANTIC_MODEL_DEVICE = "cpu"

HPO_VALID_MAX = 300


HPO_BATCH_CHOICES = [4, 8]
HPO_GRAD_ACC_CHOICES = [1, 2, 4]
HPO_EPOCH_RANGE = (2, 5)
HPO_LR_RANGE = (1e-5, 8e-5)
HPO_WD_RANGE = (0.0, 0.05)


SEEDS = [13]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(" Device:", DEVICE)
embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=SEMANTIC_MODEL_DEVICE)
_EMBEDDING_CACHE: Dict[str, Any] = {}
print(" Semantic eval model:", SEMANTIC_MODEL_NAME, "on", SEMANTIC_MODEL_DEVICE)


def resolve_input_path(candidates: List[str]) -> str:
    base = Path("/kaggle/input")
    if base.exists():
        for name in candidates:
            search_name = Path(name).name if Path(name).is_absolute() else name
            hits = list(base.rglob(search_name))
            if hits:
                hits.sort(key=lambda p: len(str(p)))
                return str(hits[0])

    mnt = Path("/mnt/data")
    if mnt.exists():
        for name in candidates:
            p = mnt / name
            if p.exists():
                return str(p)

    repository_root = Path(__file__).resolve().parents[3]
    for name in candidates:
        for path in (Path(name), repository_root / name):
            if path.exists():
                return str(path)

    raise FileNotFoundError(f"Cannot find input file. candidates={candidates}")


INPUT_JSON_PATH = resolve_input_path(INPUT_JSON_CANDIDATES)
if EXPECTED_INPUT_DIRECTORY not in Path(INPUT_JSON_PATH).parts:
    raise RuntimeError(
        "Resolved input belongs to the wrong domain: "
        f"expected directory={EXPECTED_INPUT_DIRECTORY!r}, path={INPUT_JSON_PATH!r}"
    )
print(" INPUT_JSON_PATH:", INPUT_JSON_PATH)
SPLIT_MANIFEST_PATH = resolve_input_path(SPLIT_MANIFEST_CANDIDATES)
print(" SPLIT_MANIFEST_PATH:", SPLIT_MANIFEST_PATH)


def iter_records_json(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for ln_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    x = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Bad JSONL at line {ln_no}: {e}\nPreview={line[:200]!r}"
                    )
                if isinstance(x, dict):
                    yield x
        else:
            if isinstance(obj, list):
                for x in obj:
                    if isinstance(x, dict):
                        yield x
            elif isinstance(obj, dict):
                if "data" in obj and isinstance(obj["data"], list):
                    for x in obj["data"]:
                        if isinstance(x, dict):
                            yield x
                else:
                    yield obj


PREDICTED_SKILL_ALLOWLIST = {
    ".net",
    "ai",
    "bi",
    "c",
    "c#",
    "c+",
    "c++",
    "cd",
    "ci",
    "go",
    "html5",
    "hw",
    "iec61508",
    "iec62304",
    "iso13485",
    "iso14971",
    "iso26262",
    "it",
    "oauth2",
    "p4",
    "pm3",
    "qt",
    "r",
}

PREDICTED_SKILL_FRAGMENT_BLACKLIST = {
    "ance",
    "arnas",
    "aste",
    "ation",
    "eed",
    "ement",
    "ence",
    "ern",
    "erna",
    "ernas",
    "heten",
    "hetens",
    "heterna",
    "ion",
    "ive",
    "ness",
    "ous",
}

PREDICTED_SKILL_FRAGMENT_PREFIXES: set[str] = set()

PREDICTED_SKILL_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "alla",
    "allt",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "att",
    "av",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "blev",
    "bli",
    "blir",
    "blivit",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "de",
    "dem",
    "den",
    "denna",
    "deras",
    "dess",
    "dessa",
    "det",
    "detta",
    "did",
    "didn't",
    "dig",
    "din",
    "dina",
    "ditt",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "du",
    "during",
    "där",
    "då",
    "each",
    "efter",
    "ej",
    "eller",
    "en",
    "er",
    "era",
    "ert",
    "ett",
    "few",
    "for",
    "from",
    "från",
    "further",
    "för",
    "ha",
    "had",
    "hade",
    "hadn't",
    "han",
    "hans",
    "har",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "henne",
    "hennes",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "hon",
    "honom",
    "how",
    "how's",
    "hur",
    "här",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "icke",
    "if",
    "in",
    "ingen",
    "inom",
    "inte",
    "into",
    "is",
    "isn't",
    "it's",
    "its",
    "itself",
    "jag",
    "ju",
    "kan",
    "kunde",
    "let's",
    "man",
    "me",
    "med",
    "mellan",
    "men",
    "mig",
    "min",
    "mina",
    "mitt",
    "more",
    "most",
    "mot",
    "mustn't",
    "my",
    "mycket",
    "myself",
    "ni",
    "no",
    "nor",
    "not",
    "nu",
    "när",
    "någon",
    "något",
    "några",
    "och",
    "of",
    "off",
    "om",
    "on",
    "once",
    "only",
    "or",
    "oss",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "på",
    "same",
    "samma",
    "sedan",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "sig",
    "sin",
    "sina",
    "sitt",
    "själv",
    "skulle",
    "so",
    "som",
    "some",
    "such",
    "så",
    "sådan",
    "sådana",
    "sådant",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "till",
    "to",
    "too",
    "under",
    "until",
    "up",
    "upp",
    "ut",
    "utan",
    "vad",
    "var",
    "vara",
    "varför",
    "varit",
    "varje",
    "vars",
    "vart",
    "vem",
    "very",
    "vi",
    "vid",
    "vilka",
    "vilkas",
    "vilken",
    "vilket",
    "vår",
    "våra",
    "vårt",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "än",
    "är",
    "åt",
    "över",
}


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", s)
    s = " ".join(s.strip().lower().split())
    return s


def tokenize_for_match(s: str) -> List[str]:
    s = norm_key(s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    s = " ".join(s.split())
    return s.split() if s else []


def is_contiguous_subseq(short_tokens: List[str], long_tokens: List[str]) -> bool:
    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False
    length = len(short_tokens)
    return any(
        long_tokens[index : index + length] == short_tokens
        for index in range(len(long_tokens) - length + 1)
    )


def canonicalize_skills(skills: List[str]) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for value in skills:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        key = norm_key(cleaned)
        if key and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def normalize_skill_surface(s: str) -> str:
    if not isinstance(s, str):
        return ""

    t = unicodedata.normalize("NFKC", s)
    t = t.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", t)
    t = " ".join(t.strip().split())
    t = re.sub(r"^[\-\*\u2022\u00b7]+\s*", "", t)
    t = t.strip(" \t\r\n,;:|")
    t = re.sub(r"^[\"'`\u00b4\u2018\u2019\u201c\u201d\(\[\{]+", "", t)
    t = re.sub(r"[\"'`\u00b4\u2018\u2019\u201c\u201d\)\]\}]+$", "", t)
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"\s*#\s*", "#", t)
    t = re.sub(r"\bC\s*\+\s*\+", "C++", t, flags=re.IGNORECASE)
    t = re.sub(r"\bF\s*#", "F#", t, flags=re.IGNORECASE)
    t = re.sub(r"\bC\s*#", "C#", t, flags=re.IGNORECASE)
    t = re.sub(r"\bCI\s*/\s*CD\b", "CI/CD", t, flags=re.IGNORECASE)
    t = re.sub(r"\.\s+(?=[A-Za-z])", ".", t)
    t = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", t)
    t = re.sub(r"\s*-\s*", "-", t)
    t = " ".join(t.split())

    while t.endswith(".") and norm_key(t) not in PREDICTED_SKILL_ALLOWLIST:
        t = t[:-1].rstrip()

    if norm_key(t) == "net":
        return ".NET"
    return t


def is_probably_prediction_fragment(skill: str) -> bool:
    t = normalize_skill_surface(skill)
    if not t:
        return True

    key = norm_key(t)
    if key in PREDICTED_SKILL_ALLOWLIST:
        return False

    tokens = tokenize_for_match(t)
    if not tokens:
        return True

    first = tokens[0]
    if first in PREDICTED_SKILL_FRAGMENT_PREFIXES:
        return True

    if len(tokens) == 1:
        token = tokens[0]
        if token in PREDICTED_SKILL_STOPWORDS:
            return True
        if token in PREDICTED_SKILL_FRAGMENT_BLACKLIST:
            return True

    return False


def postprocess_predicted_skills(skills: List[str]) -> List[str]:
    cleaned = []
    for skill in skills:
        normalized = normalize_skill_surface(skill)
        if not normalized:
            continue
        if is_probably_prediction_fragment(normalized):
            continue
        cleaned.append(normalized)
    return canonicalize_skills(cleaned)


def removed_by_prediction_filter(raw: List[str], processed: List[str]) -> List[str]:
    processed_keys = {norm_key(item) for item in processed}
    return [
        item
        for item in canonicalize_skills(raw)
        if norm_key(normalize_skill_surface(item)) not in processed_keys
    ]


def unsupported_predictions(predictions: List[str], source_text: str) -> List[str]:
    source_tokens = tokenize_for_match(source_text)
    return [
        prediction
        for prediction in predictions
        if not is_contiguous_subseq(tokenize_for_match(prediction), source_tokens)
    ]


def expand_span_to_word_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
    if not isinstance(text, str) or not text:
        return (start, end)

    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1

    return (start, end)


def parse_skills_any(text: Any) -> List[str]:
    if isinstance(text, list):
        items = [x.strip() for x in text if isinstance(x, str) and x.strip()]
        return canonicalize_skills(items)
    if not isinstance(text, str):
        return []
    t = text.strip()
    if not t:
        return []
    if "|" in t:
        parts = [p.strip() for p in t.split("|") if p.strip()]
    elif ";" in t:
        parts = [p.strip() for p in t.split(";") if p.strip()]
    else:
        parts = [t]
    return canonicalize_skills(parts)


def parse_gold_list_field(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    return canonicalize_skills(out)


def ensure_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]
    if isinstance(v, str):
        return parse_skills_any(v)
    return []


def list_len(v: Any) -> int:
    return len(ensure_list(v))


def split_from_manifest(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH, dtype=str, encoding="utf-8-sig")
    required_columns = {"id", "split", "seed", "is_prompt_development"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Split manifest is missing columns: {sorted(missing_columns)}"
        )

    manifest["id"] = manifest["id"].astype(str).str.strip()
    if manifest["id"].duplicated().any():
        duplicates = manifest.loc[manifest["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate IDs in split manifest: {duplicates[:5]}")

    manifest_seeds = set(pd.to_numeric(manifest["seed"], errors="raise").astype(int))
    if manifest_seeds != {seed}:
        raise ValueError(
            f"Manifest seed mismatch: expected {seed}, found {sorted(manifest_seeds)}"
        )

    valid_splits = {"train", "validation", "test", "excluded"}
    invalid_splits = set(manifest["split"]).difference(valid_splits)
    if invalid_splits:
        raise ValueError(f"Invalid split names: {sorted(invalid_splits)}")

    frame = df.copy()
    frame["_manifest_id"] = frame["id"].astype(str).str.strip()
    if frame["_manifest_id"].duplicated().any():
        duplicates = frame.loc[
            frame["_manifest_id"].duplicated(), "_manifest_id"
        ].tolist()
        raise ValueError(f"Duplicate IDs in input dataset: {duplicates[:5]}")

    data_ids = set(frame["_manifest_id"])
    manifest_ids = set(manifest["id"])
    missing_ids = sorted(data_ids.difference(manifest_ids))
    extra_ids = sorted(manifest_ids.difference(data_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "Input and split manifest IDs differ. "
            f"missing_from_manifest={missing_ids[:5]}, extra_in_manifest={extra_ids[:5]}"
        )

    prompt_flags = (
        manifest["is_prompt_development"].str.lower().isin({"true", "1", "yes"})
    )
    prompt_test_count = int(((manifest["split"] == "test") & prompt_flags).sum())
    if prompt_test_count:
        print(
            " Prompt-development IDs in test:",
            prompt_test_count,
            "(minimum required to retain the fixed 40/10/50 ratio)",
        )

    indexed = frame.set_index("_manifest_id", drop=False)

    def select(split_name: str) -> pd.DataFrame:
        split_ids = manifest.loc[manifest["split"] == split_name, "id"].tolist()
        selected = indexed.loc[split_ids].copy()
        return selected.drop(columns=["_manifest_id"]).reset_index(drop=True)

    train_df = select("train")
    valid_df = select("validation")
    test_df = select("test")
    excluded_count = int((manifest["split"] == "excluded").sum())
    print(
        " Fixed prompt-safe split:",
        f"train={len(train_df)} valid={len(valid_df)} test={len(test_df)} ",
        f"excluded={excluded_count}",
    )
    return train_df, valid_df, test_df


_DASH_CLASS = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"


def _needs_boundary(ch: str) -> bool:
    return bool(re.match(r"\w", ch, flags=re.UNICODE))


def build_skill_regex(skill: str) -> re.Pattern:
    s = skill.strip()
    s = " ".join(s.split())
    parts = s.split(" ")
    escaped_parts = [re.escape(p) for p in parts]
    core = r"\s+".join(escaped_parts)
    core = core.replace(r"\-", _DASH_CLASS)
    start_guard = r"(?<!\w)" if (s and _needs_boundary(s[0])) else ""
    end_guard = r"(?!\w)" if (s and _needs_boundary(s[-1])) else ""
    return re.compile(start_guard + core + end_guard, flags=re.IGNORECASE | re.UNICODE)


def find_spans(text: str, skills: List[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not isinstance(text, str) or not text:
        return spans
    for sk in skills:
        if not isinstance(sk, str):
            continue
        sk = sk.strip()
        if len(sk) < 2:
            continue
        try:
            rgx = build_skill_regex(sk)
        except re.error:
            continue
        for m in rgx.finditer(text):
            a, b = m.start(), m.end()
            if a < b:
                spans.append((a, b))
    return spans


LABEL2ID = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[1] > b[0] and a[0] < b[1]


def align_bio_labels(
    offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]
) -> List[int]:
    labels = []
    for ts, te in offsets:
        if ts == 0 and te == 0:
            labels.append(-100)
            continue
        best = None
        best_len = -1
        for sp in spans:
            if overlap((ts, te), sp):
                L = sp[1] - sp[0]
                if L > best_len:
                    best_len = L
                    best = sp
        if best is None:
            labels.append(LABEL2ID["O"])
        else:
            ss, _ = best
            labels.append(
                LABEL2ID["B-SKILL"] if (ts <= ss < te) else LABEL2ID["I-SKILL"]
            )
    return labels


def clean_span_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    t = re.sub(r"^[\-\*\u2022\u00b7]+\s*", "", t)
    t = t.strip(" \t\r\n,.;:|")
    t = " ".join(t.split())
    return t


def spans_from_predictions(
    text: str, offsets: List[Tuple[int, int]], pred_ids: List[int]
) -> List[Tuple[int, int]]:
    spans = []
    cur = None
    for (ts, te), lab in zip(offsets, pred_ids):
        if ts == 0 and te == 0:
            continue
        if lab == LABEL2ID["B-SKILL"]:
            if cur is not None:
                spans.append(cur)
            cur = (ts, te)
        elif lab == LABEL2ID["I-SKILL"]:
            if cur is None:
                cur = (ts, te)
            else:
                cur = (cur[0], max(cur[1], te))
        else:
            if cur is not None:
                spans.append(cur)
                cur = None
    if cur is not None:
        spans.append(cur)
    return spans


def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    out = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def f1_from_counts(overlap_count: int, pred_count: int, gold_count: int) -> float:
    if overlap_count <= 0 or pred_count <= 0 or gold_count <= 0:
        return 0.0

    precision = overlap_count / pred_count
    recall = overlap_count / gold_count
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def token_overlap_f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize_for_match(pred)
    gold_tokens = tokenize_for_match(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_remaining: dict[str, int] = {}
    for token in gold_tokens:
        gold_remaining[token] = gold_remaining.get(token, 0) + 1

    overlap_count = 0
    for token in pred_tokens:
        count = gold_remaining.get(token, 0)
        if count > 0:
            overlap_count += 1
            gold_remaining[token] = count - 1

    return f1_from_counts(overlap_count, len(pred_tokens), len(gold_tokens))


def rouge_l_f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize_for_match(pred)
    gold_tokens = tokenize_for_match(gold)
    overlap_count = lcs_len(pred_tokens, gold_tokens)
    return f1_from_counts(overlap_count, len(pred_tokens), len(gold_tokens))


def lexical_relaxed_score(pred: str, gold: str) -> float:
    return max(token_overlap_f1_score(pred, gold), rouge_l_f1_score(pred, gold))


def should_use_semantic_similarity(pred: str, gold: str) -> bool:
    pred_tokens = tokenize_for_match(pred)
    gold_tokens = tokenize_for_match(gold)
    if not pred_tokens or not gold_tokens:
        return False

    pred_key = norm_key(pred)
    gold_key = norm_key(gold)
    if pred_key in PREDICTED_SKILL_ALLOWLIST or gold_key in PREDICTED_SKILL_ALLOWLIST:
        return False

    return max(len(pred_tokens), len(gold_tokens)) > SEMANTIC_SHORT_SKILL_MAX_TOKENS


def get_skill_embedding(text: str) -> Any:
    key = norm_key(text)
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached

    embedding = embedding_model.encode(
        normalize_skill_surface(text),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _EMBEDDING_CACHE[key] = embedding
    return embedding


def semantic_similarity_score(pred: str, gold: str) -> float:
    if not should_use_semantic_similarity(pred, gold):
        return 0.0

    pred_embedding = get_skill_embedding(pred)
    gold_embedding = get_skill_embedding(gold)
    return float(np.dot(pred_embedding, gold_embedding))


def unique_normalized_items(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_skill_surface(value)
        if not normalized:
            continue
        key = norm_key(normalized)
        if key and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def prf(tp: int, pred_count: int, gold_count: int) -> Tuple[float, float, float]:
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def maximum_cardinality_pairs(
    pred_indices: List[int],
    gold_indices: List[int],
    score_fn: Any,
    threshold: float,
) -> List[Tuple[int, int]]:
    adjacency: Dict[int, List[int]] = {}
    for pred_idx in pred_indices:
        scored = [
            (float(score_fn(pred_idx, gold_idx)), gold_idx) for gold_idx in gold_indices
        ]
        adjacency[pred_idx] = [
            gold_idx
            for score, gold_idx in sorted(scored, key=lambda item: (-item[0], item[1]))
            if score >= threshold
        ]

    gold_to_pred: Dict[int, int] = {}

    def augment(pred_idx: int, visited_gold: set[int]) -> bool:
        for gold_idx in adjacency[pred_idx]:
            if gold_idx in visited_gold:
                continue
            visited_gold.add(gold_idx)
            previous_pred = gold_to_pred.get(gold_idx)
            if previous_pred is None or augment(previous_pred, visited_gold):
                gold_to_pred[gold_idx] = pred_idx
                return True
        return False

    for pred_idx in pred_indices:
        augment(pred_idx, set())
    return sorted((pred_idx, gold_idx) for gold_idx, pred_idx in gold_to_pred.items())


def match_counts(pred: List[str], gold: List[str]) -> Dict[str, float]:
    pred_items = unique_normalized_items(pred)
    gold_items = unique_normalized_items(gold)
    pred_keys = [norm_key(item) for item in pred_items]
    gold_keys = [norm_key(item) for item in gold_items]
    pred_tokens = [tokenize_for_match(item) for item in pred_items]
    gold_tokens = [tokenize_for_match(item) for item in gold_items]

    remaining_pred = list(range(len(pred_items)))
    remaining_gold = list(range(len(gold_items)))

    def consume(pairs: List[Tuple[int, int]]) -> None:
        nonlocal remaining_pred, remaining_gold
        used_pred = {pred_idx for pred_idx, _ in pairs}
        used_gold = {gold_idx for _, gold_idx in pairs}
        remaining_pred = [index for index in remaining_pred if index not in used_pred]
        remaining_gold = [index for index in remaining_gold if index not in used_gold]

    exact_pairs = maximum_cardinality_pairs(
        remaining_pred,
        remaining_gold,
        lambda pred_idx, gold_idx: float(pred_keys[pred_idx] == gold_keys[gold_idx]),
        1.0,
    )
    consume(exact_pairs)

    containment_pairs = maximum_cardinality_pairs(
        remaining_pred,
        remaining_gold,
        lambda pred_idx, gold_idx: float(
            is_contiguous_subseq(pred_tokens[pred_idx], gold_tokens[gold_idx])
            or is_contiguous_subseq(gold_tokens[gold_idx], pred_tokens[pred_idx])
        ),
        1.0,
    )
    consume(containment_pairs)

    lexical_pairs = maximum_cardinality_pairs(
        remaining_pred,
        remaining_gold,
        lambda pred_idx, gold_idx: lexical_relaxed_score(
            pred_items[pred_idx], gold_items[gold_idx]
        ),
        ROUGE_TOKEN_F1_THRESHOLD,
    )
    consume(lexical_pairs)

    semantic_pairs = maximum_cardinality_pairs(
        remaining_pred,
        remaining_gold,
        lambda pred_idx, gold_idx: semantic_similarity_score(
            pred_items[pred_idx], gold_items[gold_idx]
        ),
        SEMANTIC_SIM_THRESHOLD,
    )

    exact = len(exact_pairs)
    partial = len(containment_pairs)
    rouge = len(lexical_pairs)
    semantic = len(semantic_pairs)
    pred_count = len(pred_items)
    gold_count = len(gold_items)
    containment_total = exact + partial
    lexical_total = containment_total + rouge
    relaxed_total = lexical_total + semantic

    precision, recall, f1 = prf(exact, pred_count, gold_count)
    containment_precision, containment_recall, containment_f1 = prf(
        containment_total, pred_count, gold_count
    )
    lexical_precision, lexical_recall, lexical_f1 = prf(
        lexical_total, pred_count, gold_count
    )
    relaxed_precision, relaxed_recall, relaxed_f1 = prf(
        relaxed_total, pred_count, gold_count
    )

    return {
        "exact_tp": exact,
        "partial_tp": partial,
        "rouge_tp": rouge,
        "semantic_tp": semantic,
        "tp_total": containment_total,
        "rouge_total": lexical_total,
        "relaxed_tp": relaxed_total,
        "pred_count": pred_count,
        "gold_count": gold_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "containment_precision": containment_precision,
        "containment_recall": containment_recall,
        "containment_f1": containment_f1,
        "rouge_relaxed_precision": lexical_precision,
        "rouge_relaxed_recall": lexical_recall,
        "rouge_relaxed_f1": lexical_f1,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_f1": relaxed_f1,
    }


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)


def build_model_input(row: dict) -> str:
    parts = []
    if isinstance(row.get("headline"), str) and row["headline"].strip():
        parts.append(row["headline"].strip())
    if isinstance(row.get("description"), str) and row["description"].strip():
        parts.append(row["description"].strip())
    return "\n".join(parts).strip()


def build_train_input(row: dict) -> str:
    return build_model_input(row)


def build_test_input(row: dict) -> str:
    return build_model_input(row)


raw_records = list(iter_records_json(INPUT_JSON_PATH))
train_df_all = pd.DataFrame(raw_records)
need_cols = {"id", "description"}
missing = [c for c in need_cols if c not in train_df_all.columns]
if missing:
    raise ValueError(
        f"INPUT JSON missing columns: {missing}. Got={list(train_df_all.columns)}"
    )

for col in ["must_have_list", "nice_to_have_list", "all_skills_list"]:
    if col not in train_df_all.columns:
        train_df_all[col] = [[] for _ in range(len(train_df_all))]

if TRAIN_SKILLS_FIELD not in train_df_all.columns:
    if "all_skills_list" in train_df_all.columns:
        train_df_all[TRAIN_SKILLS_FIELD] = train_df_all["all_skills_list"].apply(
            lambda x: " | ".join(ensure_list(x))
        )
    else:
        train_df_all[TRAIN_SKILLS_FIELD] = ""

train_df_all["must_have_list"] = train_df_all["must_have_list"].apply(ensure_list)
train_df_all["nice_to_have_list"] = train_df_all["nice_to_have_list"].apply(ensure_list)
train_df_all["all_skills_list"] = train_df_all["all_skills_list"].apply(ensure_list)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_items_from_rows(rows: List[dict]) -> Tuple[List[dict], int]:
    items = []
    skipped_no_spans = 0
    for r in rows:
        txt = build_train_input(r)
        if not txt:
            continue
        skills = parse_skills_any(r.get(TRAIN_SKILLS_FIELD, ""))
        if not skills:
            continue
        spans = find_spans(txt, skills)
        if not spans:
            skipped_no_spans += 1
            continue
        items.append({"id": str(r.get("id")), "text": txt, "skills_list": skills})
    return items, skipped_no_spans


def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for text, skills in zip(batch["text"], batch["skills_list"]):
        spans = find_spans(text, skills)
        if not spans:
            continue
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            padding=False,
        )
        for input_ids, attn, offsets in zip(
            enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
        ):
            out["input_ids"].append(input_ids)
            out["attention_mask"].append(attn)
            out["labels"].append(align_bio_labels(offsets, spans))
    return out


@torch.no_grad()
def predict_raw_skills_from_text(model, text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        padding=False,
    )

    predicted_spans: List[Tuple[int, int]] = []
    for input_ids, attn, offsets in zip(
        enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]
    ):
        t_ids = torch.tensor([input_ids], device=DEVICE)
        t_attn = torch.tensor([attn], device=DEVICE)
        logits = model(input_ids=t_ids, attention_mask=t_attn).logits
        pred = torch.argmax(logits, dim=-1)[0].tolist()
        predicted_spans.extend(spans_from_predictions(text, offsets, pred))

    predicted_spans = merge_overlapping_spans(predicted_spans)

    extracted = []
    for s, e in predicted_spans:
        if 0 <= s < e <= len(text):
            s, e = expand_span_to_word_boundaries(text, s, e)
            piece = clean_span_text(text[s:e])
            if piece:
                extracted.append(piece)

    return canonicalize_skills(extracted)


def predict_skills_from_text(model, text: str) -> List[str]:
    return postprocess_predicted_skills(predict_raw_skills_from_text(model, text))


def eval_on_valid_rows_skill_metric(model, valid_rows: List[dict]) -> Dict[str, float]:
    model.eval()
    ps, rs, f1s = [], [], []
    rouge_ps, rouge_rs, rouge_f1s = [], [], []
    relaxed_ps, relaxed_rs, relaxed_f1s = [], [], []
    for r in valid_rows:
        txt = build_model_input(r)
        if not txt:
            continue
        gold = parse_skills_any(r.get(TRAIN_SKILLS_FIELD, ""))
        pred = predict_skills_from_text(model, txt)
        m = match_counts(pred, gold)
        ps.append(m["precision"])
        rs.append(m["recall"])
        f1s.append(m["f1"])
        rouge_ps.append(m["rouge_relaxed_precision"])
        rouge_rs.append(m["rouge_relaxed_recall"])
        rouge_f1s.append(m["rouge_relaxed_f1"])
        relaxed_ps.append(m["relaxed_precision"])
        relaxed_rs.append(m["relaxed_recall"])
        relaxed_f1s.append(m["relaxed_f1"])
    return {
        "precision": float(np.mean(ps)) if ps else 0.0,
        "recall": float(np.mean(rs)) if rs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "rouge_relaxed_precision": float(np.mean(rouge_ps)) if rouge_ps else 0.0,
        "rouge_relaxed_recall": float(np.mean(rouge_rs)) if rouge_rs else 0.0,
        "rouge_relaxed_f1": float(np.mean(rouge_f1s)) if rouge_f1s else 0.0,
        "relaxed_precision": float(np.mean(relaxed_ps)) if relaxed_ps else 0.0,
        "relaxed_recall": float(np.mean(relaxed_rs)) if relaxed_rs else 0.0,
        "relaxed_f1": float(np.mean(relaxed_f1s)) if relaxed_f1s else 0.0,
    }


def run_one_seed(seed: int) -> Dict[str, float]:
    set_all_seeds(seed)

    df = train_df_all.copy()
    train_df, valid_df, test_df = split_from_manifest(df, seed)

    train_rows = train_df.to_dict(orient="records")
    valid_rows = valid_df.to_dict(orient="records")
    test_rows = test_df.to_dict(orient="records")

    valid_rows_hpo = valid_rows
    if HPO_VALID_MAX is not None and len(valid_rows) > HPO_VALID_MAX:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(valid_rows), size=HPO_VALID_MAX, replace=False)
        valid_rows_hpo = [valid_rows[i] for i in idx]

    out_dir = f"./{RUN_NAME}_seed{seed}"
    result_csv = f"{RUN_NAME}_test_results_seed{seed}.csv"
    best_hp_json = f"{RUN_NAME}_best_hyperparams_seed{seed}.json"
    hpo_trials_csv = f"{RUN_NAME}_hpo_trials_seed{seed}.csv"

    train_items, skipped = build_train_items_from_rows(train_rows)
    print(
        f"\n seed={seed} train_rows={len(train_rows)} valid_rows={len(valid_rows)} (HPO_valid={len(valid_rows_hpo)})"
    )
    print(
        f" seed={seed} usable train_items={len(train_items)} |  skipped(no spans)={skipped}"
    )
    if len(train_items) == 0:
        raise RuntimeError(
            "No train_items after weak labeling. Check regex/skills coverage."
        )

    train_ds_raw = Dataset.from_list(train_items)
    train_ds = train_ds_raw.map(
        preprocess_batch, batched=True, remove_columns=train_ds_raw.column_names
    )
    print(f" seed={seed} train chunks: {len(train_ds)}")

    class_weights, class_weight_details = compute_bio_class_weights(train_ds)
    class_weight_path = f"{RUN_NAME}_bio_class_weights_seed{seed}.json"
    with open(class_weight_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "formula": "sqrt(O_count / class_count), clipped to [1, 10]",
                "classes": class_weight_details,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f" seed={seed} BIO class weights: {class_weight_details}")
    print(f" saved: {class_weight_path}")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    best_params = None
    best_score = -1.0
    trial_logs = []

    if DO_HPO:
        print(f" seed={seed} HPO on VALID (exact normalized macro F1)...")

        def objective(trial: optuna.Trial) -> float:
            hp = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", HPO_LR_RANGE[0], HPO_LR_RANGE[1], log=True
                ),
                "per_device_train_batch_size": trial.suggest_categorical(
                    "per_device_train_batch_size", HPO_BATCH_CHOICES
                ),
                "gradient_accumulation_steps": trial.suggest_categorical(
                    "gradient_accumulation_steps", HPO_GRAD_ACC_CHOICES
                ),
                "num_train_epochs": trial.suggest_int(
                    "num_train_epochs", HPO_EPOCH_RANGE[0], HPO_EPOCH_RANGE[1]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", HPO_WD_RANGE[0], HPO_WD_RANGE[1]
                ),
            }

            args = TrainingArguments(
                output_dir=out_dir,
                learning_rate=hp["learning_rate"],
                num_train_epochs=hp["num_train_epochs"],
                per_device_train_batch_size=hp["per_device_train_batch_size"],
                gradient_accumulation_steps=hp["gradient_accumulation_steps"],
                weight_decay=hp["weight_decay"],
                fp16=(DEVICE == "cuda"),
                save_strategy="no",
                logging_steps=200,
                report_to="none",
                remove_unused_columns=False,
                seed=seed,
                data_seed=seed,
            )

            model = None
            trainer = None
            try:
                set_all_seeds(seed)
                model = AutoModelForTokenClassification.from_pretrained(
                    MODEL_NAME,
                    num_labels=len(LABEL2ID),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                ).to(DEVICE)

                trainer = WeightedTokenClassificationTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    data_collator=data_collator,
                    class_weights=class_weights,
                    **trainer_tokenizer_kwargs(tokenizer),
                )

                trainer.train()

                metrics = eval_on_valid_rows_skill_metric(model, valid_rows_hpo)
                trial_logs.append(
                    {
                        **hp,
                        "eval_precision": metrics["precision"],
                        "eval_recall": metrics["recall"],
                        "eval_exact_f1": metrics["f1"],
                        "eval_rouge_relaxed_f1": metrics["rouge_relaxed_f1"],
                        "eval_relaxed_f1": metrics["relaxed_f1"],
                    }
                )
                return metrics["f1"]

            finally:
                try:
                    if trainer is not None:
                        trainer.model = None
                except Exception:
                    pass
                del trainer
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(objective, n_trials=HPO_TRIALS)

        best_params = study.best_params
        best_score = study.best_value

        with open(best_hp_json, "w", encoding="utf-8") as f:
            json.dump(
                {"best_params": best_params, "best_exact_f1": best_score},
                f,
                ensure_ascii=False,
                indent=2,
            )
        pd.DataFrame(trial_logs).to_csv(
            hpo_trials_csv, index=False, encoding="utf-8-sig"
        )
        print(f" seed={seed} best_exact_f1={best_score:.4f} best_params={best_params}")
        print(f" saved: {best_hp_json} | {hpo_trials_csv}")

    final_hp = BASE_TRAIN_DEFAULTS.copy()
    if best_params is not None:
        final_hp.update(best_params)

    final_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=final_hp["learning_rate"],
        num_train_epochs=final_hp["num_train_epochs"],
        per_device_train_batch_size=final_hp["per_device_train_batch_size"],
        gradient_accumulation_steps=final_hp["gradient_accumulation_steps"],
        weight_decay=final_hp["weight_decay"],
        fp16=(DEVICE == "cuda"),
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )

    set_all_seeds(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    trainer = WeightedTokenClassificationTrainer(
        model=model,
        args=final_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        class_weights=class_weights,
        **trainer_tokenizer_kwargs(tokenizer),
    )

    print(f"\n seed={seed} Final training...")
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f" seed={seed} Saved model to: {Path(out_dir).resolve()}")

    model.eval()
    results = []
    ps, rs, f1s = [], [], []
    rouge_ps, rouge_rs, rouge_f1s = [], [], []
    relaxed_ps, relaxed_rs, relaxed_f1s = [], [], []
    sum_exact = sum_partial = sum_total = 0
    sum_rouge = sum_semantic = sum_rouge_total = sum_relaxed = 0
    sum_gold = sum_pred = 0

    for row in tqdm(test_rows, desc=f" Testing (BERT NER, seed={seed})"):
        input_text = build_test_input(row)
        if not input_text:
            continue
        gold = parse_gold_list_field(row.get(TEST_GOLD_LIST_FIELD, []))
        raw_pred = predict_raw_skills_from_text(model, input_text)
        pred = postprocess_predicted_skills(raw_pred)
        filtered_out = removed_by_prediction_filter(raw_pred, pred)
        unsupported = unsupported_predictions(pred, input_text)
        m = match_counts(pred, gold)

        ps.append(m["precision"])
        rs.append(m["recall"])
        f1s.append(m["f1"])
        rouge_ps.append(m["rouge_relaxed_precision"])
        rouge_rs.append(m["rouge_relaxed_recall"])
        rouge_f1s.append(m["rouge_relaxed_f1"])
        relaxed_ps.append(m["relaxed_precision"])
        relaxed_rs.append(m["relaxed_recall"])
        relaxed_f1s.append(m["relaxed_f1"])
        sum_exact += int(m["exact_tp"])
        sum_partial += int(m["partial_tp"])
        sum_rouge += int(m["rouge_tp"])
        sum_semantic += int(m["semantic_tp"])
        sum_total += int(m["tp_total"])
        sum_rouge_total += int(m["rouge_total"])
        sum_relaxed += int(m["relaxed_tp"])
        sum_gold += int(m["gold_count"])
        sum_pred += int(m["pred_count"])

        results.append(
            {
                "id": row.get("id"),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "exact_tp": m["exact_tp"],
                "partial_tp": m["partial_tp"],
                "rouge_tp": m["rouge_tp"],
                "semantic_tp": m["semantic_tp"],
                "tp_total": m["tp_total"],
                "rouge_total": m["rouge_total"],
                "relaxed_tp": m["relaxed_tp"],
                "rouge_relaxed_precision": m["rouge_relaxed_precision"],
                "rouge_relaxed_recall": m["rouge_relaxed_recall"],
                "rouge_relaxed_f1": m["rouge_relaxed_f1"],
                "relaxed_precision": m["relaxed_precision"],
                "relaxed_recall": m["relaxed_recall"],
                "relaxed_f1": m["relaxed_f1"],
                "gold_count": m["gold_count"],
                "pred_count": m["pred_count"],
                "gold": " | ".join(gold),
                "raw_pred": " | ".join(raw_pred),
                "pred": " | ".join(pred),
                "filtered_out_pred": " | ".join(filtered_out),
                "filtered_out_count": len(filtered_out),
                "unsupported_pred": " | ".join(unsupported),
                "unsupported_count": len(unsupported),
                "unsupported_rate": len(unsupported) / len(pred) if pred else 0.0,
            }
        )

    pd.DataFrame(results).to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f" seed={seed} Saved per-row test results: {result_csv}")

    macro_p = float(np.mean(ps)) if ps else 0.0
    macro_r = float(np.mean(rs)) if rs else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_rouge_p = float(np.mean(rouge_ps)) if rouge_ps else 0.0
    macro_rouge_r = float(np.mean(rouge_rs)) if rouge_rs else 0.0
    macro_rouge_f1 = float(np.mean(rouge_f1s)) if rouge_f1s else 0.0
    macro_relaxed_p = float(np.mean(relaxed_ps)) if relaxed_ps else 0.0
    macro_relaxed_r = float(np.mean(relaxed_rs)) if relaxed_rs else 0.0
    macro_relaxed_f1 = float(np.mean(relaxed_f1s)) if relaxed_f1s else 0.0

    exact_rate = float(sum_exact / sum_gold) if sum_gold else 0.0
    partial_rate = float(sum_partial / sum_gold) if sum_gold else 0.0
    total_rate = float(sum_total / sum_gold) if sum_gold else 0.0
    rouge_total_rate = float(sum_rouge_total / sum_gold) if sum_gold else 0.0
    relaxed_rate = float(sum_relaxed / sum_gold) if sum_gold else 0.0

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "seed": seed,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "macro_rouge_relaxed_precision": macro_rouge_p,
        "macro_rouge_relaxed_recall": macro_rouge_r,
        "macro_rouge_relaxed_f1": macro_rouge_f1,
        "macro_relaxed_precision": macro_relaxed_p,
        "macro_relaxed_recall": macro_relaxed_r,
        "macro_relaxed_f1": macro_relaxed_f1,
        "micro_exact_rate_vs_gold": exact_rate,
        "micro_partial_rate_vs_gold": partial_rate,
        "micro_total_rate_vs_gold": total_rate,
        "micro_rouge_total_rate_vs_gold": rouge_total_rate,
        "micro_relaxed_rate_vs_gold": relaxed_rate,
        "micro_gold": float(sum_gold),
        "micro_pred": float(sum_pred),
    }


all_seed_runs = []
for s in SEEDS:
    all_seed_runs.append(run_one_seed(s))

df_runs = pd.DataFrame(all_seed_runs)
df_runs.to_csv(f"{RUN_NAME}_seed_runs_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n Saved: {RUN_NAME}_seed_runs_summary.csv")
print(df_runs)


REPORT_COLS = [
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_rouge_relaxed_precision",
    "macro_rouge_relaxed_recall",
    "macro_rouge_relaxed_f1",
    "macro_relaxed_precision",
    "macro_relaxed_recall",
    "macro_relaxed_f1",
    "micro_exact_rate_vs_gold",
    "micro_partial_rate_vs_gold",
    "micro_total_rate_vs_gold",
    "micro_rouge_total_rate_vs_gold",
    "micro_relaxed_rate_vs_gold",
]

print(f"\n===== FINAL REPORT (BERT NER, seed={SEEDS[0]}) =====")
for c in REPORT_COLS:
    value = float(pd.to_numeric(df_runs[c], errors="coerce").iloc[0])
    print(f"{c:28s}: {value:.4f}")
