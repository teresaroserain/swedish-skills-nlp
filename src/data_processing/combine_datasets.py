import json
import re
from pathlib import Path
from typing import Any, Dict, List


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPOSITORY_ROOT / "data" / "processed"


class CombineConfig:
    def __init__(self, name: str, input_path: Path, source_hint: str | None = None):
        self.name = name
        self.input_path = input_path
        self.out_path = input_path.parent / "dataset_clean_merged_5skills.jsonl"
        self.source_hint = source_hint


CONFIGS = [
    CombineConfig(
        name="IT",
        input_path=PROCESSED_ROOT / "IT" / "dataset_clean.jsonl",
        source_hint="IT_labels.xlsx",
    ),
    CombineConfig(
        name="journal",
        input_path=PROCESSED_ROOT / "journal" / "dataset_clean.jsonl",
    ),
    CombineConfig(
        name="merged",
        input_path=PROCESSED_ROOT / "merged" / "dataset_clean.jsonl",
    ),
]

KEEP_FIELDS = [
    "must_have_list",
    "nice_to_have_list",
]

COMBINE_FIELDS = [
    "hard_skills_list",
    "soft_skills_list",
    "distinct_skills_list",
]

ALL_FIELDS = KEEP_FIELDS + COMBINE_FIELDS


def as_str(x: Any) -> str:
    if x is None:
        return ""

    return str(x)


def normalize(s: str) -> str:
    return " ".join(as_str(s).strip().lower().split())


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = normalize(x)
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(as_str(x).strip())
    return out


def parse_any_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [as_str(x).strip() for x in v if as_str(x).strip()]

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [as_str(x).strip() for x in arr if as_str(x).strip()]
            except Exception:
                pass
        parts = re.split(r"\s*[;\|\n,]+\s*", s)
        return [p.strip() for p in parts if p and p.strip()]

    sx = as_str(v).strip()
    return [sx] if sx else []


def process_one(cfg: CombineConfig) -> None:
    store: Dict[str, Dict[str, Any]] = {}

    total_lines = 0
    if not cfg.input_path.exists():
        print(f" Skip {cfg.name}: missing {cfg.input_path}")
        return

    with open(cfg.input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            row = json.loads(line)

            rid = as_str(row.get("id", "")).strip()
            if not rid:
                continue

            if rid not in store:
                store[rid] = {
                    "id": rid,
                    "headline": as_str(row.get("headline", "")).strip(),
                    "description": as_str(row.get("description", "")).strip(),
                    "Note": row.get("Note", ""),
                    "must_have_list": None,
                    "nice_to_have_list": None,
                    "hard_skills_list": [],
                    "soft_skills_list": [],
                    "distinct_skills_list": [],
                }

            for k in COMBINE_FIELDS:
                store[rid][k].extend(parse_any_list(row.get(k)))

            for k in KEEP_FIELDS:
                if store[rid][k] is None or not store[rid][k]:
                    items = parse_any_list(row.get(k))
                    if items:
                        store[rid][k] = items

            if not as_str(store[rid].get("headline", "")).strip():
                headline = row.get("headline")
                if isinstance(headline, str) and headline.strip():
                    store[rid]["headline"] = headline.strip()

            if not as_str(store[rid].get("description", "")).strip():
                description = row.get("description")
                if isinstance(description, str) and description.strip():
                    store[rid]["description"] = description.strip()

            if not as_str(store[rid].get("Note", "")).strip():
                n = row.get("Note", "")
                store[rid]["Note"] = n

    out_rows = []
    for rid, r in store.items():
        for k in KEEP_FIELDS:
            if r[k] is None:
                r[k] = []

        for k in COMBINE_FIELDS:
            r[k] = dedup_keep_order(r[k])

        merged = []
        for k in ALL_FIELDS:
            merged.extend(r[k])
        merged = dedup_keep_order(merged)

        r["all_skills_list"] = merged
        r["all_skills"] = " | ".join(merged)

        out_rows.append(r)

    with open(cfg.out_path, "w", encoding="utf-8") as file:
        for r in out_rows:
            file.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out_rows)
    avg_all = sum(len(r["all_skills_list"]) for r in out_rows) / n if n else 0.0
    print(" DONE")
    print(" - dataset:", cfg.name)
    if cfg.source_hint:
        print(" - source:", cfg.source_hint)
    print(" - input lines:", total_lines)
    print(" - unique ids:", n)
    print(" - avg all_skills_count:", round(avg_all, 2))
    print(" - output:", cfg.out_path.resolve())


def main() -> None:
    for cfg in CONFIGS:
        process_one(cfg)


if __name__ == "__main__":
    main()
