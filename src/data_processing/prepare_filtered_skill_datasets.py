import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    xlsx_path: str
    sheet_name: str | None
    out_dir: Path
    drop_empty_labels: bool


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="IT",
        xlsx_path=str(
            DATA_ROOT / "raw_annotations" / "IT_10percent_filtered_labels.xlsx"
        ),
        sheet_name=None,
        out_dir=DATA_ROOT / "processed" / "IT",
        drop_empty_labels=False,
    ),
    DatasetConfig(
        name="journal",
        xlsx_path=str(
            DATA_ROOT / "raw_annotations" / "journalist_24_25_filtered_fullLabels.xlsx"
        ),
        sheet_name=None,
        out_dir=DATA_ROOT / "processed" / "journal",
        drop_empty_labels=True,
    ),
]

MERGED_OUT_DIR = DATA_ROOT / "processed" / "merged"


FILTER_NOTES = {"0", "1", 0, 1}


KEEP_MIX = True


DEDUP_LABELS = False

LABEL_COLUMNS = [
    "must_have",
    "nice_to_have",
    "hard_skills",
    "soft_skills",
    "distinct_skills",
]


SKILL_LIST_COLUMNS = {
    "must_have": "must_have_list",
    "nice_to_have": "nice_to_have_list",
    "hard_skills": "hard_skills_list",
    "soft_skills": "soft_skills_list",
    "distinct_skills": "distinct_skills_list",
}


def pick_main_sheet(xlsx_path: str) -> str:
    xls = pd.ExcelFile(xlsx_path)
    best = None
    best_rows = -1
    for sh in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sh)
        if df.shape[0] > best_rows:
            best_rows = df.shape[0]
            best = sh
    return best


def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)

    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    s = s.replace('"', "")

    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_label_list(s: str) -> list[str]:
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                out = [normalize_text(x) for x in arr if str(x).strip()]
                return out
        except Exception:
            pass

    parts = re.split(r"[;,|]\s*", s)
    parts = [normalize_text(p) for p in parts]
    parts = [p for p in parts if p]

    parts = [re.sub(r"\s*\(\s*", " (", p) for p in parts]
    parts = [re.sub(r"\s*\)\s*", ")", p) for p in parts]

    if DEDUP_LABELS:
        seen = set()
        deduped = []
        for p in parts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        return deduped

    return parts


def split_bracketed_commas(s: str) -> list[str]:
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []

    s = normalize_text(s)

    matched_open: set[int] = set()
    matched_close: set[int] = set()
    stack: list[int] = []
    for index, ch in enumerate(s):
        if ch == "[":
            stack.append(index)
        elif ch == "]" and stack:
            matched_open.add(stack.pop())
            matched_close.add(index)

    out = []
    buf = []
    depth = 0

    for index, ch in enumerate(s):
        if ch == "[":
            if index in matched_open:
                depth += 1
            continue
        if ch == "]":
            if index in matched_close:
                depth = max(depth - 1, 0)
            continue

        if depth == 0 and ch in [",", ";", "|"]:
            item = "".join(buf).strip()
            if item:
                out.append(item)
            buf = []
        else:
            buf.append(ch)

    last = "".join(buf).strip()
    if last:
        out.append(last)

    out = [re.sub(r"\s+", " ", x).strip() for x in out if x and x.strip()]

    if DEDUP_LABELS:
        seen = set()
        deduped = []
        for x in out:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                deduped.append(x)
        out = deduped

    return out


def filter_by_note(df: pd.DataFrame) -> pd.DataFrame:
    if "Note" not in df.columns:
        return df

    note = df["Note"].astype(str).str.strip().str.lower()
    drop_mask = note.isin({str(x).lower() for x in FILTER_NOTES})

    if not KEEP_MIX:
        drop_mask = drop_mask | (note == "mix")

    return df.loc[~drop_mask].copy()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip()
    return df.rename(columns=col_map)


def _empty_list_column(n: int) -> list[list[str]]:
    return [[] for _ in range(n)]


def _list_len_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].apply(len) if col in df.columns else pd.Series([0] * len(df))


def drop_empty_label_rows(df: pd.DataFrame) -> pd.DataFrame:
    list_cols = [
        "must_have_list",
        "nice_to_have_list",
        "hard_skills_list",
        "soft_skills_list",
        "distinct_skills_list",
    ]
    available = [c for c in list_cols if c in df.columns]
    if not available:
        return df

    total = None
    for col in available:
        lens = _list_len_series(df, col)
        total = lens if total is None else total + lens

    return df.loc[total > 0].copy()


def build_lists(df: pd.DataFrame) -> pd.DataFrame:
    df["must_have_list"] = (
        df["must_have"].apply(split_bracketed_commas)
        if "must_have" in df.columns
        else _empty_list_column(len(df))
    )
    df["nice_to_have_list"] = (
        df["nice_to_have"].apply(split_bracketed_commas)
        if "nice_to_have" in df.columns
        else _empty_list_column(len(df))
    )

    df["hard_skills_list"] = (
        df["hard_skills"].apply(split_bracketed_commas)
        if "hard_skills" in df.columns
        else _empty_list_column(len(df))
    )
    df["soft_skills_list"] = (
        df["soft_skills"].apply(split_bracketed_commas)
        if "soft_skills" in df.columns
        else _empty_list_column(len(df))
    )
    df["distinct_skills_list"] = (
        df["distinct_skills"].apply(split_bracketed_commas)
        if "distinct_skills" in df.columns
        else _empty_list_column(len(df))
    )

    return df


def drop_duplicate_ids_keep_first(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        return df

    normalized_ids = df["id"].astype(str).str.strip()
    duplicate_mask = normalized_ids.ne("") & normalized_ids.duplicated(keep="first")
    return df.loc[~duplicate_mask].copy()


def write_outputs(model_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "dataset_clean.csv"
    model_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_jsonl = out_dir / "dataset_clean.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in model_df.iterrows():
            obj = row.to_dict()
            for k, v in list(obj.items()):
                if isinstance(v, float) and pd.isna(v):
                    obj[k] = None
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\nSaved:")
    print(" -", out_csv.resolve())
    print(" -", out_jsonl.resolve())


def write_skill_type_slices(
    model_df: pd.DataFrame, out_dir: Path, dataset_name: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices_dir = out_dir / "by_skill_type"
    slices_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [
        c for c in ["id", "headline", "description", "Note"] if c in model_df.columns
    ]

    for skill_type, list_col in SKILL_LIST_COLUMNS.items():
        if list_col not in model_df.columns:
            continue

        df = model_df[base_cols + [list_col]].copy()
        df = df.rename(columns={list_col: "skills_list"})
        df["skill_type"] = skill_type
        df["skill_count"] = df["skills_list"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        df = df[df["skill_count"] > 0].copy()

        out_csv = slices_dir / f"skills_{skill_type}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        out_jsonl = slices_dir / f"skills_{skill_type}.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                obj = row.to_dict()
                for k, v in list(obj.items()):
                    if isinstance(v, float) and pd.isna(v):
                        obj[k] = None
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(
            f" - {dataset_name}: saved {out_csv.name} and {out_jsonl.name} in {slices_dir.resolve()}"
        )


def process_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    sheet = cfg.sheet_name or pick_main_sheet(cfg.xlsx_path)
    print(f"Using sheet: {sheet} ({cfg.name})")

    df = pd.read_excel(cfg.xlsx_path, sheet_name=sheet)
    df = standardize_columns(df)

    print("Original shape:", df.shape)

    df = filter_by_note(df)
    print("After Note filter:", df.shape)

    for col in ["headline", "description"] + LABEL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)

    df = build_lists(df)

    before = len(df)
    df = drop_duplicate_ids_keep_first(df)
    if len(df) != before:
        print(f"After duplicate-ID filter: {before} -> {len(df)}")

    if cfg.drop_empty_labels:
        before = len(df)
        df = drop_empty_label_rows(df)
        print(f"After empty-label filter: {before} -> {len(df)}")

    keep_cols = []
    for c in [
        "id",
        "headline",
        "description",
        "Note",
        "must_have_list",
        "nice_to_have_list",
        "hard_skills_list",
        "soft_skills_list",
        "distinct_skills_list",
    ]:
        if c in df.columns:
            keep_cols.append(c)

    if not keep_cols:
        keep_cols = df.columns.tolist()

    model_df = df[keep_cols].copy()

    write_outputs(model_df, cfg.out_dir)

    write_skill_type_slices(model_df, cfg.out_dir, cfg.name)

    n_mix = 0
    if "Note" in df.columns:
        n_mix = (df["Note"].astype(str).str.strip().str.lower() == "mix").sum()

    print("\nStats:")
    print(" - dataset:", cfg.name)
    print(" - rows:", len(df))
    print(" - mix rows kept:", int(n_mix))
    print(" - avg must_have count:", float(df["must_have_list"].apply(len).mean()))
    print(
        " - avg nice_to_have count:", float(df["nice_to_have_list"].apply(len).mean())
    )
    print(" - avg hard_skills count:", float(df["hard_skills_list"].apply(len).mean()))
    print(" - avg soft_skills count:", float(df["soft_skills_list"].apply(len).mean()))
    print(
        " - avg distinct_skills count:",
        float(df["distinct_skills_list"].apply(len).mean()),
    )

    return model_df


def main() -> None:
    merged_frames: list[pd.DataFrame] = []

    for cfg in DATASETS:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        df = process_dataset(cfg)
        merged_frames.append(df)

    if merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True, sort=False)
        write_outputs(merged_df, MERGED_OUT_DIR)

        write_skill_type_slices(merged_df, MERGED_OUT_DIR, "merged")

        print("\nMerged rows:", len(merged_df))


if __name__ == "__main__":
    main()
