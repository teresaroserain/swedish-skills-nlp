from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOTS = ["software_developer_analysis"]
PER_ROW_FILENAME = "per_row.csv"
OUT_SUBFOLDER = "eda_hist"
BINS = 30

METHODS = ["exact", "bm25", "embed", "hybrid"]
MODES = ["no_merge", "merge"]


PLOT_SPECS = [
    ("p1_count", "Histogram: P1 skill count per sample", "p1_count"),
    ("p2_count", "Histogram: P2 skill count per sample", "p2_count"),
    ("matched_total", "Histogram: Overlap skill count per sample", "matched_total"),
    (
        "overlap_ratio_p1",
        "Histogram: Overlap ratio per sample (P1)",
        "overlap_ratio_p1 (0-1)",
    ),
    (
        "overlap_ratio_p2",
        "Histogram: Overlap ratio per sample (P2)",
        "overlap_ratio_p2 (0-1)",
    ),
]


def save_hist(
    series: pd.Series, title: str, xlabel: str, out_path: Path, bins: int = BINS
) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_series_no_merge(df: pd.DataFrame, method: str, field: str) -> pd.Series | None:
    if field in {"p1_count", "p2_count"}:
        return df[field] if field in df.columns else None

    if field == "matched_total":
        cov_p1 = f"{method}_cov_p1"
        cov_p2 = f"{method}_cov_p2"
        if cov_p1 not in df.columns or cov_p2 not in df.columns:
            return None

        return df[[cov_p1, cov_p2]].min(axis=1)

    if field == "overlap_ratio_p1":
        col = f"{method}_ratio_p1"
        return df[col] if col in df.columns else None
    if field == "overlap_ratio_p2":
        col = f"{method}_ratio_p2"
        return df[col] if col in df.columns else None

    return None


def get_series_merge(df: pd.DataFrame, method: str, field: str) -> pd.Series | None:
    if field == "p1_count":
        col = f"p1_merge_count_{method}"
        return df[col] if col in df.columns else None
    if field == "p2_count":
        col = f"p2_merge_count_{method}"
        return df[col] if col in df.columns else None
    if field == "matched_total":
        col = f"overlap_merge_count_{method}"
        return df[col] if col in df.columns else None
    if field == "overlap_ratio_p1":
        col = f"ratio_p1_merge_{method}"
        return df[col] if col in df.columns else None
    if field == "overlap_ratio_p2":
        col = f"ratio_p2_merge_{method}"
        return df[col] if col in df.columns else None

    return None


def plot_for_folder(df: pd.DataFrame, folder: Path, dataset_name: str) -> None:
    out_base = folder / OUT_SUBFOLDER
    out_base.mkdir(parents=True, exist_ok=True)

    has_merge = any(f"p1_merge_count_{m}" in df.columns for m in METHODS)
    modes = MODES if has_merge else ["no_merge"]

    for mode in modes:
        mode_dir = out_base / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for method in METHODS:
            for field, title, xlabel in PLOT_SPECS:
                if mode == "merge":
                    series = get_series_merge(df, method, field)
                else:
                    series = get_series_no_merge(df, method, field)

                if series is None:
                    continue

                fname = f"{method}__{field}.png"
                mode_label = "merge" if mode == "merge" else "no_merge"
                save_hist(
                    series,
                    f"{dataset_name} | {folder.name} | {mode_label} | {method} | {title}",
                    xlabel,
                    mode_dir / fname,
                    bins=BINS,
                )


def main() -> None:
    roots = [BASE_DIR / name for name in DATASET_ROOTS]
    existing_roots = [r for r in roots if r.exists()]
    if not existing_roots:
        roots_str = ", ".join(str(r) for r in roots)
        raise FileNotFoundError(f"No dataset roots found. Checked: {roots_str}")

    processed_any = False

    for root in existing_roots:
        skill_folders = [p for p in root.iterdir() if p.is_dir()]
        if not skill_folders:
            print(f" Skip {root.name}: no skill folders found")
            continue

        for folder in sorted(skill_folders, key=lambda p: p.name):
            per_row_path = folder / PER_ROW_FILENAME
            if not per_row_path.exists():
                print(f" Skip {folder.name}: missing {PER_ROW_FILENAME}")
                continue

            df = pd.read_csv(per_row_path, encoding="utf-8-sig")
            plot_for_folder(df, folder, root.name)
            processed_any = True

            out_dir = folder / OUT_SUBFOLDER
            print(
                f" Saved histograms for '{root.name}/{folder.name}' -> {out_dir.resolve()}"
            )

    if not processed_any:
        raise RuntimeError("No per_row.csv files found to process.")

    print("\n Done.")


if __name__ == "__main__":
    main()
