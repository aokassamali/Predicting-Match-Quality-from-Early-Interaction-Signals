"""
00_data_audit.py

Purpose:
- Sanity-check the Speed Dating dataset so we can design leakage-safe splits,
  candidate sets (wave-scoped), and targets (match + quality proxy).

Why this first:
- If wave/ids/labels are messy or missing, everything downstream breaks.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Speed_Dating_Data.csv"
OUT_PATH = REPO_ROOT / "results" / "data_audit.txt"


def main() -> None:
    # Expectation before run:
    # - The file loads
    # - We have columns like: iid, pid, wave, match, dec, dec_o, like (like may have missing)
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    lines = []
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")
    lines.append("Key columns presence:")
    key_cols = ["iid", "pid", "wave", "match", "dec", "dec_o", "like", "round", "order"]
    for c in key_cols:
        lines.append(f"  {c:>6}: {'YES' if c in df.columns else 'NO'}")

    # Basic uniqueness
    if "iid" in df.columns:
        lines.append("")
        lines.append(f"Unique iid: {df['iid'].nunique():,}")
    if "pid" in df.columns:
        lines.append(f"Unique pid: {df['pid'].nunique():,}")
    if "wave" in df.columns:
        lines.append(f"Unique wave: {df['wave'].nunique():,}")

    # Label distributions
    if "match" in df.columns:
        lines.append("")
        lines.append("match distribution:")
        lines.append(str(df["match"].value_counts(dropna=False)))
        lines.append(f"match rate: {df['match'].mean():.4f}")

    # Missingness snapshot
    lines.append("")
    lines.append("Missingness (selected columns):")
    miss_cols = [c for c in ["match", "dec", "dec_o", "like", "wave", "iid", "pid"] if c in df.columns]
    miss = df[miss_cols].isna().mean().sort_values(ascending=False)
    lines.append(str(miss))

    # Wave-level sanity
    if all(c in df.columns for c in ["wave", "iid", "pid"]):
        lines.append("")
        lines.append("Meetings per (wave, iid): summary")
        counts = df.groupby(["wave", "iid"])["pid"].count()
        lines.append(str(counts.describe()))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit to: {OUT_PATH}")


if __name__ == "__main__":
    main()
