"""
10_build_stage_b_quality_dataset.py

Goal
----
Create a Stage B dataset for "match quality" ranking.

Stage B label (Option 1):
- quality = min(like, like_o)

Why this label?
---------------
12-year-old analogy:
- A "good date" is only good if BOTH people liked it.
- If one person hated it, it wasn't a good match.

Reality:
- Using min(like, like_o) gives a graded, mutual satisfaction signal.
- Graded labels work better with NDCG than binary labels.

Leakage rule
------------
Features must be *pre-date* only. We keep the same "core" feature set used in Stage A.
We DO NOT include any post-date ratings or outcomes as features.
The label itself is post-date (that's allowed).

Input
-----
- results/pairs_enriched.parquet (from your earlier steps)

Output
------
- results/pairs_stage_b_quality.parquet
  Columns:
    wave, iid, pid, quality, quality_bin (optional), plus core features

Run
---
python scripts/10_build_stage_b_quality_dataset.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "Speed_Dating_Data.csv"
IN_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_PATH = REPO_ROOT / "results" / "pairs_stage_b_quality.parquet"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = [str(c).strip().lower() for c in df.columns]
    # guard against collisions like "Like" and "like"
    if len(set(new_cols)) != len(new_cols):
        dupes = pd.Series(new_cols)[pd.Series(new_cols).duplicated()].unique().tolist()
        raise ValueError(f"Column-name collision after lowercasing: {dupes}")
    df = df.copy()
    df.columns = new_cols
    return df



def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_core_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns we will keep as FEATURES (pre-date only).
    This mirrors the 'core' set from Stage A.
    """
    base_u = [
        "age_u", "gender_u", "race_u", "field_cd_u", "goal_u",
        "imprace_u", "imprelig_u", "expnum_u", "exphappy_u",
        "date_u", "go_out_u",
    ]
    base_v = [
        "age_v", "gender_v", "race_v", "field_cd_v", "goal_v",
        "imprace_v", "imprelig_v", "expnum_v", "exphappy_v",
        "date_v", "go_out_v",
    ]
    prefs_u = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
    prefs_v = ["attr1_1_v", "sinc1_1_v", "intel1_1_v", "fun1_1_v", "amb1_1_v", "shar1_1_v"]
    self_u = ["attr3_1_u", "sinc3_1_u", "intel3_1_u", "fun3_1_u", "amb3_1_u"]
    self_v = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v"]

    feature_cols = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)

    # We'll also compute compatibility features later in the training script,
    # but it's fine to keep raw columns here.
    return feature_cols


def ensure_like_columns(df_pairs: pd.DataFrame) -> pd.DataFrame:
    # df_pairs is already normalized to lowercase
    if "like" in df_pairs.columns and "like_o" in df_pairs.columns:
        return df_pairs

    if not RAW_PATH.exists():
        raise ValueError(f"Raw CSV not found at {RAW_PATH}")

    raw = normalize_columns(pd.read_csv(RAW_PATH, encoding_errors="ignore", low_memory=False))

    needed = {"wave", "iid", "pid", "like", "like_o"}
    missing = needed - set(raw.columns)
    if missing:
        cand = [c for c in raw.columns if "like" in c]
        raise ValueError(f"Raw CSV missing {sorted(missing)}. 'like' candidates: {cand[:30]}")

    raw_lab = raw[["wave", "iid", "pid", "like", "like_o"]].copy()

    # If raw has duplicates for (wave,iid,pid), keep first (or you can aggregate)
    raw_lab = raw_lab.drop_duplicates(["wave", "iid", "pid"], keep="first")

    # IMPORTANT: drop any existing like/like_o in pairs to avoid like_x/like_y
    df_clean = df_pairs.drop(columns=[c for c in ["like", "like_o"] if c in df_pairs.columns], errors="ignore")

    out = df_clean.merge(raw_lab, on=["wave", "iid", "pid"], how="left", validate="many_to_one")

    if "like" not in out.columns or "like_o" not in out.columns:
        raise ValueError(f"Post-merge missing like columns. Columns now include: {[c for c in out.columns if 'like' in c]}")

    return out


def make_quality_label(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    like = pd.to_numeric(df["like"], errors="coerce")
    like_o = pd.to_numeric(df["like_o"], errors="coerce")

    quality = np.minimum(like, like_o)

    quality_bin = pd.cut(
        quality,
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("float")

    return quality, quality_bin



def main() -> None:
    df = pd.read_parquet(IN_PATH).copy()
    df = normalize_columns(pd.read_parquet(IN_PATH))
    df = ensure_like_columns(df)


    # Keep identifiers + label inputs
    required_ids = ["wave", "iid", "pid"]
    missing = [c for c in required_ids if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required id columns: {missing}")

    # Build label
    quality, quality_bin = make_quality_label(df)

    out = df[required_ids].copy()
    out["quality"] = quality
    out["quality_bin"] = quality_bin

    # Keep core feature cols only (pre-date)
    feature_cols = build_core_feature_cols(df)
    out = pd.concat([out, df[feature_cols]], axis=1)

    # Drop rows without a label
    before = len(out)
    out = out.dropna(subset=["quality"]).reset_index(drop=True)
    after = len(out)

    # Basic sanity prints
    print(f"Loaded: {before:,} rows | Kept (non-null quality): {after:,}")
    print("Quality stats (min of like/like_o):")
    print(out["quality"].describe())

    # Check group sizes
    g = out.groupby(["wave", "iid"]).size()
    print(f"Queries (wave,iid): {len(g):,}")
    print(f"Avg candidates per query: {g.mean():.2f} | median: {g.median():.0f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
