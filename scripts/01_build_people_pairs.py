"""
01_build_people_pairs.py

Purpose
-------
Turn the raw Speed Dating CSV (one row per "iid met pid" interaction) into:

1) people.parquet
   - One row per (wave, iid)
   - Contains stable, pre-event-ish person features (demographics, stated prefs, self-perception, etc.)

2) pairs_enriched.parquet
   - One row per interaction (wave, iid, pid)
   - Has labels (match, like) and both sides' features joined (suffix _u, _v)

Why this script exists
---------------------
- The raw data is "interaction-shaped". For modeling/ranking we need both u and v features in one row.
- We also want a clean artifact we can reuse for baselines + models without repeated merges.

Design decisions
----------------
- We group by (wave, iid), not just iid, because some features may vary by wave (and it’s safer).
- We choose person feature columns by excluding obviously post-date scorecard columns that leak.
  This is conservative; you can widen later after confirming the data key.

Expected output
---------------
- results/people.parquet
- results/pairs.parquet
- results/pairs_enriched.parquet
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "Speed_Dating_Data.csv"
OUT_DIR = REPO_ROOT / "results"

# Core columns we need to define interactions/candidate sets and labels
PAIR_CORE_COLS = ["wave", "iid", "pid", "round", "order", "match", "like", "dec", "dec_o"]

# Columns that are almost certainly post-date scorecard ratings and should NOT be treated as "person features"
# (We will still keep `like` in pairs as a Stage B target.)
POST_DATE_RATING_COLS = {
    # Common "scorecard" columns after meeting
    "attr", "sinc", "intel", "fun", "amb", "shar", "like",
    # Partner versions / derivatives often present
    "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o", "like_o",
}

# Some columns are identifiers/housekeeping — do not treat as person features
ID_LIKE_COLS = {
    "iid", "pid", "wave", "round", "order", "position", "field_cd", "career_c",
    "from", "zipcode", "mn_sat", "tuition", "income", "date", "undergra",
}

# If you want to be extra conservative, exclude anything that looks like it was collected after the date.
# We'll rely primarily on POST_DATE_RATING_COLS, but keep this pattern list to widen later if needed.
EXCLUDE_SUBSTRINGS = [
    "dec",        # decisions made in-date (we keep as labels, not as features)
    "_o",         # partner-specific columns (keep some in pairs if needed, but avoid in people features)
    "prob",       # sometimes a "perceived probability of match" etc (often post-interaction)
]


def first_non_null(series: pd.Series):
    """Return the first non-null value in a series; if all null, return np.nan."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else np.nan


def infer_people_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer a conservative set of "people features" from the raw dataframe.

    We exclude:
    - core pair columns
    - obvious post-date ratings
    - id-like columns
    - columns containing exclude substrings (very conservative)
    """
    cols = []
    pair_core = set(PAIR_CORE_COLS)
    for c in df.columns:
        if c in pair_core:
            continue
        if c in POST_DATE_RATING_COLS:
            continue
        if c in ID_LIKE_COLS:
            continue

        # Exclude columns with suspicious substrings
        lowered = c.lower()
        if any(sub in lowered for sub in EXCLUDE_SUBSTRINGS):
            continue

        cols.append(c)

    # Keep only columns that are not entirely missing
    cols = [c for c in cols if df[c].notna().any()]
    return cols


def build_people(df: pd.DataFrame, people_cols: Sequence[str]) -> pd.DataFrame:
    """
    Build one row per (wave, iid).
    For each feature column, take first non-null value within that group.
    """
    group_cols = ["wave", "iid"]
    # Aggregation dict: first non-null
    agg = {c: first_non_null for c in people_cols}

    people = df[group_cols + list(people_cols)].groupby(group_cols, as_index=False).agg(agg)

    # Sanity: unique key
    assert people.duplicated(subset=group_cols).sum() == 0, "people table has duplicate (wave, iid)"
    return people


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only core pair columns and drop rows missing key ids.
    """
    existing = [c for c in PAIR_CORE_COLS if c in df.columns]
    pairs = df[existing].copy()

    # Drop rows missing key identifiers
    pairs = pairs.dropna(subset=["iid", "pid", "wave"])

    # Ensure types are sane (iid/pid/wave are often numeric in this dataset)
    for c in ["iid", "pid", "wave"]:
        pairs[c] = pd.to_numeric(pairs[c], errors="coerce")

    pairs = pairs.dropna(subset=["iid", "pid", "wave"])

    # Remove self-pairs if any
    pairs = pairs[pairs["iid"] != pairs["pid"]].copy()

    return pairs


def enrich_pairs(pairs: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """
    Join people features twice: once for iid (u) and once for pid (v) within the same wave.
    """
    # Join user features
    merged = pairs.merge(
        people,
        how="left",
        left_on=["wave", "iid"],
        right_on=["wave", "iid"],
        suffixes=("", "_u"),
    )

    # Rename joined columns for user with suffix _u
    # (merge above didn't suffix because keys are same; we suffix manually for clarity)
    user_feature_cols = [c for c in people.columns if c not in ["wave", "iid"]]
    merged = merged.rename(columns={c: f"{c}_u" for c in user_feature_cols})

    # Join candidate features: people keyed by (wave, iid) but our candidate id is `pid`
    people_for_v = people.rename(columns={"iid": "pid"})
    merged = merged.merge(
        people_for_v,
        how="left",
        on=["wave", "pid"],
        suffixes=("", "_v"),
    )

    # Suffix candidate feature columns with _v
    cand_feature_cols = [c for c in people_for_v.columns if c not in ["wave", "pid"]]
    # After the merge, columns from people_for_v appear without suffix (since no collision)
    merged = merged.rename(columns={c: f"{c}_v" for c in cand_feature_cols})

    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Expectation before run:
    # - CSV loads with latin-1 encoding
    # - Key columns exist: iid, pid, wave, match, like
    df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Minimal clean: drop rows with missing pid early (audit showed tiny fraction)
    df = df.dropna(subset=["pid"])

    # Build pairs first
    pairs = build_pairs(df)

    # Infer a conservative feature set for people
    people_cols = infer_people_feature_columns(df)

    # Build people table
    people = build_people(df, people_cols)

    # Enrich pairs with both sides' features
    pairs_enriched = enrich_pairs(pairs, people)

    # Basic sanity checks
    # 1) pairs_enriched should keep same number of rows as pairs (left joins won't expand)
    if len(pairs_enriched) != len(pairs):
        raise ValueError(f"Row count changed after enrichment: pairs={len(pairs)} vs enriched={len(pairs_enriched)}")

    # 2) Ensure labels exist
    for label in ["match", "like"]:
        if label in pairs_enriched.columns:
            pass
        else:
            print(f"WARNING: label column missing: {label}")

    # Save artifacts
    people_path = OUT_DIR / "people.parquet"
    pairs_path = OUT_DIR / "pairs.parquet"
    enriched_path = OUT_DIR / "pairs_enriched.parquet"

    people.to_parquet(people_path, index=False)
    pairs.to_parquet(pairs_path, index=False)
    pairs_enriched.to_parquet(enriched_path, index=False)

    # Print a concise summary
    print("Wrote:")
    print(f"  {people_path}  (rows={len(people):,}, cols={len(people.columns):,})")
    print(f"  {pairs_path}   (rows={len(pairs):,}, cols={len(pairs.columns):,})")
    print(f"  {enriched_path} (rows={len(pairs_enriched):,}, cols={len(pairs_enriched.columns):,})")

    # Helpful: save a column inventory so we can choose the Stage A whitelist precisely
    col_inv_path = OUT_DIR / "column_inventory.txt"
    col_inv_path.write_text("\n".join(pairs_enriched.columns), encoding="utf-8")
    print(f"Also wrote column inventory: {col_inv_path}")


if __name__ == "__main__":
    main()
