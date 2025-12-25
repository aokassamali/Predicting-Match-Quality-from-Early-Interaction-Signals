"""
02_baseline_ranker.py

Purpose
-------
Compute a simple, explainable baseline recommender:
- Candidate set: all pids each iid met within the same wave
- Score: dot(pref_u, self_v)
- Evaluate ranking quality for mutual match (match) using Recall@K, NDCG@K, MAP@K

Why this now
------------
Before training ML, we want:
- a working end-to-end evaluation harness
- a baseline number to beat
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_PATH = REPO_ROOT / "results" / "baseline_metrics.txt"

K_LIST = [5, 10]

PREF_COLS_U = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
SELF_COLS_V = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v", "shar3_1_v"]


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def ndcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(y_true_sorted, k)
    ideal = np.sort(y_true_sorted)[::-1]
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def average_precision_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    y = y_true_sorted[:k]
    if y.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(y, start=1):
        if rel == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    # "did we retrieve any positives in top-k?" (user-level hit rate)
    return float(y_true_sorted[:k].sum() > 0)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    # Expectation before run:
    # - Columns wave, iid, pid exist
    # - match exists (binary)
    needed = ["wave", "iid", "pid", "match"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check baseline feature availability
    missing_pref = [c for c in PREF_COLS_U if c not in df.columns]
    missing_self = [c for c in SELF_COLS_V if c not in df.columns]

    lines = []
    lines.append(f"Loaded: {DATA_PATH}")
    lines.append(f"Rows: {len(df):,}  Cols: {len(df.columns):,}")
    lines.append("")

    if missing_pref or missing_self:
        lines.append("WARNING: Missing baseline feature columns.")
        if missing_pref:
            lines.append(f"  Missing pref_u: {missing_pref}")
        if missing_self:
            lines.append(f"  Missing self_v: {missing_self}")
        lines.append("")
        lines.append("Fallback baseline: popularity within wave (rank by candidate's overall match rate in wave).")

        # Popularity baseline: candidate's mean match when they appear as pid within wave
        # Note: this uses outcome label; acceptable as a naive baseline but call it out in report.
        pop = df.groupby(["wave", "pid"])["match"].mean().rename("pop_score").reset_index()
        df = df.merge(pop, on=["wave", "pid"], how="left")
        df["score"] = df["pop_score"].fillna(0.0)
    else:
        # Dot product compatibility
        pref = df[PREF_COLS_U].to_numpy(dtype=float)
        selfv = df[SELF_COLS_V].to_numpy(dtype=float)

        # Handle missing values: treat NaNs as 0 contribution
        pref = np.nan_to_num(pref, nan=0.0)
        selfv = np.nan_to_num(selfv, nan=0.0)

        df["score"] = np.sum(pref * selfv, axis=1)

    # Evaluate per (wave, iid) candidate set
    metrics = {k: {"recall": [], "ndcg": [], "map": []} for k in K_LIST}

    grouped = df.groupby(["wave", "iid"], sort=False)
    for (_, _), g in grouped:
        # Sort candidates by descending score
        g_sorted = g.sort_values("score", ascending=False)
        y_sorted = g_sorted["match"].to_numpy(dtype=int)

        for k in K_LIST:
            metrics[k]["recall"].append(recall_at_k(y_sorted, k))
            metrics[k]["ndcg"].append(ndcg_at_k(y_sorted, k))
            metrics[k]["map"].append(average_precision_at_k(y_sorted, k))

    lines.append("Baseline ranking metrics for mutual match (match):")
    for k in K_LIST:
        lines.append(
            f"K={k:>2} | "
            f"Recall@K={np.mean(metrics[k]['recall']):.4f} | "
            f"NDCG@K={np.mean(metrics[k]['ndcg']):.4f} | "
            f"MAP@K={np.mean(metrics[k]['map']):.4f}"
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote baseline metrics to: {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
