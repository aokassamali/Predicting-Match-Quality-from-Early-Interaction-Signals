"""
02b_baseline_wave_split.py

Purpose:
- Evaluate the dot-product baseline under the SAME wave split used by logreg.
- This makes comparisons fair (out-of-sample vs out-of-sample).

Outputs:
- results/baseline_wave_split_metrics.txt
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_PATH = REPO_ROOT / "results" / "baseline_wave_split_metrics.txt"

RANDOM_SEED = 42
TEST_WAVE_FRAC = 0.30
K_LIST = [5, 10]

PREF_COLS_U = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
SELF_COLS_V = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v"]  # shar3_1_v not present


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
    return float(y_true_sorted[:k].sum() > 0)


def evaluate(df: pd.DataFrame, score_col: str) -> dict:
    metrics = {k: {"recall": [], "ndcg": [], "map": []} for k in K_LIST}
    for (_, _), g in df.groupby(["wave", "iid"], sort=False):
        g_sorted = g.sort_values(score_col, ascending=False)
        y_sorted = g_sorted["match"].to_numpy(dtype=int)
        for k in K_LIST:
            metrics[k]["recall"].append(recall_at_k(y_sorted, k))
            metrics[k]["ndcg"].append(ndcg_at_k(y_sorted, k))
            metrics[k]["map"].append(average_precision_at_k(y_sorted, k))
    out = {}
    for k in K_LIST:
        out[k] = {
            "Recall@K": float(np.mean(metrics[k]["recall"])),
            "NDCG@K": float(np.mean(metrics[k]["ndcg"])),
            "MAP@K": float(np.mean(metrics[k]["map"])),
        }
    return out


def main() -> None:
    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    # wave split identical logic
    waves = np.array(sorted(df["wave"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(waves)
    n_test = max(1, int(len(waves) * TEST_WAVE_FRAC))
    test_waves = set(waves[:n_test])

    test_df = df[df["wave"].isin(test_waves)].copy()

    # compute dot-product baseline score (NaNs -> 0)
    pref = np.nan_to_num(test_df[PREF_COLS_U].to_numpy(dtype=float), nan=0.0)
    selfv = np.nan_to_num(test_df[SELF_COLS_V].to_numpy(dtype=float), nan=0.0)

    # align dims: pref has 6, selfv has 5; drop shar weight to match
    pref = pref[:, :5]
    test_df["score"] = np.sum(pref * selfv, axis=1)

    metrics = evaluate(test_df, "score")

    lines = []
    lines.append(f"Test waves: {sorted(test_waves)}")
    lines.append(f"Test rows: {len(test_df):,} | Test match rate: {test_df['match'].mean():.4f}")
    lines.append("Baseline dot-product ranking metrics on TEST waves:")
    for k, vals in metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
