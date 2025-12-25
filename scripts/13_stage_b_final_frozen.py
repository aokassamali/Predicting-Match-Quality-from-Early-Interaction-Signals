"""
13_stage_b_final_frozen.py

Runs Stage B final evaluation using the frozen best config (no grid search).
Assumes you already created results/pairs_stage_b_quality.parquet (Script 10).

Outputs:
- prints baseline + model metrics (outer-test mean±std)
- writes a small CSV to results/stage_b_final_metrics.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Import from your Script 12 (adjust import paths if needed)
# These must exist in your codebase (they do in your Script 12).
from stage_b_lib import (  # <- if you don't have this, see note below
    make_folds,
    inner_train_val_split,
    fit_predict_lgbm,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_stage_b_quality.parquet"
OUT_CSV = REPO_ROOT / "results" / "stage_b_final_metrics.csv"

RANDOM_SEED_BASE = 7
N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20

# Frozen config from your run:
FINAL_CFG = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 150,
    "lambda_l2": 20.0,
    "max_depth": 8,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
}

K_LIST = [5, 10]


class FrozenConfig:
    # minimal wrapper matching your fit_predict_lgbm expectations (cfg.leaves/minleaf/l2)
    def __init__(self, d: dict):
        self.lr = d["learning_rate"]
        self.leaves = d["num_leaves"]
        self.minleaf = d["min_data_in_leaf"]
        self.l2 = d["lambda_l2"]
        self.md = d["max_depth"]
        self.ff = d["feature_fraction"]
        self.bf = d["bagging_fraction"]

    def params(self, seed: int) -> dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "verbosity": -1,
            "seed": seed,
            "learning_rate": self.lr,
            "num_leaves": self.leaves,
            "min_data_in_leaf": self.minleaf,
            "feature_fraction": self.ff,
            "bagging_fraction": self.bf,
            "bagging_freq": 1,
            "lambda_l2": self.l2,
            "max_depth": self.md,
            "label_gain": list(range(0, 11)),
        }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run: python scripts/10_build_stage_b_quality_dataset.py first."
        )

    df = pd.read_parquet(DATA_PATH).copy()
    df = df.dropna(subset=["wave", "iid", "pid", "quality"]).copy()
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype(int)
    if "quality_bin" not in df.columns:
        df["quality_bin"] = 0.0
    df["quality_bin"] = np.nan_to_num(df["quality_bin"].to_numpy(float), nan=0.0)
    df["quality_grade"] = np.clip(np.rint(pd.to_numeric(df["quality"], errors="coerce")), 0, 10).astype(int)

    all_waves = df["wave"].unique()
    cfg = FrozenConfig(FINAL_CFG)

    rows = []

    for r in range(N_REPEATS):
        seed = RANDOM_SEED_BASE + r
        rng = np.random.default_rng(seed)
        folds = make_folds(all_waves, N_FOLDS, rng)

        for fold_idx in range(N_FOLDS):
            test_waves = folds[fold_idx]
            remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]
            train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

            train_df = df[df["wave"].isin(train_waves)]
            val_df = df[df["wave"].isin(val_waves)]
            test_df = df[df["wave"].isin(test_waves)]

            rng_fold = np.random.default_rng(seed * 1000 + fold_idx)

            # Baseline: random
            test_eval_rand = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            test_eval_rand["score"] = rng_fold.random(len(test_eval_rand))
            m_rand = evaluate(test_eval_rand, "score", graded_col="quality_grade")

            # Baseline: pid mean (train-only) + jitter tie-break
            pid_mean_train = train_df.groupby("pid")["quality"].mean()
            global_mean = float(train_df["quality"].mean())
            test_eval_pid = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            test_eval_pid["score"] = test_eval_pid["pid"].map(pid_mean_train).fillna(global_mean)
            test_eval_pid["score"] += 1e-6 * rng_fold.random(len(test_eval_pid))
            m_pid = evaluate(test_eval_pid, "score", graded_col="quality_grade")

            # Model
            val_scores, test_scores = fit_predict_lgbm(train_df, val_df, test_df, "quality", cfg, seed)

            val_eval = val_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            test_eval = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            val_eval["score"] = val_scores
            test_eval["score"] = test_scores

            val_m = evaluate(val_eval, "score", graded_col="quality_grade")
            test_m = evaluate(test_eval, "score", graded_col="quality_grade")

            for k in K_LIST:
                rows.append({
                    "repeat": r,
                    "fold": fold_idx,
                    "k": k,
                    "baseline_random_ndcg": m_rand["ndcg"][k],
                    "baseline_pidmean_ndcg": m_pid["ndcg"][k],
                    "model_val_ndcg": val_m["ndcg"][k],
                    "model_test_ndcg": test_m["ndcg"][k],
                    "model_test_recall_ge7": test_m["recall"][7.0][k],
                    "model_test_recall_ge8": test_m["recall"][8.0][k],
                })

    res = pd.DataFrame(rows)

    def summarize(name: str, col: str, k: int) -> str:
        sub = res[res["k"] == k]
        return f"{name} {sub[col].mean():.4f} ± {sub[col].std():.4f}"

    print("\n===== STAGE B FINAL (Frozen Config) =====")
    print("Config:", FINAL_CFG)

    for k in K_LIST:
        sub = res[res["k"] == k]
        print(f"\nK={k}")
        print("  Random baseline NDCG:", f"{sub['baseline_random_ndcg'].mean():.4f} ± {sub['baseline_random_ndcg'].std():.4f}")
        print("  Pid-mean baseline NDCG:", f"{sub['baseline_pidmean_ndcg'].mean():.4f} ± {sub['baseline_pidmean_ndcg'].std():.4f}")
        print("  Model TEST NDCG:", f"{sub['model_test_ndcg'].mean():.4f} ± {sub['model_test_ndcg'].std():.4f}")
        print("  Model TEST Recall>=7:", f"{sub['model_test_recall_ge7'].mean():.4f}")
        print("  Model TEST Recall>=8:", f"{sub['model_test_recall_ge8'].mean():.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
