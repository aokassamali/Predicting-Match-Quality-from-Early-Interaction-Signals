"""
11_train_stage_b_ranker.py

Stage B: Predict mutual date quality and rank candidates per (wave, iid).

Label choices
-------------
- y = quality_bin (recommended): binned 0..3 relevance (more stable)
- We also report metrics on raw quality for visibility (optional)

Evaluation
----------
We compute per-(wave,iid) ranking metrics:
- NDCG@K (graded relevance)
- MAP@K (treating "high quality" as relevant in a thresholded way)
- Recall@K (did we surface any "high quality" in top K?)

Important note:
--------------
With graded labels, NDCG is the main KPI. Recall/MAP require defining what counts as "relevant".
We define "relevant" = quality >= 8 (mutual) by default (configurable).

Run
---
python scripts/11_train_stage_b_ranker.py

Outputs
-------
- Prints nested-CV mean±std metrics for K=5 and K=10.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_stage_b_quality.parquet"

RANDOM_SEED_BASE = 42

N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20

K_LIST = [5, 10]

# Relevance threshold for Recall/MAP (binary relevance derived from raw quality)
RELEVANT_QUALITY_THRESHOLD = 8.0

# Early stopping
NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 300


# -------------------------
# Metrics (graded NDCG + thresholded MAP/Recall)
# -------------------------
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


def average_precision_at_k(y_true_sorted_bin: np.ndarray, k: int) -> float:
    y = y_true_sorted_bin[:k]
    if y.sum() == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(y, start=1):
        if rel == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(y_true_sorted_bin: np.ndarray, k: int) -> float:
    return float(y_true_sorted_bin[:k].sum() > 0)


def evaluate_ranking_stage_b(df: pd.DataFrame, score_col: str) -> Dict[int, Dict[str, float]]:
    """
    Uses:
      - graded relevance for NDCG: quality_bin (0..3)
      - binary relevance for MAP/Recall: (quality >= threshold)
    """
    out = {}
    for k in K_LIST:
        ndcgs, maps, recalls = [], [], []

        for (_, _), g in df.groupby(["wave", "iid"], sort=False):
            g_sorted = g.sort_values(score_col, ascending=False)

            rel_graded = g_sorted["quality_bin"].to_numpy(dtype=float)
            # Replace NaN bins with 0 relevance
            rel_graded = np.nan_to_num(rel_graded, nan=0.0)

            rel_bin = (g_sorted["quality"].to_numpy(dtype=float) >= RELEVANT_QUALITY_THRESHOLD).astype(int)

            ndcgs.append(ndcg_at_k(rel_graded, k))
            maps.append(average_precision_at_k(rel_bin, k))
            recalls.append(recall_at_k(rel_bin, k))

        out[k] = {
            "NDCG@K": float(np.mean(ndcgs)),
            "MAP@K": float(np.mean(maps)),
            "Recall@K": float(np.mean(recalls)),
        }
    return out


# -------------------------
# Groups + weights
# -------------------------
def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    sizes_per_row = df_sorted.groupby(["wave", "iid"], sort=False)["pid"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(sizes_per_row, 1.0)


# -------------------------
# Core features (same as Stage A)
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_core_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base_u = ["age_u","gender_u","race_u","field_cd_u","goal_u","imprace_u","imprelig_u","expnum_u","exphappy_u","date_u","go_out_u"]
    base_v = ["age_v","gender_v","race_v","field_cd_v","goal_v","imprace_v","imprelig_v","expnum_v","exphappy_v","date_v","go_out_v"]
    prefs_u = ["attr1_1_u","sinc1_1_u","intel1_1_u","fun1_1_u","amb1_1_u","shar1_1_u"]
    prefs_v = ["attr1_1_v","sinc1_1_v","intel1_1_v","fun1_1_v","amb1_1_v","shar1_1_v"]
    self_u = ["attr3_1_u","sinc3_1_u","intel3_1_u","fun3_1_u","amb3_1_u"]
    self_v = ["attr3_1_v","sinc3_1_v","intel3_1_v","fun3_1_v","amb3_1_v"]

    keep_cols = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)

    cat_guess = {"gender_u","gender_v","race_u","race_v","field_cd_u","field_cd_v","goal_u","goal_v"}
    cat_cols = [c for c in keep_cols if c in cat_guess]
    num_cols = [c for c in keep_cols if c not in cat_cols]

    X = df[keep_cols].copy()

    # compatibility
    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        num_cols.append("abs_age_diff")
    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        num_cols.append("same_race")
    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        num_cols.append("same_field")

    # preference dot products
    trait_map = [
        ("attr","attr1_1_u","attr3_1_v","attr1_1_v","attr3_1_u"),
        ("sinc","sinc1_1_u","sinc3_1_v","sinc1_1_v","sinc3_1_u"),
        ("intel","intel1_1_u","intel3_1_v","intel1_1_v","intel3_1_u"),
        ("fun","fun1_1_u","fun3_1_v","fun1_1_v","fun3_1_u"),
        ("amb","amb1_1_u","amb3_1_v","amb1_1_v","amb3_1_u"),
    ]
    u_pref, v_self, v_pref, u_self = [], [], [], []
    for _, pu, sv, pv, su in trait_map:
        if pu in df.columns and sv in df.columns:
            u_pref.append(pu); v_self.append(sv)
        if pv in df.columns and su in df.columns:
            v_pref.append(pv); u_self.append(su)

    if u_pref and v_self:
        pu = np.nan_to_num(df[u_pref].to_numpy(float), nan=0.0)
        sv = np.nan_to_num(df[v_self].to_numpy(float), nan=0.0)
        X["pref_dot_self_v"] = np.sum(pu * sv, axis=1)
        num_cols.append("pref_dot_self_v")

    if v_pref and u_self:
        pv = np.nan_to_num(df[v_pref].to_numpy(float), nan=0.0)
        su = np.nan_to_num(df[u_self].to_numpy(float), nan=0.0)
        X["pref_v_dot_self_u"] = np.sum(pv * su, axis=1)
        num_cols.append("pref_v_dot_self_u")

    # dedup
    def dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return X, dedup(num_cols), dedup(cat_cols)


# -------------------------
# Splits
# -------------------------
def make_folds(waves: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[List[int]]:
    waves = np.array(sorted(waves))
    rng.shuffle(waves)
    folds = np.array_split(waves, n_folds)
    return [f.tolist() for f in folds]


def inner_train_val_split(remaining_waves: List[int], val_frac: float, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    waves = np.array(remaining_waves, dtype=int)
    rng.shuffle(waves)
    n_val = max(1, int(len(waves) * val_frac))
    val = waves[:n_val].tolist()
    train = waves[n_val:].tolist()
    if len(train) == 0:
        train = waves[1:].tolist()
        val = waves[:1].tolist()
    return train, val


# -------------------------
# Minimal grid (small, regularized)
# -------------------------
@dataclass(frozen=True)
class Config:
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    max_depth: int

    def to_params(self, seed: int) -> Dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "verbosity": -1,
            "seed": seed,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
        }

    def id(self) -> str:
        return (
            f"lr={self.learning_rate}_leaves={self.num_leaves}_minleaf={self.min_data_in_leaf}_"
            f"ff={self.feature_fraction}_bf={self.bagging_fraction}_l2={self.lambda_l2}_md={self.max_depth}"
        )


def make_param_grid() -> List[Config]:
    learning_rates = [0.03]
    num_leaves = [15, 31]
    min_data_in_leaf = [40, 80, 150]
    lambda_l2 = [1.0, 5.0, 20.0]
    max_depth = [8]
    feature_fraction = [0.85]
    bagging_fraction = [0.85]

    grid = []
    for lr, nl, mdl, l2, md, ff, bf in product(
        learning_rates, num_leaves, min_data_in_leaf, lambda_l2, max_depth, feature_fraction, bagging_fraction
    ):
        grid.append(Config(lr, nl, mdl, ff, bf, l2, md))
    return grid


def train_eval_one(df_all: pd.DataFrame, train_waves: List[int], val_waves: List[int], test_waves: List[int], cfg: Config, seed: int):
    train_df = df_all[df_all["wave"].isin(train_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)
    val_df = df_all[df_all["wave"].isin(val_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)
    test_df = df_all[df_all["wave"].isin(test_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)

    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _ = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    # Labels: use quality_bin as training signal (graded but small)
    y_train = np.nan_to_num(train_df["quality_bin"].to_numpy(float), nan=0.0)
    y_val = np.nan_to_num(val_df["quality_bin"].to_numpy(float), nan=0.0)

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    group_train = compute_groups(train_df)
    group_val = compute_groups(val_df)

    w_train = compute_row_weights_equal_query(train_df)
    w_val = compute_row_weights_equal_query(val_df)

    dtrain = lgb.Dataset(Z_train, label=y_train, group=group_train, weight=w_train, free_raw_data=False)
    dval = lgb.Dataset(Z_val, label=y_val, group=group_val, weight=w_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        cfg.to_params(seed),
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train","val"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)

    val_eval = val_df[["wave","iid","pid","quality","quality_bin"]].copy()
    val_eval["score"] = val_scores

    test_eval = test_df[["wave","iid","pid","quality","quality_bin"]].copy()
    test_eval["score"] = test_scores

    val_metrics = evaluate_ranking_stage_b(val_eval, "score")
    test_metrics = evaluate_ranking_stage_b(test_eval, "score")

    return val_metrics, test_metrics, booster.best_iteration


def main() -> None:
    df = pd.read_parquet(DATA_PATH).copy()
    df = df.dropna(subset=["wave","iid","pid","quality"]).copy()
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype(int)

    all_waves = df["wave"].unique()
    grid = make_param_grid()

    rows = []

    for cfg in grid:
        cfg_id = cfg.id()
        print(f"Config: {cfg_id}")

        for r in range(N_REPEATS):
            seed = RANDOM_SEED_BASE + r
            rng = np.random.default_rng(seed)
            folds = make_folds(all_waves, N_FOLDS, rng)

            for fold_idx in range(N_FOLDS):
                test_waves = folds[fold_idx]
                remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]
                train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

                val_m, test_m, best_iter = train_eval_one(df, train_waves, val_waves, test_waves, cfg, seed)

                for k in K_LIST:
                    rows.append({
                        "config_id": cfg_id,
                        "repeat": r,
                        "fold": fold_idx,
                        "k": k,
                        "val_ndcg": val_m[k]["NDCG@K"],
                        "val_map": val_m[k]["MAP@K"],
                        "val_recall": val_m[k]["Recall@K"],
                        "test_ndcg": test_m[k]["NDCG@K"],
                        "test_map": test_m[k]["MAP@K"],
                        "test_recall": test_m[k]["Recall@K"],
                        "best_iteration": best_iter,
                    })

    res = pd.DataFrame(rows)

    # Aggregate (just print summary)
    def summarize(k: int) -> pd.DataFrame:
        sub = res[res["k"] == k]
        return sub.groupby("config_id")[["test_ndcg","test_map","test_recall"]].agg(["mean","std"]).sort_values(("test_ndcg","mean"), ascending=False)

    print("\n===== STAGE B RESULTS (sorted by test NDCG mean; do NOT use test for selection) =====")
    for k in K_LIST:
        s = summarize(k).head(10)
        print(f"\nTop 10 by TEST NDCG@{k} (for inspection):")
        print(s)

    # Proper selection metric: inner-val NDCG@5 mean
    sel = res[res["k"] == 5].groupby("config_id")["val_ndcg"].mean().sort_values(ascending=False)
    best = sel.index[0]
    print(f"\nSelected BEST by INNER-VAL mean NDCG@5: {best}")

    # Report BEST's test mean±std
    best_rows = res[(res["config_id"] == best)]
    for k in K_LIST:
        sub = best_rows[best_rows["k"] == k]
        print(
            f"K={k} | TEST NDCG {sub['test_ndcg'].mean():.4f} ± {sub['test_ndcg'].std():.4f} | "
            f"TEST MAP {sub['test_map'].mean():.4f} ± {sub['test_map'].std():.4f} | "
            f"TEST Recall {sub['test_recall'].mean():.4f} ± {sub['test_recall'].std():.4f}"
        )


if __name__ == "__main__":
    main()
