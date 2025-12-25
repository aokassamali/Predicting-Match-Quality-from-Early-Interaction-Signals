"""
08b_tune_lgbm_ranker_nested_cv.py

Goal
----
Do *proper* hyperparameter tuning for a LightGBM LambdaRank (LambdaMART-style) ranker
using nested cross-validation over waves, with per-query weighting (each (wave,iid) sums to 1).

Conceptual overview (intuition first)
-------------------------------------
We have three layers of "not cheating":

1) Outer CV (test folds):
   - Pick a set of waves as TEST.
   - These are *never* used for training decisions.

2) Inner split (train vs val):
   - From the remaining waves (not test), we split into TRAIN and VAL.
   - We use VAL for early stopping and for selecting hyperparameters.

3) Repeat:
   - We reshuffle waves multiple times (repeats) to reduce randomness.

Analogy
-------
- Outer TEST folds = the real exam you never peek at.
- Inner VAL = practice exams for deciding when to stop studying and which strategy works.
- Hyperparameters = study strategy knobs (how complex, how cautious, etc.).
- Repeats = taking multiple exam variants so your score isn't luck.

What this script outputs
------------------------
- results/tuning_nested_cv_raw.csv
  Raw results per (config, repeat, fold, k)

- results/tuning_nested_cv_summary.csv
  Aggregated mean±std of TEST metrics per config, plus mean inner-VAL NDCG@10

- results/tuning_nested_cv_best.txt
  Human-readable summary of the best config

Runtime control
---------------
- N_REPEATS and N_FOLDS set compute.
- PARAM_GRID size set compute.

Start with defaults; if slow, reduce N_REPEATS or grid size.
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


# -------------------------
# Paths / constants
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"

RAW_CSV = OUT_DIR / "tuning_nested_cv_raw.csv"
SUMMARY_CSV = OUT_DIR / "tuning_nested_cv_summary.csv"
BEST_TXT = OUT_DIR / "tuning_nested_cv_best.txt"

RANDOM_SEED_BASE = 42

# Nested CV controls
N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20

# Ranking eval points
K_LIST = [5, 10]

FEATURE_SET = "core"

# Early stopping controls
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 300


# -------------------------
# Ranking metrics
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


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str = "match") -> Dict[int, Dict[str, float]]:
    metrics = {k: {"recall": [], "ndcg": [], "map": []} for k in K_LIST}
    for (_, _), g in df.groupby(["wave", "iid"], sort=False):
        g_sorted = g.sort_values(score_col, ascending=False)
        y_sorted = g_sorted[label_col].to_numpy(dtype=int)
        for k in K_LIST:
            metrics[k]["recall"].append(recall_at_k(y_sorted, k))
            metrics[k]["ndcg"].append(ndcg_at_k(y_sorted, k))
            metrics[k]["map"].append(average_precision_at_k(y_sorted, k))

    out: Dict[int, Dict[str, float]] = {}
    for k in K_LIST:
        out[k] = {
            "Recall@K": float(np.mean(metrics[k]["recall"])),
            "NDCG@K": float(np.mean(metrics[k]["ndcg"])),
            "MAP@K": float(np.mean(metrics[k]["map"])),
        }
    return out


# -------------------------
# Ranking group helpers
# -------------------------
def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    # weight(row) = 1 / group_size, so each group sums to 1
    sizes_per_row = df_sorted.groupby(["wave", "iid"], sort=False)["pid"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(sizes_per_row, 1.0)


# -------------------------
# Features: core only
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_core_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
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

    keep_cols = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)

    categorical_guess = {"gender_u", "gender_v", "race_u", "race_v", "field_cd_u", "field_cd_v", "goal_u", "goal_v"}
    categorical_cols = [c for c in keep_cols if c in categorical_guess]
    numeric_cols = [c for c in keep_cols if c not in categorical_cols]

    X = df[keep_cols].copy()

    # Compatibility features
    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        numeric_cols.append("abs_age_diff")

    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        numeric_cols.append("same_race")

    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        numeric_cols.append("same_field")

    # Preference dot products
    trait_map = [
        ("attr", "attr1_1_u", "attr3_1_v", "attr1_1_v", "attr3_1_u"),
        ("sinc", "sinc1_1_u", "sinc3_1_v", "sinc1_1_v", "sinc3_1_u"),
        ("intel", "intel1_1_u", "intel3_1_v", "intel1_1_v", "intel3_1_u"),
        ("fun", "fun1_1_u", "fun3_1_v", "fun1_1_v", "fun3_1_u"),
        ("amb", "amb1_1_u", "amb3_1_v", "amb1_1_v", "amb3_1_u"),
    ]

    u_pref, v_self, v_pref, u_self = [], [], [], []
    for _, pu, sv, pv, su in trait_map:
        if pu in df.columns and sv in df.columns:
            u_pref.append(pu); v_self.append(sv)
        if pv in df.columns and su in df.columns:
            v_pref.append(pv); u_self.append(su)

    if u_pref and v_self and len(u_pref) == len(v_self):
        pu = np.nan_to_num(df[u_pref].to_numpy(dtype=float), nan=0.0)
        sv = np.nan_to_num(df[v_self].to_numpy(dtype=float), nan=0.0)
        X["pref_dot_self_v"] = np.sum(pu * sv, axis=1)
        numeric_cols.append("pref_dot_self_v")

    if v_pref and u_self and len(v_pref) == len(u_self):
        pv = np.nan_to_num(df[v_pref].to_numpy(dtype=float), nan=0.0)
        su = np.nan_to_num(df[u_self].to_numpy(dtype=float), nan=0.0)
        X["pref_v_dot_self_u"] = np.sum(pv * su, axis=1)
        numeric_cols.append("pref_v_dot_self_u")

    # Dedup lists
    def dedup(seq: List[str]) -> List[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return X, dedup(numeric_cols), dedup(categorical_cols)


def leakage_tripwire(feature_cols: List[str]) -> None:
    forbidden_exact = {"match", "dec", "dec_o", "like", "attr", "sinc", "intel", "fun", "amb", "shar", "prob", "met"}
    bad = []
    for c in feature_cols:
        if c in forbidden_exact:
            bad.append(c)
        if c.endswith("_o"):
            bad.append(c)
    if bad:
        raise ValueError(f"Leakage tripwire hit: {sorted(set(bad))}")


# -------------------------
# Nested CV splitting
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
# Hyperparameter grid
# -------------------------
@dataclass(frozen=True)
class Config:
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    max_depth: int  # -1 means no limit

    def to_params(self, seed: int) -> Dict:
        p = {
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
        return p

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
    max_depth = [8]  # keep fixed to reduce noise/variance
    feature_fraction = [0.85]
    bagging_fraction = [0.85]

    grid = []
    for lr, nl, mdl, l2, md, ff, bf in product(
        learning_rates, num_leaves, min_data_in_leaf, lambda_l2, max_depth, feature_fraction, bagging_fraction
    ):
        grid.append(Config(lr, nl, mdl, ff, bf, l2, md))
    return grid


# -------------------------
# Train one nested fold for one config
# -------------------------
def train_eval_one(
    df_all: pd.DataFrame,
    train_waves: List[int],
    val_waves: List[int],
    test_waves: List[int],
    cfg: Config,
    seed: int,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], int]:
    train_df = df_all[df_all["wave"].isin(train_waves)].copy()
    val_df = df_all[df_all["wave"].isin(val_waves)].copy()
    test_df = df_all[df_all["wave"].isin(test_waves)].copy()

    # Sort for contiguous groups
    train_df = train_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    # Features
    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _ = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    leakage_tripwire(list(X_train.columns))

    y_train = train_df["match"].to_numpy(dtype=int)
    y_val = val_df["match"].to_numpy(dtype=int)

    # Preprocess (fit only on train)
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    # Groups and weights
    group_train = compute_groups(train_df)
    group_val = compute_groups(val_df)
    group_test = compute_groups(test_df)

    w_train = compute_row_weights_equal_query(train_df)
    w_val = compute_row_weights_equal_query(val_df)

    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, weight=w_train, free_raw_data=False)
    lgb_val = lgb.Dataset(Z_val, label=y_val, group=group_val, weight=w_val, reference=lgb_train, free_raw_data=False)

    params = cfg.to_params(seed)

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    # Evaluate
    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)

    val_eval = val_df[["wave", "iid", "pid", "match"]].copy()
    val_eval["score"] = val_scores
    test_eval = test_df[["wave", "iid", "pid", "match"]].copy()
    test_eval["score"] = test_scores

    val_metrics = evaluate_ranking(val_eval, "score", "match")
    test_metrics = evaluate_ranking(test_eval, "score", "match")

    return val_metrics, test_metrics, booster.best_iteration


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    all_waves = df["wave"].unique()
    grid = make_param_grid()

    print(f"Loaded rows: {len(df):,} | waves: {len(all_waves)}")
    print(f"Grid size: {len(grid)} configs | repeats={N_REPEATS} folds={N_FOLDS}")
    print("This may take a while; results will be saved incrementally at the end.\n")

    rows = []

    for cfg_idx, cfg in enumerate(grid):
        cfg_id = cfg.id()
        print(f"[{cfg_idx+1}/{len(grid)}] Config: {cfg_id}")

        for r in range(N_REPEATS):
            seed = RANDOM_SEED_BASE + r
            rng = np.random.default_rng(seed)

            folds = make_folds(all_waves, N_FOLDS, rng)

            for fold_idx in range(N_FOLDS):
                test_waves = folds[fold_idx]
                remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]
                train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

                val_metrics, test_metrics, best_iter = train_eval_one(
                    df_all=df,
                    train_waves=train_waves,
                    val_waves=val_waves,
                    test_waves=test_waves,
                    cfg=cfg,
                    seed=seed,
                )

                for k in K_LIST:
                    rows.append({
                        "config_id": cfg_id,
                        "cfg_idx": cfg_idx,
                        "repeat": r,
                        "seed": seed,
                        "fold": fold_idx,
                        "k": k,
                        "val_ndcg": val_metrics[k]["NDCG@K"],
                        "val_map": val_metrics[k]["MAP@K"],
                        "val_recall": val_metrics[k]["Recall@K"],
                        "test_ndcg": test_metrics[k]["NDCG@K"],
                        "test_map": test_metrics[k]["MAP@K"],
                        "test_recall": test_metrics[k]["Recall@K"],
                        "best_iteration": best_iter,
                        "n_train_waves": len(train_waves),
                        "n_val_waves": len(val_waves),
                        "n_test_waves": len(test_waves),
                    })

    res = pd.DataFrame(rows)
    res.to_csv(RAW_CSV, index=False)

    # Summary aggregation per config
    # Primary selection metric: mean INNER-VAL NDCG@10
    val5 = (
        res[res["k"] == 5]
        .groupby("config_id")["val_ndcg"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "val_ndcg5_mean", "std": "val_ndcg5_std", "count": "n_runs"})
    )

    # Report TEST metrics too (but do not select by test)
    def agg_at_k(col: str, k: int) -> pd.DataFrame:
        return (
            res[res["k"] == k].groupby("config_id")[col].agg(["mean", "std"])
            .rename(columns={"mean": f"{col}_k{k}_mean", "std": f"{col}_k{k}_std"})
        )

    test_ndcg5 = agg_at_k("test_ndcg", 5)
    test_map5 = agg_at_k("test_map", 5)
    test_recall5 = agg_at_k("test_recall", 5)

    test_ndcg10 = agg_at_k("test_ndcg", 10)
    test_map10 = agg_at_k("test_map", 10)
    test_recall10 = agg_at_k("test_recall", 10)

    summary = val5.join(test_ndcg5).join(test_map5).join(test_recall5).join(
        test_ndcg10).join(test_map10).join(test_recall10)

    summary = summary.sort_values("val_ndcg5_mean", ascending=False).reset_index()

    summary.to_csv(SUMMARY_CSV, index=False)

    # Pick best config by inner-val mean NDCG@10
    best_row = summary.iloc[0]
    best_config_id = best_row["config_id"]

    lines = []
    best_row = summary.iloc[0]
    best_config_id = best_row["config_id"]

    lines = []
    lines.append("Best config selected by INNER-VAL mean NDCG@5 (nested CV, weighted-by-query)")
    lines.append("")
    lines.append(f"BEST config_id: {best_config_id}")
    lines.append(f"Runs: {int(best_row['n_runs'])} (repeat×fold)")
    lines.append("")
    lines.append("Inner-VAL (K=5):")
    lines.append(f"  NDCG@5: {best_row['val_ndcg5_mean']:.4f} ± {best_row['val_ndcg5_std']:.4f}")
    lines.append("")
    lines.append("Outer-TEST (report only):")
    lines.append(
        f"  K=5  NDCG {best_row['test_ndcg_k5_mean']:.4f} ± {best_row['test_ndcg_k5_std']:.4f} | "
        f"MAP {best_row['test_map_k5_mean']:.4f} ± {best_row['test_map_k5_std']:.4f} | "
        f"Recall {best_row['test_recall_k5_mean']:.4f} ± {best_row['test_recall_k5_std']:.4f}"
    )
    lines.append(
        f"  K=10 NDCG {best_row['test_ndcg_k10_mean']:.4f} ± {best_row['test_ndcg_k10_std']:.4f} | "
        f"MAP {best_row['test_map_k10_mean']:.4f} ± {best_row['test_map_k10_std']:.4f} | "
        f"Recall {best_row['test_recall_k10_mean']:.4f} ± {best_row['test_recall_k10_std']:.4f}"
    )

    lines.append("")
    lines.append(f"Wrote raw: {RAW_CSV}")
    lines.append(f"Wrote summary: {SUMMARY_CSV}")

    BEST_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote: {BEST_TXT}")


if __name__ == "__main__":
    main()
