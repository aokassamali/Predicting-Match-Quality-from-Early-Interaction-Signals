"""
09_xgb_ranker_nested_cv_topk.py

Purpose
-------
Compare XGBoost ranking objectives under the same *nested wave CV* protocol,
optimizing for top-of-list quality (NDCG@5 / Recall@5).

We evaluate:
- objective = "rank:pairwise"
- objective = "rank:ndcg"

We also do a small param grid (manageable) and select the best config by
INNER-VAL mean NDCG@5 (not test!).

Outputs
-------
- results/xgb_nested_cv_raw.csv
- results/xgb_nested_cv_summary.csv
- results/xgb_nested_cv_best.txt

Notes
-----
- This uses per-row weights = 1/group_size to align training with per-query evaluation.
- XGBoost typically runs on CPU. If you have GPU XGBoost installed, you can set TREE_METHOD="gpu_hist".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# If missing: pip install xgboost
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "xgb_nested_cv_raw.csv"
SUMMARY_CSV = OUT_DIR / "xgb_nested_cv_summary.csv"
BEST_TXT = OUT_DIR / "xgb_nested_cv_best.txt"

RANDOM_SEED_BASE = 42

N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20

K_LIST = [5, 10]
TOPK_SELECT = 5  # IMPORTANT: we select by inner-VAL NDCG@5

# If your XGBoost build supports GPU, set to "gpu_hist"
TREE_METHOD = "hist"


# -------------------------
# Metrics (same as before)
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
# Groups + weights
# -------------------------
def compute_group_sizes(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    sizes_per_row = df_sorted.groupby(["wave", "iid"], sort=False)["pid"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(sizes_per_row, 1.0)


# -------------------------
# Core features (same as your core)
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
    cat_cols = [c for c in keep_cols if c in categorical_guess]
    num_cols = [c for c in keep_cols if c not in cat_cols]

    X = df[keep_cols].copy()

    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        num_cols.append("abs_age_diff")
    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        num_cols.append("same_race")
    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        num_cols.append("same_field")

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
        num_cols.append("pref_dot_self_v")

    if v_pref and u_self and len(v_pref) == len(u_self):
        pv = np.nan_to_num(df[v_pref].to_numpy(dtype=float), nan=0.0)
        su = np.nan_to_num(df[u_self].to_numpy(dtype=float), nan=0.0)
        X["pref_v_dot_self_u"] = np.sum(pv * su, axis=1)
        num_cols.append("pref_v_dot_self_u")

    # Dedup
    def dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return X, dedup(num_cols), dedup(cat_cols)


# -------------------------
# Nested split helpers
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
# Param grid (small but meaningful)
# -------------------------
@dataclass(frozen=True)
class XGBConfig:
    objective: str  # "rank:pairwise" or "rank:ndcg"
    eta: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float

    def id(self) -> str:
        return (
            f"obj={self.objective}_eta={self.eta}_md={self.max_depth}_mcw={self.min_child_weight}_"
            f"sub={self.subsample}_col={self.colsample_bytree}_l2={self.reg_lambda}_l1={self.reg_alpha}"
        )


def make_grid() -> List[XGBConfig]:
    objectives = ["rank:pairwise", "rank:ndcg"]

    # keep it small but meaningful
    eta = [0.05]              # stable, let early stopping pick trees
    max_depth = [4, 6]        # capacity
    min_child_weight = [1.0, 5.0]  # regularization
    subsample = [0.85]        # variance reduction
    colsample_bytree = [0.85] # variance reduction
    reg_lambda = [1.0, 10.0]  # L2
    reg_alpha = [0.0]         # keep off initially

    grid = []
    for obj, e, md, mcw, sub, col, l2, l1 in product(
        objectives, eta, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, reg_alpha
    ):
        grid.append(XGBConfig(obj, e, md, mcw, sub, col, l2, l1))
    return grid


# -------------------------
# Train/eval one fold
# -------------------------
def train_eval_one(
    df_all: pd.DataFrame,
    train_waves: List[int],
    val_waves: List[int],
    test_waves: List[int],
    cfg: XGBConfig,
    seed: int,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], int]:
    train_df = df_all[df_all["wave"].isin(train_waves)].copy()
    val_df = df_all[df_all["wave"].isin(val_waves)].copy()
    test_df = df_all[df_all["wave"].isin(test_waves)].copy()

    # sort -> contiguous groups
    train_df = train_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _ = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    y_train = train_df["match"].to_numpy(dtype=float)
    y_val = val_df["match"].to_numpy(dtype=float)

    # preprocess
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    # group sizes + row weights
    g_train = compute_group_sizes(train_df)
    g_val = compute_group_sizes(val_df)
    g_test = compute_group_sizes(test_df)

    gw_train = np.ones(len(g_train), dtype=float)
    gw_val = np.ones(len(g_val), dtype=float)

    dtrain = xgb.DMatrix(Z_train, label=y_train)
    dval = xgb.DMatrix(Z_val, label=y_val)
    dtest = xgb.DMatrix(Z_test)

    
    dtest = xgb.DMatrix(Z_test)

    dtrain.set_group(g_train)
    dval.set_group(g_val)
    dtest.set_group(g_test)

    dtrain.set_weight(gw_train)
    dval.set_weight(gw_val)

    params = {
        "objective": cfg.objective,
        "eta": cfg.eta,
        "max_depth": cfg.max_depth,
        "min_child_weight": cfg.min_child_weight,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "reg_alpha": cfg.reg_alpha,
        "tree_method": TREE_METHOD,
        "eval_metric": "ndcg@5",  # align early stopping with top-of-list
        "seed": seed,
        "verbosity": 0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    val_scores = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    test_scores = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))

    val_eval = val_df[["wave", "iid", "pid", "match"]].copy()
    val_eval["score"] = val_scores
    test_eval = test_df[["wave", "iid", "pid", "match"]].copy()
    test_eval["score"] = test_scores

    val_metrics = evaluate_ranking(val_eval, "score", "match")
    test_metrics = evaluate_ranking(test_eval, "score", "match")

    return val_metrics, test_metrics, int(booster.best_iteration)


def main() -> None:
    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    all_waves = df["wave"].unique()
    grid = make_grid()

    rows = []

    for cfg_idx, cfg in enumerate(grid):
        cfg_id = cfg.id()
        print(f"[{cfg_idx+1}/{len(grid)}] {cfg_id}")

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
                        "fold": fold_idx,
                        "k": k,
                        "val_ndcg": val_metrics[k]["NDCG@K"],
                        "val_recall": val_metrics[k]["Recall@K"],
                        "val_map": val_metrics[k]["MAP@K"],
                        "test_ndcg": test_metrics[k]["NDCG@K"],
                        "test_recall": test_metrics[k]["Recall@K"],
                        "test_map": test_metrics[k]["MAP@K"],
                        "best_iteration": best_iter,
                    })

    res = pd.DataFrame(rows)
    res.to_csv(RAW_CSV, index=False)

    # Aggregate per config; select by INNER-VAL NDCG@5 (k=5)
    val_sel = (
        res[res["k"] == TOPK_SELECT]
        .groupby("config_id")["val_ndcg"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "val_ndcg5_mean", "std": "val_ndcg5_std", "count": "n_runs"})
    )

    test5 = (
        res[res["k"] == 5].groupby("config_id")[["test_ndcg", "test_recall", "test_map"]]
        .agg(["mean", "std"])
    )
    test10 = (
        res[res["k"] == 10].groupby("config_id")[["test_ndcg", "test_recall", "test_map"]]
        .agg(["mean", "std"])
    )

    # flatten columns
    def flatten(df_: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df_.columns = [f"{prefix}_{a}_{b}" for (a, b) in df_.columns]
        return df_

    summary = val_sel.join(flatten(test5, "k5")).join(flatten(test10, "k10"))
    summary = summary.sort_values("val_ndcg5_mean", ascending=False).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)

    best = summary.iloc[0]
    lines = []
    lines.append("Best XGBoost ranker config selected by INNER-VAL mean NDCG@5 (nested CV)")
    lines.append(f"TOPK_SELECT={TOPK_SELECT} | repeats={N_REPEATS} folds={N_FOLDS}")
    lines.append("")
    lines.append(f"BEST config_id: {best['config_id']}")
    lines.append(f"Inner-VAL NDCG@5: {best['val_ndcg5_mean']:.4f} ± {best['val_ndcg5_std']:.4f} (n={int(best['n_runs'])})")
    lines.append("")
    lines.append("Outer-TEST (report only):")
    lines.append(f"  K=5  NDCG {best['k5_test_ndcg_mean']:.4f} ± {best['k5_test_ndcg_std']:.4f} | "
                 f"Recall {best['k5_test_recall_mean']:.4f} ± {best['k5_test_recall_std']:.4f} | "
                 f"MAP {best['k5_test_map_mean']:.4f} ± {best['k5_test_map_std']:.4f}")
    lines.append(f"  K=10 NDCG {best['k10_test_ndcg_mean']:.4f} ± {best['k10_test_ndcg_std']:.4f} | "
                 f"Recall {best['k10_test_recall_mean']:.4f} ± {best['k10_test_recall_std']:.4f} | "
                 f"MAP {best['k10_test_map_mean']:.4f} ± {best['k10_test_map_std']:.4f}")
    lines.append("")
    lines.append(f"Wrote raw: {RAW_CSV}")
    lines.append(f"Wrote summary: {SUMMARY_CSV}")

    BEST_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {BEST_TXT}")


if __name__ == "__main__":
    main()
