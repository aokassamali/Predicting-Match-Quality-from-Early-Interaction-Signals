"""
08a_weighted_query_lgbm_nested_cv.py

Goal
----
Rerun nested wave cross-validation for the *core* feature set, but add per-row weights so each
(query = wave,iid) contributes equally to the training objective.

Why (intuition)
---------------
Evaluation averages metrics per query/user. But training loss is row-based.
Users with more candidates contribute more rows -> they dominate training.
We fix by weighting each row as 1 / group_size, so each query sums to weight ~1.

Outputs
-------
- results/weighted_core_nested_cv_results.csv
- results/weighted_core_nested_cv_summary.txt
"""

from __future__ import annotations

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
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
OUT_CSV = OUT_DIR / "weighted_core_nested_cv_results.csv"
OUT_SUMMARY = OUT_DIR / "weighted_core_nested_cv_summary.txt"

RANDOM_SEED_BASE = 42

# Use the same structure as Script 7 for comparability
N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20

K_LIST = [5, 10]

FEATURE_SET = "core"  # this script focuses on core only

# More regularized defaults to stabilize across waves
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
}


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
    """
    Return group sizes in the current row order.
    Assumes df_sorted is sorted so each (wave,iid) group is contiguous.
    """
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    """
    Per-row weights so each query contributes equally:

      weight(row in group g) = 1 / |g|

    This makes sum(weights in group) = 1 for each group.
    """
    group_sizes = df_sorted.groupby(["wave", "iid"], sort=False).size()
    # Broadcast to rows via transform
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
# Train + eval one fold with weights
# -------------------------
def train_ranker_on_fold_weighted(
    df_all: pd.DataFrame,
    train_waves: List[int],
    val_waves: List[int],
    test_waves: List[int],
    seed: int,
) -> Tuple[dict, dict, int]:
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
    y_test = test_df["match"].to_numpy(dtype=int)

    # Preprocess (impute + onehot)
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    group_train = compute_groups(train_df)
    group_val = compute_groups(val_df)
    group_test = compute_groups(test_df)

    # NEW: per-row weights so each query sums to 1
    w_train = compute_row_weights_equal_query(train_df)
    w_val = compute_row_weights_equal_query(val_df)
    w_test = compute_row_weights_equal_query(test_df)  # not strictly needed, but kept symmetrical

    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, weight=w_train, free_raw_data=False)
    lgb_val = lgb.Dataset(Z_val, label=y_val, group=group_val, weight=w_val, reference=lgb_train, free_raw_data=False)

    params = dict(LGB_PARAMS)
    params["seed"] = seed

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=8000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
    )

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

    rows = []

    for r in range(N_REPEATS):
        seed = RANDOM_SEED_BASE + r
        rng = np.random.default_rng(seed)

        folds = make_folds(all_waves, N_FOLDS, rng)

        for fold_idx in range(N_FOLDS):
            test_waves = folds[fold_idx]
            remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]
            train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

            val_metrics, test_metrics, best_iter = train_ranker_on_fold_weighted(
                df_all=df,
                train_waves=train_waves,
                val_waves=val_waves,
                test_waves=test_waves,
                seed=seed,
            )

            for k in K_LIST:
                rows.append({
                    "repeat": r,
                    "seed": seed,
                    "fold": fold_idx,
                    "feature_set": FEATURE_SET,
                    "k": k,
                    "val_recall": val_metrics[k]["Recall@K"],
                    "val_ndcg": val_metrics[k]["NDCG@K"],
                    "val_map": val_metrics[k]["MAP@K"],
                    "test_recall": test_metrics[k]["Recall@K"],
                    "test_ndcg": test_metrics[k]["NDCG@K"],
                    "test_map": test_metrics[k]["MAP@K"],
                    "best_iteration": best_iter,
                    "n_train_waves": len(train_waves),
                    "n_val_waves": len(val_waves),
                    "n_test_waves": len(test_waves),
                })

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    # Summary mean±std across repeat×fold
    def fmt(mu: float, sd: float) -> str:
        return f"{mu:.4f} ± {sd:.4f}"

    lines = []
    lines.append("Weighted-by-query Nested Wave CV — LightGBM Ranker (core only)")
    lines.append(f"N_REPEATS={N_REPEATS} | N_FOLDS={N_FOLDS} | INNER_VAL_FRAC={INNER_VAL_FRAC}")
    lines.append("Row weights: weight(row) = 1 / group_size, so each (wave,iid) sums to 1.")
    lines.append("")

    for k in K_LIST:
        sub = res[res["k"] == k]
        lines.append(
            f"K={k:>2} | TEST NDCG {fmt(sub['test_ndcg'].mean(), sub['test_ndcg'].std(ddof=1))} | "
            f"TEST MAP {fmt(sub['test_map'].mean(), sub['test_map'].std(ddof=1))} | "
            f"TEST Recall {fmt(sub['test_recall'].mean(), sub['test_recall'].std(ddof=1))}"
        )

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote raw results: {OUT_CSV}")
    print(f"Wrote summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
