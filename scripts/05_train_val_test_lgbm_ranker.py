"""
05_train_val_test_lgbm_ranker.py

Purpose
-------
Train a LightGBM ranking model (LambdaMART-style) with a *proper* train/val/test protocol.

Why this matters
----------------
Previously, we used the test split for early stopping. That leaks information about the test set into
model selection. This script fixes that by:

- Train waves: fit the model
- Val waves: early stopping + model selection (best_iteration)
- Test waves: final reporting only

Outputs
-------
- results/lgbm_ranker_stage_a_train_val_test_metrics.txt
- results/models/lgbm_ranker_stage_a_train_val_test.txt
- results/models/lgbm_ranker_preprocessor_train_val_test.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
MODEL_DIR = OUT_DIR / "models"

METRICS_PATH = OUT_DIR / "lgbm_ranker_stage_a_train_val_test_metrics.txt"
MODEL_PATH = MODEL_DIR / "lgbm_ranker_stage_a_train_val_test.txt"
PREPROCESSOR_PATH = MODEL_DIR / "lgbm_ranker_preprocessor_train_val_test.joblib"

RANDOM_SEED = 42

# Split fractions over waves (21 total waves, so keep these moderate)
TEST_WAVE_FRAC = 0.20
VAL_WAVE_FRAC = 0.20

K_LIST = [5, 10]


# -------------------------
# Ranking metric helpers
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
    # User-level hit rate: any positive in top-k?
    return float(y_true_sorted[:k].sum() > 0)


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str = "match") -> dict:
    """
    Evaluate per (wave, iid) and average across users.
    """
    metrics = {k: {"recall": [], "ndcg": [], "map": []} for k in K_LIST}
    for (_, _), g in df.groupby(["wave", "iid"], sort=False):
        g_sorted = g.sort_values(score_col, ascending=False)
        y_sorted = g_sorted[label_col].to_numpy(dtype=int)
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


# -------------------------
# Feature building (guided + engineered)
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build leakage-safe features for Stage A:
    - pre-event demographics/intents
    - stated preferences (attr1_1 etc.)
    - self-perception (attr3_1 etc.)
    - engineered compatibility features (abs_age_diff, same_race, dot-products)

    NOTE: We intentionally exclude post-date scorecard columns like `like`, `attr`, etc.
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

    keep_cols = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)

    categorical_guess = {"gender_u", "gender_v", "race_u", "race_v", "field_cd_u", "field_cd_v", "goal_u", "goal_v"}
    categorical_cols = [c for c in keep_cols if c in categorical_guess]
    numeric_cols = [c for c in keep_cols if c not in categorical_cols]

    X = df[keep_cols].copy()

    # Engineered compatibility features
    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        numeric_cols.append("abs_age_diff")

    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        numeric_cols.append("same_race")

    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        numeric_cols.append("same_field")

    # Dot products of preferences with partner self-perception (baseline idea as features)
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
    """
    Fail fast if we accidentally included post-date variables.
    """
    forbidden_exact = {"match", "dec", "dec_o", "like", "attr", "sinc", "intel", "fun", "amb", "shar", "prob", "met"}
    bad = []
    for c in feature_cols:
        if c in forbidden_exact:
            bad.append(c)
        if c.endswith("_o"):  # partner scorecard columns often end with _o
            bad.append(c)
    if bad:
        raise ValueError(f"Leakage tripwire hit: {sorted(set(bad))}")


def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    """
    LightGBM ranking requires group sizes for each query (here query = (wave, iid)),
    and rows must be ordered so each group is contiguous.
    """
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def split_waves(waves: np.ndarray, seed: int, val_frac: float, test_frac: float) -> Tuple[List[int], List[int], List[int]]:
    """
    Split wave IDs into train/val/test lists deterministically.
    """
    rng = np.random.default_rng(seed)
    waves = np.array(sorted(waves))
    rng.shuffle(waves)

    n = len(waves)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test = waves[:n_test].tolist()
    val = waves[n_test:n_test + n_val].tolist()
    train = waves[n_test + n_val:].tolist()

    # Guard: ensure train isn't empty
    if len(train) == 0:
        # If waves are too few, reduce val size
        train = waves[n_test + 1:].tolist()
        val = waves[n_test:n_test + 1].tolist()

    return train, val, test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load + basic clean
    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    # Split by wave (generalize to unseen events)
    all_waves = df["wave"].unique()
    train_waves, val_waves, test_waves = split_waves(all_waves, RANDOM_SEED, VAL_WAVE_FRAC, TEST_WAVE_FRAC)

    train_df = df[df["wave"].isin(train_waves)].copy()
    val_df = df[df["wave"].isin(val_waves)].copy()
    test_df = df[df["wave"].isin(test_waves)].copy()

    # Sort each split so (wave, iid) groups are contiguous
    train_df = train_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    # Build features per split (important: don't rely on global indices)
    X_train, numeric_cols, categorical_cols = build_features(train_df)
    X_val, _, _ = build_features(val_df)
    X_test, _, _ = build_features(test_df)

    leakage_tripwire(list(X_train.columns))

    y_train = train_df["match"].to_numpy(dtype=int)
    y_val = val_df["match"].to_numpy(dtype=int)
    y_test = test_df["match"].to_numpy(dtype=int)

    # Preprocess:
    # Trees don't need scaling. We just impute missing values and one-hot encode categoricals.
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        [("num", num_tf, numeric_cols), ("cat", cat_tf, categorical_cols)],
        remainder="drop",
    )

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    # Group sizes for ranking
    group_train = compute_groups(train_df)
    group_val = compute_groups(val_df)
    group_test = compute_groups(test_df)

    # LightGBM datasets
    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, free_raw_data=False)
    lgb_val = lgb.Dataset(Z_val, label=y_val, group=group_val, reference=lgb_train, free_raw_data=False)

    # LambdaMART-style params (you'll tune these later)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }

    # Train using VAL for early stopping
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=True)],
    )

    # Score VAL and TEST using the selected best_iteration
    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)

    val_eval = val_df[["wave", "iid", "pid", "match"]].copy()
    val_eval["score"] = val_scores

    test_eval = test_df[["wave", "iid", "pid", "match"]].copy()
    test_eval["score"] = test_scores

    val_metrics = evaluate_ranking(val_eval, score_col="score", label_col="match")
    test_metrics = evaluate_ranking(test_eval, score_col="score", label_col="match")

    # Save model + preprocessor
    booster.save_model(str(MODEL_PATH))
    joblib.dump(pre, PREPROCESSOR_PATH)

    lines = []
    lines.append("Stage A LightGBM Ranker (LambdaMART-style) — TRAIN/VAL/TEST wave split")
    lines.append(f"Train waves: {sorted(train_waves)}")
    lines.append(f"Val waves:   {sorted(val_waves)}")
    lines.append(f"Test waves:  {sorted(test_waves)}")
    lines.append("")
    lines.append(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")
    lines.append(f"Match rate train: {y_train.mean():.4f} | val: {y_val.mean():.4f} | test: {y_test.mean():.4f}")
    lines.append(f"Best iteration (chosen on val): {booster.best_iteration}")
    lines.append("")
    lines.append("VAL ranking metrics (mutual match):")
    for k, vals in val_metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )
    lines.append("")
    lines.append("TEST ranking metrics (mutual match):")
    for k, vals in test_metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )
    lines.append("")
    lines.append(f"Saved model: {MODEL_PATH}")
    lines.append(f"Saved preprocessor: {PREPROCESSOR_PATH}")

    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {METRICS_PATH}")


if __name__ == "__main__":
    main()
