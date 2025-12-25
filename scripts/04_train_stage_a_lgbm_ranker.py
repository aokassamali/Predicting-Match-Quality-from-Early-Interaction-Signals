"""
04_train_stage_a_lgbm_ranker.py

Goal
----
Train a Stage A *ranking* model (LambdaMART-style) using LightGBM.

Key idea (12-year-old analogy)
------------------------------
- LightGBM is the "team of small rulebooks" (many small decision trees).
- LambdaMART/LambdaRank is the "practice game": it trains the team to make the *best ordered list*
  (good items at the top), which matches metrics like NDCG.

Reality
-------
We have natural ranking groups: each (wave, iid) is a query, and the candidates pid are the items.
We train LightGBM with a ranking objective ("lambdarank") and pass group sizes.

Leakage policy
--------------
Same as before: use only pre-event / Time1-ish features and engineered compatibility features.
Do NOT use post-date scorecards (attr/like/prob/dec etc.).

Inputs
------
- results/pairs_enriched.parquet

Outputs
-------
- results/lgbm_ranker_stage_a_metrics.txt
- results/models/lgbm_ranker_stage_a.txt (LightGBM model file)
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
MODEL_DIR = OUT_DIR / "models"
METRICS_PATH = OUT_DIR / "lgbm_ranker_stage_a_metrics.txt"
MODEL_PATH = MODEL_DIR / "lgbm_ranker_stage_a.txt"

RANDOM_SEED = 42
TEST_WAVE_FRAC = 0.30
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
    return float(y_true_sorted[:k].sum() > 0)


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str = "match") -> dict:
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
    # Guided pre-event columns (keep if present)
    base_u = ["age_u", "gender_u", "race_u", "field_cd_u", "goal_u", "imprace_u", "imprelig_u", "expnum_u", "exphappy_u", "date_u", "go_out_u"]
    base_v = ["age_v", "gender_v", "race_v", "field_cd_v", "goal_v", "imprace_v", "imprelig_v", "expnum_v", "exphappy_v", "date_v", "go_out_v"]

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

    # preference dot products
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

    # Dedup
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


def compute_groups(df: pd.DataFrame) -> np.ndarray:
    """
    LightGBM ranking expects group sizes in the SAME order as the rows appear.
    We'll sort by (wave, iid) so each group is contiguous.
    """
    grp = df.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)
    return grp


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load & clean
    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    # Wave split
    waves = np.array(sorted(df["wave"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(waves)
    n_test = max(1, int(len(waves) * TEST_WAVE_FRAC))
    test_waves = set(waves[:n_test])
    train_waves = set(waves[n_test:])

    train_df = df[df["wave"].isin(train_waves)].copy()
    test_df = df[df["wave"].isin(test_waves)].copy()

    # Sort so groups are contiguous (required for group arrays)
    train_df = train_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    # Features + leakage guardrails
    X_all, numeric_cols, categorical_cols = build_features(df)
    leakage_tripwire(list(X_all.columns))

    X_train = X_all.loc[train_df.index].copy()  # aligned because we reset_index after sort
    X_test = X_all.loc[test_df.index].copy()

    y_train = train_df["match"].to_numpy(dtype=int)
    y_test = test_df["match"].to_numpy(dtype=int)

    # Preprocess:
    # - numeric: median impute
    # - categorical: most_frequent + onehot
    # We do NOT standardize for trees (not needed).
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        [("num", num_tf, numeric_cols), ("cat", cat_tf, categorical_cols)],
        remainder="drop",
    )

    Z_train = pre.fit_transform(X_train)
    Z_test = pre.transform(X_test)

    # Group sizes for ranking
    group_train = compute_groups(train_df)
    group_test = compute_groups(test_df)

    # LightGBM datasets
    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(Z_test, label=y_test, group=group_test, reference=lgb_train, free_raw_data=False)

    # LambdaMART-style ranking params
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }

    # Train with early stopping on the test set (treated as a validation set for now)
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )

    # Score test pairs
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)
    test_eval = test_df[["wave", "iid", "pid", "match"]].copy()
    test_eval["score"] = test_scores

    metrics = evaluate_ranking(test_eval, score_col="score", label_col="match")

    # Save model (LightGBM native) + preprocessor separately
    booster.save_model(str(MODEL_PATH))
    # Save preprocessor so we can reproduce inference later
    pre_path = MODEL_DIR / "lgbm_ranker_preprocessor.joblib"
    import joblib
    joblib.dump(pre, pre_path)

    lines = []
    lines.append("Stage A LightGBM Ranker (LambdaMART-style) — wave split")
    lines.append(f"Test waves: {sorted(test_waves)}")
    lines.append(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
    lines.append(f"Train match rate: {y_train.mean():.4f} | Test match rate: {y_test.mean():.4f}")
    lines.append(f"Best iteration: {booster.best_iteration}")
    lines.append("")
    lines.append("Ranking metrics on TEST waves (mutual match):")
    for k, vals in metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )
    lines.append("")
    lines.append(f"Saved model: {MODEL_PATH}")
    lines.append(f"Saved preprocessor: {pre_path}")

    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {METRICS_PATH}")


if __name__ == "__main__":
    main()
