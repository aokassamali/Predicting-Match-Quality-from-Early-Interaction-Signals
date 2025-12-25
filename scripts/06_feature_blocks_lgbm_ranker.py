"""
06_feature_blocks_lgbm_ranker.py

Purpose
-------
Run feature-block ablations for the LightGBM ranker under a fixed TRAIN/VAL/TEST wave split.

Why this helps
--------------
When test performance is weak, it might be because:
- the model is overfitting (too flexible), OR
- the features don't contain enough generalizable signal.

Ablations answer: "Which feature blocks actually improve generalization?"

Feature sets we run
-------------------
1) core
2) core+interests
3) expanded
4) expanded+interests

Outputs
-------
- results/feature_blocks_lgbm_ranker_metrics.txt
- saves the best-by-VAL model file under results/models/ (optional; controlled below)
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
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
MODEL_DIR = OUT_DIR / "models"

OUT_PATH = OUT_DIR / "feature_blocks_lgbm_ranker_metrics.txt"

RANDOM_SEED = 42
TEST_WAVE_FRAC = 0.20
VAL_WAVE_FRAC = 0.20
K_LIST = [5, 10]

SAVE_BEST_MODEL = True  # saves the feature-set with best VAL NDCG@10


# -------------------------
# Metrics
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
# Splitting + groups
# -------------------------
def split_waves(waves: np.ndarray, seed: int, val_frac: float, test_frac: float) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    waves = np.array(sorted(waves))
    rng.shuffle(waves)

    n = len(waves)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test = waves[:n_test].tolist()
    val = waves[n_test:n_test + n_val].tolist()
    train = waves[n_test + n_val:].tolist()

    if len(train) == 0:
        # Safety for tiny wave counts
        train = waves[n_test + 1:].tolist()
        val = waves[n_test:n_test + 1].tolist()

    return train, val, test


def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


# -------------------------
# Features
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def cosine_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity row-wise between A and B.
    Handles all-zero rows safely (returns 0 in that case).
    """
    A = np.nan_to_num(A, nan=0.0)
    B = np.nan_to_num(B, nan=0.0)
    dot = np.sum(A * B, axis=1)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    denom = na * nb
    out = np.zeros_like(dot, dtype=float)
    mask = denom > 1e-12
    out[mask] = dot[mask] / denom[mask]
    return out


def build_features(df: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    feature_set in:
      - "core"
      - "core+interests"
      - "expanded"
      - "expanded+interests"
    """
    # Core pre-event columns (from earlier scripts)
    base_u_core = [
        "age_u", "gender_u", "race_u", "field_cd_u", "goal_u",
        "imprace_u", "imprelig_u", "expnum_u", "exphappy_u",
        "date_u", "go_out_u",
    ]
    base_v_core = [
        "age_v", "gender_v", "race_v", "field_cd_v", "goal_v",
        "imprace_v", "imprelig_v", "expnum_v", "exphappy_v",
        "date_v", "go_out_v",
    ]

    # Expanded pre-event columns (if they exist in your enriched table)
    # These come from the original dataset and are plausibly available pre-event.
    base_u_extra = ["income_u", "career_c_u", "career_u", "mn_sat_u", "tuition_u", "undergra_u", "from_u"]
    base_v_extra = ["income_v", "career_c_v", "career_v", "mn_sat_v", "tuition_v", "undergra_v", "from_v"]

    prefs_u = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
    prefs_v = ["attr1_1_v", "sinc1_1_v", "intel1_1_v", "fun1_1_v", "amb1_1_v", "shar1_1_v"]

    self_u = ["attr3_1_u", "sinc3_1_u", "intel3_1_u", "fun3_1_u", "amb3_1_u"]
    self_v = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v"]

    keep = []
    keep += base_u_core + base_v_core + prefs_u + prefs_v + self_u + self_v

    if "expanded" in feature_set:
        keep += base_u_extra + base_v_extra

    keep_cols = pick_existing(df, keep)

    # Guess categorical vs numeric
    categorical_guess = {
        "gender_u", "gender_v", "race_u", "race_v", "field_cd_u", "field_cd_v", "goal_u", "goal_v",
        "career_c_u", "career_c_v", "career_u", "career_v", "undergra_u", "undergra_v", "from_u", "from_v",
    }
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

    # Preference dot products (baseline idea as features)
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

    # Interests block (cosine similarity + summary diffs)
    if "interests" in feature_set:
        interest = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
                    "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
        u_int = pick_existing(df, [f"{c}_u" for c in interest])
        v_int = pick_existing(df, [f"{c}_v" for c in interest])

        # Only add if we have a reasonable overlap
        if u_int and v_int and len(u_int) == len(v_int):
            U = df[u_int].to_numpy(dtype=float)
            V = df[v_int].to_numpy(dtype=float)
            X["cos_interests_uv"] = cosine_sim_rows(U, V)
            numeric_cols.append("cos_interests_uv")

            # Summary “distance” features
            diffs = np.abs(np.nan_to_num(U, nan=0.0) - np.nan_to_num(V, nan=0.0))
            X["mean_abs_interest_diff"] = diffs.mean(axis=1)
            X["max_abs_interest_diff"] = diffs.max(axis=1)
            numeric_cols += ["mean_abs_interest_diff", "max_abs_interest_diff"]

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


# -------------------------
# Training
# -------------------------
def train_and_eval(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, feature_set: str) -> Tuple[dict, dict, int]:
    X_train, num_cols, cat_cols = build_features(df_train, feature_set)
    X_val, _, _ = build_features(df_val, feature_set)
    X_test, _, _ = build_features(df_test, feature_set)

    leakage_tripwire(list(X_train.columns))

    y_train = df_train["match"].to_numpy(dtype=int)
    y_val = df_val["match"].to_numpy(dtype=int)
    y_test = df_test["match"].to_numpy(dtype=int)

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    group_train = compute_groups(df_train)
    group_val = compute_groups(df_val)
    group_test = compute_groups(df_test)

    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, free_raw_data=False)
    lgb_val = lgb.Dataset(Z_val, label=y_val, group=group_val, reference=lgb_train, free_raw_data=False)

    # A bit more regularization by default (helps generalization)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
    }

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=250, verbose=False)],
    )

    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)

    val_eval = df_val[["wave", "iid", "pid", "match"]].copy()
    val_eval["score"] = val_scores
    test_eval = df_test[["wave", "iid", "pid", "match"]].copy()
    test_eval["score"] = test_scores

    val_metrics = evaluate_ranking(val_eval, "score", "match")
    test_metrics = evaluate_ranking(test_eval, "score", "match")

    # Optionally save model + preprocessor for the best feature set
    return val_metrics, test_metrics, booster.best_iteration, booster, pre


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    train_waves, val_waves, test_waves = split_waves(df["wave"].unique(), RANDOM_SEED, VAL_WAVE_FRAC, TEST_WAVE_FRAC)

    train_df = df[df["wave"].isin(train_waves)].sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    val_df = df[df["wave"].isin(val_waves)].sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = df[df["wave"].isin(test_waves)].sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    feature_sets = ["core", "core+interests", "expanded", "expanded+interests"]

    lines = []
    lines.append("LightGBM Ranker — Feature Block Ablations (fixed train/val/test wave split)")
    lines.append(f"Train waves: {sorted(train_waves)}")
    lines.append(f"Val waves:   {sorted(val_waves)}")
    lines.append(f"Test waves:  {sorted(test_waves)}")
    lines.append("")
    lines.append(f"Rows | train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    lines.append("")

    best_val_ndcg10 = -1.0
    best_payload = None

    for fs in feature_sets:
        val_metrics, test_metrics, best_iter, booster, pre = train_and_eval(train_df, val_df, test_df, fs)

        lines.append(f"=== Feature set: {fs} | best_iter={best_iter} ===")
        lines.append("VAL:")
        for k, vals in val_metrics.items():
            lines.append(
                f"  K={k:>2} | Recall={vals['Recall@K']:.4f} | NDCG={vals['NDCG@K']:.4f} | MAP={vals['MAP@K']:.4f}"
            )
        lines.append("TEST:")
        for k, vals in test_metrics.items():
            lines.append(
                f"  K={k:>2} | Recall={vals['Recall@K']:.4f} | NDCG={vals['NDCG@K']:.4f} | MAP={vals['MAP@K']:.4f}"
            )
        lines.append("")

        # choose best by VAL NDCG@10
        if val_metrics[10]["NDCG@K"] > best_val_ndcg10:
            best_val_ndcg10 = val_metrics[10]["NDCG@K"]
            best_payload = (fs, booster, pre)

    if SAVE_BEST_MODEL and best_payload is not None:
        fs, booster, pre = best_payload
        model_path = MODEL_DIR / f"lgbm_ranker_best_{fs.replace('+','_')}.txt"
        pre_path = MODEL_DIR / f"lgbm_ranker_best_{fs.replace('+','_')}_preprocessor.joblib"
        booster.save_model(str(model_path))
        joblib.dump(pre, pre_path)
        lines.append(f"Saved best-by-VAL model: {model_path}")
        lines.append(f"Saved best-by-VAL preprocessor: {pre_path}")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()

