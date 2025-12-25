"""
07_nested_wave_cv_ranker_eval.py

Goal
----
Evaluate LightGBM ranking models robustly using *nested cross-validation over waves*.

Why
---
With only ~21 waves, a single train/val/test split is high variance.
This script uses:
- Outer K-fold CV over waves: each fold is a TEST set once.
- Inner train/val split (over remaining waves): used for early stopping / model selection.

We repeat the K-fold procedure across multiple random shuffles ("repeats") and report mean±std.

What you get
------------
- results/nested_wave_cv_ranker_results.csv  (raw per (repeat, fold, feature_set))
- results/nested_wave_cv_ranker_summary.txt  (mean±std aggregated by feature_set)

This is the "right way" to compare feature sets under wave distribution shift.
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
OUT_CSV = OUT_DIR / "nested_wave_cv_ranker_results.csv"
OUT_SUMMARY = OUT_DIR / "nested_wave_cv_ranker_summary.txt"

RANDOM_SEED_BASE = 42

# "Right way" defaults (adjust if runtime is too high)
N_REPEATS = 5      # number of random reshuffles of waves
N_FOLDS = 5        # outer CV folds over waves
INNER_VAL_FRAC = 0.20  # fraction of non-test waves used as validation for early stopping

K_LIST = [5, 10]

FEATURE_SETS = [
    "core",
    "core+interests",
    "expanded",
    "expanded+interests",
]

# LightGBM params: a bit regularized to reduce variance across waves
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
# Ranking metrics (same as before)
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
# Utility: group sizes for LightGBM ranking
# -------------------------
def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    # Requires rows ordered so each (wave,iid) group is contiguous.
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


# -------------------------
# Features (same block logic as script 6)
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def cosine_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
    # Core
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

    # Expanded pre-event columns (only used if they exist)
    base_u_extra = ["income_u", "career_c_u", "career_u", "mn_sat_u", "tuition_u", "undergra_u", "from_u"]
    base_v_extra = ["income_v", "career_c_v", "career_v", "mn_sat_v", "tuition_v", "undergra_v", "from_v"]

    prefs_u = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
    prefs_v = ["attr1_1_v", "sinc1_1_v", "intel1_1_v", "fun1_1_v", "amb1_1_v", "shar1_1_v"]

    self_u = ["attr3_1_u", "sinc3_1_u", "intel3_1_u", "fun3_1_u", "amb3_1_u"]
    self_v = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v"]

    keep = base_u_core + base_v_core + prefs_u + prefs_v + self_u + self_v
    if "expanded" in feature_set:
        keep += base_u_extra + base_v_extra

    keep_cols = pick_existing(df, keep)

    categorical_guess = {
        "gender_u", "gender_v", "race_u", "race_v", "field_cd_u", "field_cd_v", "goal_u", "goal_v",
        "career_c_u", "career_c_v", "career_u", "career_v", "undergra_u", "undergra_v", "from_u", "from_v",
    }
    categorical_cols = [c for c in keep_cols if c in categorical_guess]
    numeric_cols = [c for c in keep_cols if c not in categorical_cols]

    X = df[keep_cols].copy()

    # Pair interaction features
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

    # Interests similarity block
    if "interests" in feature_set:
        interest = ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming",
                    "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga"]
        u_int = pick_existing(df, [f"{c}_u" for c in interest])
        v_int = pick_existing(df, [f"{c}_v" for c in interest])
        if u_int and v_int and len(u_int) == len(v_int):
            U = df[u_int].to_numpy(dtype=float)
            V = df[v_int].to_numpy(dtype=float)
            X["cos_interests_uv"] = cosine_sim_rows(U, V)
            numeric_cols.append("cos_interests_uv")

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
# Nested CV splitting
# -------------------------
def make_folds(waves: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[List[int]]:
    """
    Shuffle waves and split into n_folds approximately equal folds.
    """
    waves = np.array(sorted(waves))
    rng.shuffle(waves)
    folds = np.array_split(waves, n_folds)
    return [f.tolist() for f in folds]


def inner_train_val_split(remaining_waves: List[int], val_frac: float, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    """
    Split remaining waves into train/val for early stopping.
    """
    waves = np.array(remaining_waves, dtype=int)
    rng.shuffle(waves)
    n_val = max(1, int(len(waves) * val_frac))
    val = waves[:n_val].tolist()
    train = waves[n_val:].tolist()
    # guard: ensure train nonempty
    if len(train) == 0:
        train = waves[1:].tolist()
        val = waves[:1].tolist()
    return train, val


# -------------------------
# Train + eval one fold
# -------------------------
def train_ranker_on_fold(df_all: pd.DataFrame, train_waves: List[int], val_waves: List[int], test_waves: List[int], feature_set: str, seed: int) -> Tuple[dict, dict, int]:
    # Prepare splits
    train_df = df_all[df_all["wave"].isin(train_waves)].copy()
    val_df = df_all[df_all["wave"].isin(val_waves)].copy()
    test_df = df_all[df_all["wave"].isin(test_waves)].copy()

    # Sort for contiguous groups
    train_df = train_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    # Features per split (fit only on train)
    X_train, num_cols, cat_cols = build_features(train_df, feature_set)
    X_val, _, _ = build_features(val_df, feature_set)
    X_test, _, _ = build_features(test_df, feature_set)

    leakage_tripwire(list(X_train.columns))

    y_train = train_df["match"].to_numpy(dtype=int)
    y_val = val_df["match"].to_numpy(dtype=int)
    y_test = test_df["match"].to_numpy(dtype=int)

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

    lgb_train = lgb.Dataset(Z_train, label=y_train, group=group_train, free_raw_data=False)
    lgb_val = lgb.Dataset(Z_val, label=y_val, group=group_val, reference=lgb_train, free_raw_data=False)

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

    # Score & evaluate
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

        # Outer folds: each fold becomes TEST once
        folds = make_folds(all_waves, N_FOLDS, rng)

        for fold_idx in range(N_FOLDS):
            test_waves = folds[fold_idx]
            remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]

            # Inner split: train vs val from remaining waves
            train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

            for fs in FEATURE_SETS:
                val_metrics, test_metrics, best_iter = train_ranker_on_fold(
                    df_all=df,
                    train_waves=train_waves,
                    val_waves=val_waves,
                    test_waves=test_waves,
                    feature_set=fs,
                    seed=seed,
                )

                for k in K_LIST:
                    rows.append({
                        "repeat": r,
                        "seed": seed,
                        "fold": fold_idx,
                        "feature_set": fs,
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

    # Summarize mean±std over all (repeat, fold) runs
    summary_lines = []
    summary_lines.append("Nested Wave CV (repeated) — LightGBM Ranker Summary")
    summary_lines.append(f"N_REPEATS={N_REPEATS} | N_FOLDS={N_FOLDS} | INNER_VAL_FRAC={INNER_VAL_FRAC}")
    summary_lines.append(f"Feature sets: {FEATURE_SETS}")
    summary_lines.append("")
    summary_lines.append("Mean ± std over all runs (repeat×fold):")

    def fmt(mu: float, sd: float) -> str:
        return f"{mu:.4f} ± {sd:.4f}"

    for fs in FEATURE_SETS:
        summary_lines.append("")
        summary_lines.append(f"== {fs} ==")
        for k in K_LIST:
            sub = res[(res["feature_set"] == fs) & (res["k"] == k)]
            summary_lines.append(
                f"K={k:>2} | "
                f"TEST NDCG {fmt(sub['test_ndcg'].mean(), sub['test_ndcg'].std(ddof=1))} | "
                f"TEST MAP {fmt(sub['test_map'].mean(), sub['test_map'].std(ddof=1))} | "
                f"TEST Recall {fmt(sub['test_recall'].mean(), sub['test_recall'].std(ddof=1))}"
            )

    # Which feature set wins most often on TEST NDCG@10?
    # (per repeat-fold, pick max across feature sets)
    winners = []
    key = (res["k"] == 10)
    for (r, f), block in res[key].groupby(["repeat", "fold"]):
        # choose winner by test_ndcg
        best = block.loc[block["test_ndcg"].idxmax()]
        winners.append(best["feature_set"])
    win_counts = pd.Series(winners).value_counts()

    summary_lines.append("")
    summary_lines.append("Winner counts by TEST NDCG@10 (per repeat×fold):")
    for fs, cnt in win_counts.items():
        summary_lines.append(f"  {fs}: {int(cnt)}")

    OUT_SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nWrote raw results: {OUT_CSV}")
    print(f"Wrote summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
