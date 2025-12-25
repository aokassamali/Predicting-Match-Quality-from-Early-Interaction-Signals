"""
08c_rerun_top_configs_lgbm_k5.py

Goal
----
Re-evaluate only the top configs (from 08b summary) with more repeats to reduce selection noise.

How to use
----------
1) Paste the top config_id strings into TOP_CONFIG_IDS below.
2) Run:
   python scripts/08c_rerun_top_configs_lgbm_k5.py

Outputs
-------
- results/08c_rerun_raw.csv
- results/08c_rerun_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "08c_rerun_raw.csv"
SUMMARY_CSV = OUT_DIR / "08c_rerun_summary.csv"

RANDOM_SEED_BASE = 123
N_REPEATS = 20
N_FOLDS = 5
INNER_VAL_FRAC = 0.20
K_LIST = [5, 10]

# Paste top 5 from 08b summary here (order doesn't matter)
TOP_CONFIG_IDS = [
    "lr=0.03_leaves=15_minleaf=80_ff=0.85_bf=0.85_l2=5.0_md=8",
    "lr=0.03_leaves=31_minleaf=40_ff=0.85_bf=0.85_l2=20.0_md=8",
    "lr=0.03_leaves=15_minleaf=80_ff=0.85_bf=0.85_l2=20.0_md=8",
    "lr=0.03_leaves=31_minleaf=150_ff=0.85_bf=0.85_l2=20.0_md=8",
    "lr=0.03_leaves=15_minleaf=150_ff=0.85_bf=0.85_l2=20.0_md=8"
]

# Base params shared by all configs
BASE_LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5],   # align with top-of-list
    "verbosity": -1,
    "bagging_freq": 1,
}


@dataclass(frozen=True)
class Config:
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    lambda_l2: float
    max_depth: int

    @staticmethod
    def from_id(config_id: str) -> "Config":
        # Parses strings like:
        # lr=0.03_leaves=15_minleaf=80_ff=0.85_bf=0.75_l2=5.0_md=8
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in config_id.split("_")}
        return Config(
            learning_rate=float(parts["lr"]),
            num_leaves=int(parts["leaves"]),
            min_data_in_leaf=int(parts["minleaf"]),
            feature_fraction=float(parts["ff"]),
            bagging_fraction=float(parts["bf"]),
            lambda_l2=float(parts["l2"]),
            max_depth=int(parts["md"]),
        )

    def to_params(self, seed: int) -> Dict:
        p = dict(BASE_LGB_PARAMS)
        p.update({
            "seed": seed,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
        })
        return p


# ---------- metrics ----------
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
    precisions, hits = [], 0
    for i, rel in enumerate(y, start=1):
        if rel == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    return float(y_true_sorted[:k].sum() > 0)


def evaluate_ranking(df: pd.DataFrame, score_col: str) -> Dict[int, Dict[str, float]]:
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
            "recall": float(np.mean(metrics[k]["recall"])),
            "ndcg": float(np.mean(metrics[k]["ndcg"])),
            "map": float(np.mean(metrics[k]["map"])),
        }
    return out


def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    sizes_per_row = df_sorted.groupby(["wave", "iid"], sort=False)["pid"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(sizes_per_row, 1.0)


# ---------- features (core) ----------
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

    categorical_guess = {"gender_u","gender_v","race_u","race_v","field_cd_u","field_cd_v","goal_u","goal_v"}
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

    # dot products
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

    def dedup(xs: List[str]) -> List[str]:
        out, seen = [], set()
        for x in xs:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return X, dedup(num_cols), dedup(cat_cols)


# ---------- splits ----------
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


def train_eval_one(df_all: pd.DataFrame, train_waves: List[int], val_waves: List[int], test_waves: List[int], cfg: Config, seed: int):
    train_df = df_all[df_all["wave"].isin(train_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)
    val_df = df_all[df_all["wave"].isin(val_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)
    test_df = df_all[df_all["wave"].isin(test_waves)].copy().sort_values(["wave","iid","pid"]).reset_index(drop=True)

    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _ = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    y_train = train_df["match"].to_numpy(int)
    y_val = val_df["match"].to_numpy(int)

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

    w_train = compute_row_weights_equal_query(train_df)
    w_val = compute_row_weights_equal_query(val_df)

    dtrain = lgb.Dataset(Z_train, label=y_train, group=group_train, weight=w_train, free_raw_data=False)
    dval = lgb.Dataset(Z_val, label=y_val, group=group_val, weight=w_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        cfg.to_params(seed),
        dtrain,
        num_boost_round=8000,
        valid_sets=[dtrain, dval],
        valid_names=["train","val"],
        callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)],
    )

    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)

    val_eval = val_df[["wave","iid","pid","match"]].copy()
    val_eval["score"] = val_scores
    test_eval = test_df[["wave","iid","pid","match"]].copy()
    test_eval["score"] = test_scores

    return evaluate_ranking(val_eval, "score"), evaluate_ranking(test_eval, "score"), booster.best_iteration


def main():
    if not TOP_CONFIG_IDS:
        raise ValueError("Paste top config IDs into TOP_CONFIG_IDS first.")

    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave","iid","pid","match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)
    all_waves = df["wave"].unique()

    configs = [(cid, Config.from_id(cid)) for cid in TOP_CONFIG_IDS]
    rows = []

    for cid, cfg in configs:
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
                        "config_id": cid,
                        "repeat": r,
                        "fold": fold_idx,
                        "k": k,
                        "val_ndcg": val_m[k]["ndcg"],
                        "val_map": val_m[k]["map"],
                        "val_recall": val_m[k]["recall"],
                        "test_ndcg": test_m[k]["ndcg"],
                        "test_map": test_m[k]["map"],
                        "test_recall": test_m[k]["recall"],
                        "best_iteration": best_iter,
                    })

    res = pd.DataFrame(rows)
    res.to_csv(RAW_CSV, index=False)

    # Summaries
    def summarize(k: int) -> pd.DataFrame:
        sub = res[res["k"] == k]
        return sub.groupby("config_id")[["test_ndcg","test_map","test_recall"]].agg(["mean","std"])

    sum5 = summarize(5)
    sum10 = summarize(10)

    # Flatten
    def flat(df_: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df_.columns = [f"{prefix}_{a}_{b}" for (a, b) in df_.columns]
        return df_

    summary = flat(sum5, "k5").join(flat(sum10, "k10")).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nTop-config rerun complete. See:")
    print(f"  {RAW_CSV}")
    print(f"  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
