# scripts/stage_b_lib.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------- Metrics ----------------
def dcg_at_k(rels: np.ndarray, k: int) -> float:
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) * discounts))

def ndcg_at_k(y_true_sorted: np.ndarray, k: int) -> float:
    dcg = dcg_at_k(y_true_sorted, k)
    ideal = np.sort(y_true_sorted)[::-1]
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate(df: pd.DataFrame, score_col: str, graded_col: str = "quality"):
    df = df.sort_values(["wave", "iid", score_col], ascending=[True, True, False]).reset_index(drop=True)

    ks = [5, 10]
    thresholds = [7.0, 8.0]

    ndcg = {k: [] for k in ks}
    recall = {t: {k: [] for k in ks} for t in thresholds}

    for (_, _), g in df.groupby(["wave", "iid"], sort=False):
        y = np.nan_to_num(g[graded_col].to_numpy(float), nan=0.0)
        y = np.clip(np.rint(y), 0, 10)  # keep as float thresholds are float
        y_sorted = y  # already sorted by score_col

        # NDCG@k
        for k in ks:
            rels = y_sorted[:k].astype(int)
            ideal = np.sort(y_sorted)[::-1][:k].astype(int)
            # simple ndcg
            def dcg(rels_):
                rels_ = rels_.astype(float)
                discounts = 1.0 / np.log2(np.arange(2, len(rels_) + 2))
                return float(np.sum((2.0 ** rels_ - 1.0) * discounts))
            dcg_k = dcg(rels)
            idcg_k = dcg(ideal)
            ndcg[k].append(0.0 if idcg_k == 0 else dcg_k / idcg_k)

        # Recall@k for “has >=1 relevant in top-k”, where relevant = (quality >= threshold)
        for t in thresholds:
            is_rel = (y_sorted >= t).astype(int)
            total_rel = int(is_rel.sum())
            for k in ks:
                hit = int(is_rel[:k].sum() > 0)
                # two common definitions:
                # (1) hit-rate style: fraction of queries with >=1 relevant in top-k  (this matches your earlier prints)
                recall[t][k].append(hit)

                # If you instead want true recall = (#rel in top-k) / (#rel total), use:
                # recall[t][k].append(0.0 if total_rel == 0 else float(is_rel[:k].sum() / total_rel))

    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k]))

    out["recall"] = {t: {k: float(np.mean(recall[t][k])) for k in ks} for t in thresholds}
    return out


# ---------------- Grouping / weights ----------------
def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)

def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    sizes = df_sorted.groupby(["wave", "iid"], sort=False).size()
    w = df_sorted.set_index(["wave", "iid"]).index.map(lambda t: 1.0 / sizes.loc[t])
    return np.asarray(w, dtype=float)


# ---------------- Feature building (from Script 12) ----------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def add_compat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age_u" in df.columns and "age_v" in df.columns:
        df["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
    if "race_u" in df.columns and "race_v" in df.columns:
        df["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        df["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
    return df

def build_core_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df2 = add_compat_features(df)

    base_u = ["age_u","gender_u","race_u","field_cd_u","goal_u","imprace_u","imprelig_u","expnum_u","exphappy_u","date_u","go_out_u"]
    base_v = ["age_v","gender_v","race_v","field_cd_v","goal_v","imprace_v","imprelig_v","expnum_v","exphappy_v","date_v","go_out_v"]
    prefs_u = ["attr1_1_u","sinc1_1_u","intel1_1_u","fun1_1_u","amb1_1_u","shar1_1_u"]
    prefs_v = ["attr1_1_v","sinc1_1_v","intel1_1_v","fun1_1_v","amb1_1_v","shar1_1_v"]
    self_u  = ["attr3_1_u","sinc3_1_u","intel3_1_u","fun3_1_u","amb3_1_u"]
    self_v  = ["attr3_1_v","sinc3_1_v","intel3_1_v","fun3_1_v","amb3_1_v"]

    keep = pick_existing(df2, base_u + base_v + prefs_u + prefs_v + self_u + self_v)
    for c in ["abs_age_diff","same_race","same_field"]:
        if c in df2.columns and c not in keep:
            keep.append(c)

    cat_guess = {"gender_u","gender_v","race_u","race_v","field_cd_u","field_cd_v","goal_u","goal_v"}
    cat_cols = [c for c in keep if c in cat_guess]
    num_cols = [c for c in keep if c not in cat_cols]

    X = df2[keep].copy()

    # preference alignment dot-products (if inputs exist)
    trait_map = [
        ("attr","attr1_1_u","attr3_1_v","attr1_1_v","attr3_1_u"),
        ("sinc","sinc1_1_u","sinc3_1_v","sinc1_1_v","sinc3_1_u"),
        ("intel","intel1_1_u","intel3_1_v","intel1_1_v","intel3_1_u"),
        ("fun","fun1_1_u","fun3_1_v","fun1_1_v","fun3_1_u"),
        ("amb","amb1_1_u","amb3_1_v","amb1_1_v","amb3_1_u"),
    ]

    u_pref, v_self, v_pref, u_self = [], [], [], []
    for _, pu, sv, pv, su in trait_map:
        if pu in df2.columns and sv in df2.columns:
            u_pref.append(pu); v_self.append(sv)
        if pv in df2.columns and su in df2.columns:
            v_pref.append(pv); u_self.append(su)

    if u_pref and v_self:
        pu = np.nan_to_num(df2[u_pref].to_numpy(float), nan=0.0)
        sv = np.nan_to_num(df2[v_self].to_numpy(float), nan=0.0)
        X["pref_dot_self_v"] = np.sum(pu * sv, axis=1)
        num_cols.append("pref_dot_self_v")

    if v_pref and u_self:
        pv = np.nan_to_num(df2[v_pref].to_numpy(float), nan=0.0)
        su = np.nan_to_num(df2[u_self].to_numpy(float), nan=0.0)
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


# ---------------- Splits ----------------
def make_folds(waves: np.ndarray, n_folds: int, rng: np.random.Generator) -> List[List[int]]:
    waves = np.array(sorted(np.unique(waves)))
    rng.shuffle(waves)
    return [waves[i::n_folds].tolist() for i in range(n_folds)]

def inner_train_val_split(remaining_waves: List[int], val_frac: float, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    waves = remaining_waves.copy()
    rng.shuffle(waves)
    n_val = max(1, int(round(val_frac * len(waves))))
    val = waves[:n_val]
    train = waves[n_val:]
    if len(train) == 0:
        train = waves[1:]
        val = waves[:1]
    return train, val


# ---------------- Model fit/predict ----------------
def fit_predict_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    cfg,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    train_df = train_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)
    val_df   = val_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)

    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _  = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    y_train = np.nan_to_num(train_df[label_col].to_numpy(float), nan=0.0)
    y_val   = np.nan_to_num(val_df[label_col].to_numpy(float), nan=0.0)

    # Lambdarank wants integer relevance
    if label_col == "quality":
        y_train = np.clip(np.rint(y_train), 0, 10).astype(int)
        y_val   = np.clip(np.rint(y_val),   0, 10).astype(int)
    else:
        y_train = np.clip(np.rint(y_train), 0, 3).astype(int)
        y_val   = np.clip(np.rint(y_val),   0, 3).astype(int)

    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val   = pre.transform(X_val)
    Z_test  = pre.transform(X_test)

    g_train = compute_groups(train_df)
    g_val   = compute_groups(val_df)
    w_train = compute_row_weights_equal_query(train_df)
    w_val   = compute_row_weights_equal_query(val_df)

    params = cfg.params(seed)

    dtrain = lgb.Dataset(Z_train, label=y_train, group=g_train, weight=w_train, free_raw_data=False)
    dval   = lgb.Dataset(Z_val,   label=y_val,   group=g_val,   weight=w_val,   free_raw_data=False)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    val_scores  = booster.predict(Z_val,  num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)
    return val_scores, test_scores
