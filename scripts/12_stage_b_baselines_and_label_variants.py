"""
12_stage_b_baselines_and_label_variants.py

Stage B: Compare baselines + label variants for ranking mutual date quality.

Baselines
---------
1) Random within query (wave,iid)
2) Preference dot-product baseline (pre-date prefs vs self-ratings)
3) Partner global mean quality prior (rank by avg quality of pid)

Models
------
LightGBM LambdaMART trained with nested wave CV:
- Train label = quality_bin (0..3)
- Train label = quality (0..10)

Evaluation
----------
- NDCG@K: graded relevance (we compute using quality_bin by default)
- Recall@K: "did we surface any good date" with thresholds >=7 and >=8
- MAP@K: same thresholds

Run
---
python scripts/12_stage_b_baselines_and_label_variants.py
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

RANDOM_SEED_BASE = 7
N_REPEATS = 5
N_FOLDS = 5
INNER_VAL_FRAC = 0.20
K_LIST = [5, 10]

NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 300

# Two "good date" thresholds for binary recall/MAP
THRESHOLDS = [7.0, 8.0]


# ---------------- Metrics ----------------
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
    hits = 0
    precisions = []
    for i, rel in enumerate(y, start=1):
        if rel == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(y_true_sorted_bin: np.ndarray, k: int) -> float:
    return float(y_true_sorted_bin[:k].sum() > 0)


def evaluate(df: pd.DataFrame, score_col: str, graded_col: str = "quality_bin") -> Dict:
    """
    NDCG uses graded relevance (quality_bin).
    MAP/Recall computed for thresholds over raw quality.
    """
    out = {"ndcg": {}, "map": {t: {} for t in THRESHOLDS}, "recall": {t: {} for t in THRESHOLDS}}

    for k in K_LIST:
        ndcgs = []
        maps = {t: [] for t in THRESHOLDS}
        recalls = {t: [] for t in THRESHOLDS}

        for (_, _), g in df.groupby(["wave", "iid"], sort=False):
            g_sorted = g.sort_values(score_col, ascending=False)

            rel_graded = np.nan_to_num(g_sorted[graded_col].to_numpy(float), nan=0.0)
            ndcgs.append(ndcg_at_k(rel_graded, k))

            q = g_sorted["quality"].to_numpy(float)
            for t in THRESHOLDS:
                rel_bin = (q >= t).astype(int)
                maps[t].append(average_precision_at_k(rel_bin, k))
                recalls[t].append(recall_at_k(rel_bin, k))

        out["ndcg"][k] = float(np.mean(ndcgs))
        for t in THRESHOLDS:
            out["map"][t][k] = float(np.mean(maps[t]))
            out["recall"][t][k] = float(np.mean(recalls[t]))

    return out


# ---------------- Groups/weights ----------------
def compute_groups(df_sorted: pd.DataFrame) -> np.ndarray:
    return df_sorted.groupby(["wave", "iid"], sort=False).size().to_numpy(dtype=int)


def compute_row_weights_equal_query(df_sorted: pd.DataFrame) -> np.ndarray:
    sizes_per_row = df_sorted.groupby(["wave", "iid"], sort=False)["pid"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(sizes_per_row, 1.0)


# ---------------- Features (core) ----------------
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
    base_u = ["age_u","gender_u","race_u","field_cd_u","goal_u","imprace_u","imprelig_u","expnum_u","exphappy_u","date_u","go_out_u"]
    base_v = ["age_v","gender_v","race_v","field_cd_v","goal_v","imprace_v","imprelig_v","expnum_v","exphappy_v","date_v","go_out_v"]
    prefs_u = ["attr1_1_u","sinc1_1_u","intel1_1_u","fun1_1_u","amb1_1_u","shar1_1_u"]
    prefs_v = ["attr1_1_v","sinc1_1_v","intel1_1_v","fun1_1_v","amb1_1_v","shar1_1_v"]
    self_u = ["attr3_1_u","sinc3_1_u","intel3_1_u","fun3_1_u","amb3_1_u"]
    self_v = ["attr3_1_v","sinc3_1_v","intel3_1_v","fun3_1_v","amb3_1_v"]

    keep = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)
    df2 = add_compat_features(df)

    # include compat cols if created
    for c in ["abs_age_diff","same_race","same_field"]:
        if c in df2.columns and c not in keep:
            keep.append(c)

    cat_guess = {"gender_u","gender_v","race_u","race_v","field_cd_u","field_cd_v","goal_u","goal_v"}
    cat_cols = [c for c in keep if c in cat_guess]
    num_cols = [c for c in keep if c not in cat_cols]

    X = df2[keep].copy()

    # dot products (preference alignment)
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


# ---------------- Models ----------------
@dataclass(frozen=True)
class Config:
    lr: float
    leaves: int
    minleaf: int
    ff: float
    bf: float
    l2: float
    md: int

    def params(self, seed: int) -> Dict:
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5],
            "verbosity": -1,
            "seed": seed,

            "learning_rate": self.lr,
            "num_leaves": self.leaves,
            "min_data_in_leaf": self.minleaf,
            "feature_fraction": self.ff,
            "bagging_fraction": self.bf,
            "bagging_freq": 1,
            "lambda_l2": self.l2,
            "max_depth": self.md,
            "label_gain": list(range(0, 11)),
        }


def make_grid() -> List[Config]:
    # Keep tiny: 18 configs
    grid = []
    for leaves, minleaf, l2 in product([15, 31], [40, 80, 150], [1.0, 5.0, 20.0]):
        grid.append(Config(0.03, leaves, minleaf, 0.85, 0.85, l2, 8))
    return grid


def fit_predict_lgbm(train_df, val_df, test_df, label_col: str, cfg: Config, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_df = train_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)
    val_df = val_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)
    test_df = test_df.sort_values(["wave","iid","pid"]).reset_index(drop=True)

    X_train, num_cols, cat_cols = build_core_features(train_df)
    X_val, _, _ = build_core_features(val_df)
    X_test, _, _ = build_core_features(test_df)

    y_train = np.nan_to_num(train_df[label_col].to_numpy(float), nan=0.0)
    y_val = np.nan_to_num(val_df[label_col].to_numpy(float), nan=0.0)

    if label_col == "quality":
        # LightGBM ranking requires integer relevance labels.
        # Convert to 0..10 integer grades.
        y_train = np.clip(np.rint(y_train), 0, 10).astype(int)
        y_val = np.clip(np.rint(y_val), 0, 10).astype(int)
    else:
        # quality_bin already numeric 0..3; ensure int-like
        y_train = np.clip(np.rint(y_train), 0, 3).astype(int)
        y_val = np.clip(np.rint(y_val), 0, 3).astype(int)



    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)], remainder="drop")

    Z_train = pre.fit_transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)

    g_train = compute_groups(train_df)
    g_val = compute_groups(val_df)

    w_train = compute_row_weights_equal_query(train_df)
    w_val = compute_row_weights_equal_query(val_df)

    dtrain = lgb.Dataset(Z_train, label=y_train, group=g_train, weight=w_train, free_raw_data=False)
    dval = lgb.Dataset(Z_val, label=y_val, group=g_val, weight=w_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        cfg.params(seed),
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train","val"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    val_scores = booster.predict(Z_val, num_iteration=booster.best_iteration)
    test_scores = booster.predict(Z_test, num_iteration=booster.best_iteration)
    return val_scores, test_scores


# ---------------- Baselines ----------------
def add_baseline_scores(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()

    # Random baseline
    out["score_random"] = rng.random(len(out))

    # Dot-product baseline (if available)
    # Uses preference dot products if present, else creates them from columns where possible.
    # Here we assume build_core_features adds pref_dot_self_* columns only inside model pipeline,
    # so we approximate dot baseline using raw columns if present.
    # If these columns are missing, score becomes 0 and baseline is trivial (still okay).
    out["score_dot"] = 0.0
    if "pref_dot_self_v" in out.columns:
        out["score_dot"] = out["pref_dot_self_v"].fillna(0.0)
    # If not present, we keep 0.0 (safe).

    # Partner global mean quality prior (popularity-ish)
    pid_mean = out.groupby("pid")["quality"].mean()
    out["score_pid_mean_quality"] = out["pid"].map(pid_mean).fillna(out["quality"].mean())

    return out


def main() -> None:
    """
    Stage B (quality) final evaluation:
    - Train label: quality (cast to int grades 0..10 for LambdaRank)
    - Baselines evaluated properly under outer-fold CV (no leakage)
      * random (within test fold)
      * pid-mean prior computed on TRAIN fold only, applied to TEST fold
        + tie-break jitter so it doesn't get crushed by arbitrary ordering
    - Metrics:
      * NDCG computed on quality_grade (0..10) for more resolution
      * Recall thresholds (>=7, >=8) from raw quality
    """
    df = pd.read_parquet(DATA_PATH).copy()
    df = df.dropna(subset=["wave", "iid", "pid", "quality"]).copy()
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype(int)

    # Keep bins if present (safe), but we will evaluate NDCG using quality_grade for more signal.
    if "quality_bin" in df.columns:
        df["quality_bin"] = np.nan_to_num(df["quality_bin"].to_numpy(float), nan=0.0)
    else:
        df["quality_bin"] = 0.0

    # 0..10 integer grade for evaluation + can be used as training label too
    df["quality_grade"] = np.clip(np.rint(pd.to_numeric(df["quality"], errors="coerce")), 0, 10).astype(int)

    all_waves = df["wave"].unique()
    grid = make_grid()

    label_col = "quality"  # fit_predict_lgbm already casts quality -> int grades for ranking
    print(f"\n\n===== MODEL: LightGBM LambdaMART | train_label={label_col} =====")

    rows_model: list[dict] = []
    rows_baselines: list[dict] = []

    for r in range(N_REPEATS):
        seed = RANDOM_SEED_BASE + r
        rng = np.random.default_rng(seed)
        folds = make_folds(all_waves, N_FOLDS, rng)

        for fold_idx in range(N_FOLDS):
            # Outer test split
            test_waves = folds[fold_idx]
            remaining = [w for i, f in enumerate(folds) if i != fold_idx for w in f]
            train_waves, val_waves = inner_train_val_split(remaining, INNER_VAL_FRAC, rng)

            train_df = df[df["wave"].isin(train_waves)]
            val_df = df[df["wave"].isin(val_waves)]
            test_df = df[df["wave"].isin(test_waves)]

            # Deterministic RNG for this fold (used for baselines + tie-break jitter)
            rng_fold = np.random.default_rng(seed * 1000 + fold_idx)

            # -------------------------------
            # Baselines (computed once per fold; NO leakage)
            # -------------------------------

            # (A) Random baseline
            test_eval_rand = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            test_eval_rand["score"] = rng_fold.random(len(test_eval_rand))
            m_rand = evaluate(test_eval_rand, "score", graded_col="quality_grade")

            rows_baselines.append({
                "baseline": "random",
                "repeat": r,
                "fold": fold_idx,
                "test_ndcg5": m_rand["ndcg"][5],
                "test_ndcg10": m_rand["ndcg"][10],
                "test_recall5_ge7": m_rand["recall"][7.0][5],
                "test_recall5_ge8": m_rand["recall"][8.0][5],
            })

            # (B) Partner mean-quality prior computed on TRAIN only (then applied to TEST)
            pid_mean_train = train_df.groupby("pid")["quality"].mean()
            global_mean = float(train_df["quality"].mean())

            test_eval_pid = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
            test_eval_pid["score"] = test_eval_pid["pid"].map(pid_mean_train).fillna(global_mean)

            # Tie-break jitter so we don't get punished by arbitrary ordering when many pids are unseen
            test_eval_pid["score"] += 1e-6 * rng_fold.random(len(test_eval_pid))

            m_pid = evaluate(test_eval_pid, "score", graded_col="quality_grade")

            rows_baselines.append({
                "baseline": "pid_mean_train_only",
                "repeat": r,
                "fold": fold_idx,
                "test_ndcg5": m_pid["ndcg"][5],
                "test_ndcg10": m_pid["ndcg"][10],
                "test_recall5_ge7": m_pid["recall"][7.0][5],
                "test_recall5_ge8": m_pid["recall"][8.0][5],
            })

            # -------------------------------
            # Models (loop over configs)
            # -------------------------------
            for cfg in grid:
                cfg_name = f"leaves={cfg.leaves}_minleaf={cfg.minleaf}_l2={cfg.l2}"

                val_scores, test_scores = fit_predict_lgbm(
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    label_col=label_col,
                    cfg=cfg,
                    seed=seed,
                )

                val_eval = val_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
                test_eval = test_df[["wave", "iid", "pid", "quality", "quality_bin", "quality_grade"]].copy()
                val_eval["score"] = val_scores
                test_eval["score"] = test_scores

                # Evaluate NDCG on quality_grade (0..10) for more resolution than 0..3 bins
                val_m = evaluate(val_eval, "score", graded_col="quality_grade")
                test_m = evaluate(test_eval, "score", graded_col="quality_grade")

                rows_model.append({
                    "cfg": cfg_name,
                    "repeat": r,
                    "fold": fold_idx,
                    "val_ndcg5": val_m["ndcg"][5],
                    "test_ndcg5": test_m["ndcg"][5],
                    "test_ndcg10": test_m["ndcg"][10],
                    "test_recall5_ge7": test_m["recall"][7.0][5],
                    "test_recall5_ge8": test_m["recall"][8.0][5],
                })

    res = pd.DataFrame(rows_model)
    res_b = pd.DataFrame(rows_baselines)

    # -------------------------------
    # Baseline summary (outer-test only)
    # -------------------------------
    print("\n===== BASELINES (proper CV, outer-test only) =====")
    for bname in sorted(res_b["baseline"].unique()):
        sub = res_b[res_b["baseline"] == bname]
        print(f"\nBaseline: {bname}")
        print(
            f"  K=5  | TEST NDCG {sub['test_ndcg5'].mean():.4f} ± {sub['test_ndcg5'].std():.4f} | "
            f"Recall>=7 {sub['test_recall5_ge7'].mean():.4f} | Recall>=8 {sub['test_recall5_ge8'].mean():.4f}"
        )
        print(f"  K=10 | TEST NDCG {sub['test_ndcg10'].mean():.4f} ± {sub['test_ndcg10'].std():.4f}")

    # -------------------------------
    # Model selection + reporting
    # -------------------------------
    sel = res.groupby("cfg")["val_ndcg5"].mean().sort_values(ascending=False)
    best_cfg = sel.index[0]
    sub = res[res["cfg"] == best_cfg]

    print(f"\nSelected best cfg by INNER-VAL mean NDCG@5: {best_cfg}")
    print(
        f"K=5  | TEST NDCG {sub['test_ndcg5'].mean():.4f} ± {sub['test_ndcg5'].std():.4f} | "
        f"Recall>=7 {sub['test_recall5_ge7'].mean():.4f} | Recall>=8 {sub['test_recall5_ge8'].mean():.4f}"
    )
    print(f"K=10 | TEST NDCG {sub['test_ndcg10'].mean():.4f} ± {sub['test_ndcg10'].std():.4f}")



if __name__ == "__main__":
    main()
