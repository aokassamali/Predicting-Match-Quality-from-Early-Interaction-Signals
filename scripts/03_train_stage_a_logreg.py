"""
03_train_stage_a_logreg.py

Goal
----
Train a Stage A model: "rank candidates within a wave by probability of mutual match".

Why logistic regression first?
-----------------------------
- It's fast, stable, and interpretable.
- It gives a strong baseline beyond hand-designed scoring.
- It pairs nicely with a disciplined feature policy ("pre-event only") to avoid leakage.

Core recommender framing
------------------------
For each (wave, iid), we rank the candidates pid that iid met in that wave.
We evaluate ranking quality with Recall@K, NDCG@K, and MAP@K using the mutual match label `match`.

Leakage policy (Stage A)
------------------------
We only use features that would be available BEFORE the date:
- demographics / background / stated preferences / self-perception (Time1-like)
We do NOT use scorecard ratings after the meeting (e.g., attr, like, prob, dec, etc.).

Inputs / Outputs
----------------
Input:
- results/pairs_enriched.parquet  (built by 01_build_people_pairs.py)

Outputs:
- results/logreg_stage_a_metrics.txt
- results/models/logreg_stage_a.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
MODEL_DIR = OUT_DIR / "models"
METRICS_PATH = OUT_DIR / "logreg_stage_a_metrics.txt"

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
    # User-level hit rate: did we retrieve at least one positive in top-k?
    return float(y_true_sorted[:k].sum() > 0)


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str = "match") -> dict:
    """
    Evaluate ranking metrics per (wave, iid) and average across users.
    """
    metrics = {k: {"recall": [], "ndcg": [], "map": []} for k in K_LIST}

    grouped = df.groupby(["wave", "iid"], sort=False)
    for (_, _), g in grouped:
        g_sorted = g.sort_values(score_col, ascending=False)
        y_sorted = g_sorted[label_col].to_numpy(dtype=int)

        for k in K_LIST:
            metrics[k]["recall"].append(recall_at_k(y_sorted, k))
            metrics[k]["ndcg"].append(ndcg_at_k(y_sorted, k))
            metrics[k]["map"].append(average_precision_at_k(y_sorted, k))

    # average
    out = {}
    for k in K_LIST:
        out[k] = {
            "Recall@K": float(np.mean(metrics[k]["recall"])),
            "NDCG@K": float(np.mean(metrics[k]["ndcg"])),
            "MAP@K": float(np.mean(metrics[k]["map"])),
        }
    return out


# -------------------------
# Guided feature policy
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build a leakage-safe feature frame X from pairs_enriched.

    We do two things:
    1) Select guided pre-event columns (demographics, prefs, self-perception) for both sides.
    2) Add engineered compatibility features (age diff, same race, preference dot products).

    Returns:
    - X: feature DataFrame
    - numeric_cols: list of numeric feature names
    - categorical_cols: list of categorical feature names
    """

    # --- 1) Guided pre-event-ish columns (attempt; we'll only keep those that exist) ---
    # Demographics / intent (u and v)
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

    # Stated preferences (Time1): importance weights
    prefs_u = ["attr1_1_u", "sinc1_1_u", "intel1_1_u", "fun1_1_u", "amb1_1_u", "shar1_1_u"]
    prefs_v = ["attr1_1_v", "sinc1_1_v", "intel1_1_v", "fun1_1_v", "amb1_1_v", "shar1_1_v"]

    # Self-perception (Time1-ish)
    self_u = ["attr3_1_u", "sinc3_1_u", "intel3_1_u", "fun3_1_u", "amb3_1_u"]
    self_v = ["attr3_1_v", "sinc3_1_v", "intel3_1_v", "fun3_1_v", "amb3_1_v"]

    keep_cols = pick_existing(df, base_u + base_v + prefs_u + prefs_v + self_u + self_v)

    # We split numeric vs categorical based on known column types.
    # (This is guided; if your dataset has these as numeric codes, that's fine.)
    categorical_guess = {"gender_u", "gender_v", "race_u", "race_v", "field_cd_u", "field_cd_v", "goal_u", "goal_v"}
    categorical_cols = [c for c in keep_cols if c in categorical_guess]
    numeric_cols = [c for c in keep_cols if c not in categorical_cols]

    X = df[keep_cols].copy()

    # --- 2) Engineered compatibility features (purely pre-event constructs) ---
    # These features help logistic regression capture "pair interaction" effects with a simple model.
    # They don't rely on post-date scorecards.
    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        numeric_cols.append("abs_age_diff")

    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        numeric_cols.append("same_race")

    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        numeric_cols.append("same_field")

    # preference dot products (baseline idea, but now as a feature)
    pref_u_cols = pick_existing(df, prefs_u)
    pref_v_cols = pick_existing(df, prefs_v)
    self_u_cols = pick_existing(df, self_u)
    self_v_cols = pick_existing(df, self_v)

    # We can only compute dot products if we have aligned dimensions
    # Use shared trait list where both sides have the needed columns.
    trait_map = [
        ("attr", "attr1_1_u", "attr3_1_v", "attr1_1_v", "attr3_1_u"),
        ("sinc", "sinc1_1_u", "sinc3_1_v", "sinc1_1_v", "sinc3_1_u"),
        ("intel", "intel1_1_u", "intel3_1_v", "intel1_1_v", "intel3_1_u"),
        ("fun", "fun1_1_u", "fun3_1_v", "fun1_1_v", "fun3_1_u"),
        ("amb", "amb1_1_u", "amb3_1_v", "amb1_1_v", "amb3_1_u"),
        # shar3_1 doesn't exist in this dataset version, so we omit it
    ]

    u_pref = []
    v_self = []
    v_pref = []
    u_self = []
    for _, pu, sv, pv, su in trait_map:
        if pu in df.columns and sv in df.columns:
            u_pref.append(pu)
            v_self.append(sv)
        if pv in df.columns and su in df.columns:
            v_pref.append(pv)
            u_self.append(su)

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

    # Final cleanup: remove duplicates in numeric_cols/categorical_cols while preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    numeric_cols = dedup(numeric_cols)
    categorical_cols = dedup(categorical_cols)

    return X, numeric_cols, categorical_cols


def leakage_tripwire(feature_cols: List[str]) -> None:
    """
    Fail fast if suspicious post-date columns were included.
    This is a guardrail for the "guided" approach.

    We allow Time1-like columns such as attr1_1_u, attr3_1_v.
    We forbid raw scorecard columns like 'attr', 'like', 'prob', 'dec', etc.
    """
    forbidden_exact = {
        "match", "dec", "dec_o", "like",
        "attr", "sinc", "intel", "fun", "amb", "shar",
        "prob", "met",
    }
    forbidden_suffixes = ("_o",)  # partner scorecard columns often end with _o in this dataset

    bad = []
    for c in feature_cols:
        if c in forbidden_exact:
            bad.append(c)
        if c.endswith(forbidden_suffixes):
            bad.append(c)

    if bad:
        raise ValueError(f"Leakage tripwire hit! Forbidden feature columns detected: {sorted(set(bad))}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Expectation before load:
    # - pairs_enriched exists
    # - columns wave/iid/pid/match exist
    df = pd.read_parquet(DATA_PATH)

    required = ["wave", "iid", "pid", "match"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"pairs_enriched is missing required columns: {missing_req}")

    # Drop rows with missing essentials
    df = df.dropna(subset=["wave", "iid", "pid", "match"]).copy()

    # Ensure match is 0/1 ints
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    # -------------------------
    # Build features (guided + engineered) + run leakage guardrails
    # -------------------------
    X, numeric_cols, categorical_cols = build_features(df)
    feature_cols = list(X.columns)
    leakage_tripwire(feature_cols)

    y = df["match"].to_numpy(dtype=int)

    # -------------------------
    # Train/test split by wave (tests generalization to unseen events)
    # -------------------------
    waves = np.array(sorted(df["wave"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(waves)

    n_test = max(1, int(len(waves) * TEST_WAVE_FRAC))
    test_waves = set(waves[:n_test])
    train_waves = set(waves[n_test:])

    train_mask = df["wave"].isin(train_waves)
    test_mask = df["wave"].isin(test_waves)

    X_train = X.loc[train_mask]
    y_train = y[train_mask.to_numpy()]
    X_test = X.loc[test_mask]
    y_test = y[test_mask.to_numpy()]

    # -------------------------
    # Preprocessing + model pipeline
    # -------------------------
    # Numeric: median impute + standardize
    # Categorical: impute missing + one-hot encode
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Logistic regression: balanced classes to handle match imbalance
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # -------------------------
    # Fit
    # -------------------------
    clf.fit(X_train, y_train)

    # -------------------------
    # Predict: probabilities for ranking + sanity AUC
    # -------------------------
    # We rank candidates by predicted P(match=1)
    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    # AUC is a secondary sanity metric (not the recommender objective)
    auc_train = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else float("nan")
    auc_test = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")

    # Build eval frames that still include wave/iid/pid so we can group and rank
    df_test_eval = df.loc[test_mask, ["wave", "iid", "pid", "match"]].copy()
    df_test_eval["score"] = p_test

    df_test_eval["score_std_within_user"] = df_test_eval.groupby(["wave","iid"])["score"].transform("std")
    print("Avg score std within (wave,iid):", df_test_eval["score_std_within_user"].mean())
    print("Pct users with ~0 score std:", (df_test_eval.groupby(["wave","iid"])["score"].std().fillna(0) < 1e-6).mean())


    ranking_metrics = evaluate_ranking(df_test_eval, score_col="score", label_col="match")

    # -------------------------
    # Save model + metrics
    # -------------------------
    model_path = MODEL_DIR / "logreg_stage_a.joblib"
    joblib.dump(clf, model_path)

    lines = []
    lines.append("Stage A Logistic Regression (wave split)")
    lines.append(f"Data: {DATA_PATH}")
    lines.append("")
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    lines.append(f"Unique waves: {len(waves)} | Train waves: {len(train_waves)} | Test waves: {len(test_waves)}")
    lines.append(f"Test waves: {sorted(test_waves)}")
    lines.append("")
    lines.append(f"Match rate train: {y_train.mean():.4f} | Match rate test: {y_test.mean():.4f}")
    lines.append(f"AUC train: {auc_train:.4f} | AUC test: {auc_test:.4f}")
    lines.append("")
    lines.append(f"Feature columns used: {len(feature_cols)}")
    lines.append(f"  numeric: {len(numeric_cols)} -> {numeric_cols}")
    lines.append(f"  categorical: {len(categorical_cols)} -> {categorical_cols}")
    lines.append("")
    lines.append("Ranking metrics on TEST waves (mutual match):")
    for k, vals in ranking_metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )
    lines.append("")
    lines.append(f"Saved model: {model_path}")

    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
