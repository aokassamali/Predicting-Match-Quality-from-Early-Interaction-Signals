"""
03c_train_stage_a_pairwise_logreg.py

What we're doing (intuition first)
----------------------------------
Instead of training logistic regression to predict match probability for each pair independently,
we train it to *rank* candidates for a user.

12-year-old analogy:
- For each person at the event, we show the model examples like:
  "Person A liked B (mutual match) more than C (no match)."
  The model learns a scoring rule so B gets a higher score than C.

Reality:
- For each (wave, iid), create training examples from (positive, negative) candidate pairs.
- Train on feature differences: x_pos - x_neg.
- Learn a weight vector w such that w·x_pos > w·x_neg.

Why this helps:
- It directly aligns the training objective with NDCG/MAP style ranking evaluation.

Outputs:
- results/pairwise_logreg_stage_a_metrics.txt
- results/models/pairwise_logreg_stage_a.joblib
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "results" / "pairs_enriched.parquet"
OUT_DIR = REPO_ROOT / "results"
MODEL_DIR = OUT_DIR / "models"
METRICS_PATH = OUT_DIR / "pairwise_logreg_stage_a_metrics.txt"

RANDOM_SEED = 42
TEST_WAVE_FRAC = 0.30
K_LIST = [5, 10]

# Pairwise sampling controls (keeps dataset size sane)
NEG_PER_POS = 5           # for each positive, sample this many negatives
MAX_POS_PER_USER = 10     # cap positives per user to avoid huge users dominating


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
# Feature building (same idea as prior script, kept self-contained)
# -------------------------
def pick_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Guided-ish pre-event columns (keep if present)
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

    # Engineered compatibility
    if "age_u" in df.columns and "age_v" in df.columns:
        X["abs_age_diff"] = (df["age_u"] - df["age_v"]).abs()
        numeric_cols.append("abs_age_diff")

    if "race_u" in df.columns and "race_v" in df.columns:
        X["same_race"] = (df["race_u"] == df["race_v"]).astype(int)
        numeric_cols.append("same_race")

    if "field_cd_u" in df.columns and "field_cd_v" in df.columns:
        X["same_field"] = (df["field_cd_u"] == df["field_cd_v"]).astype(int)
        numeric_cols.append("same_field")

    # preference dot products as features
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
        if c.endswith("_o"):  # partner scorecard-ish columns
            bad.append(c)
    if bad:
        raise ValueError(f"Leakage tripwire hit: {sorted(set(bad))}")


# -------------------------
# Pairwise dataset creation
# -------------------------
def make_pairwise_dataset(Z: np.ndarray, df_meta: pd.DataFrame, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pairwise training examples from transformed feature matrix Z.

    For each (wave, iid):
    - choose some positive indices (match=1)
    - for each positive, sample NEG_PER_POS negatives (match=0)
    - create diff = z_pos - z_neg labeled 1
    - also include -diff labeled 0 (symmetry) to balance
    """
    diffs = []
    labels = []

    for (_, _), g in df_meta.groupby(["wave", "iid"], sort=False):
        idx = g.index.to_numpy()
        y = g["match"].to_numpy(dtype=int)

        pos_idx = idx[y == 1]
        neg_idx = idx[y == 0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        # cap positives per user to avoid domination
        if len(pos_idx) > MAX_POS_PER_USER:
            pos_idx = rng.choice(pos_idx, size=MAX_POS_PER_USER, replace=False)

        for p in pos_idx:
            # sample negatives for this positive
            n_samp = min(NEG_PER_POS, len(neg_idx))
            ns = rng.choice(neg_idx, size=n_samp, replace=False)

            for n in ns:
                row = Z[p] - Z[n]
                # If it's sparse, convert to dense; if it's already dense numpy, just use it.
                if hasattr(row, "toarray"):
                    row = row.toarray()
                d = np.asarray(row).ravel()
                diffs.append(d); labels.append(1)
                diffs.append(-d); labels.append(0)

    X_pair = np.vstack(diffs) if diffs else np.zeros((0, Z.shape[1]))
    y_pair = np.array(labels, dtype=int)
    return X_pair, y_pair


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load and basic clean
    df = pd.read_parquet(DATA_PATH).dropna(subset=["wave", "iid", "pid", "match"]).copy()
    df["match"] = pd.to_numeric(df["match"], errors="coerce").fillna(0).astype(int)

    # Wave split (same philosophy as before)
    waves = np.array(sorted(df["wave"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(waves)
    n_test = max(1, int(len(waves) * TEST_WAVE_FRAC))
    test_waves = set(waves[:n_test])
    train_waves = set(waves[n_test:])

    train_mask = df["wave"].isin(train_waves)
    test_mask = df["wave"].isin(test_waves)

    df_train = df.loc[train_mask].copy().reset_index(drop=True)
    df_test = df.loc[test_mask, ["wave", "iid", "pid", "match"]].copy().reset_index(drop=True)

    # Build features + leakage guardrails
    X_all, numeric_cols, categorical_cols = build_features(df)
    leakage_tripwire(list(X_all.columns))

    X_train = X_all.loc[train_mask].copy().reset_index(drop=True)
    X_test = X_all.loc[test_mask].copy().reset_index(drop=True)
    y_train = df_train["match"].to_numpy(dtype=int)

    # Preprocess: fit ONLY on train, transform both
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(
        [("num", num_tf, numeric_cols), ("cat", cat_tf, categorical_cols)],
        remainder="drop",
    )

    # Expectation: Z_train/Z_test become numeric matrices with same column space
    Z_train = pre.fit_transform(X_train)
    Z_test = pre.transform(X_test)

    # Create pairwise training set
    # IMPORTANT: df_train's index aligns with rows in Z_train because we didn't reset indices.
    X_pair, y_pair = make_pairwise_dataset(Z_train, df_train[["wave", "iid", "match"]], rng)

    if X_pair.shape[0] == 0:
        raise RuntimeError("Pairwise dataset ended up empty. Check that training split has positives and negatives per user.")

    # Train logistic regression on pairwise diffs
    ranker = LogisticRegression(max_iter=2000, solver="lbfgs")
    ranker.fit(X_pair, y_pair)

    # Score test pairs by utility w·z (sigmoid not needed for ranking)
    test_scores = ranker.decision_function(Z_test)

    df_test["score"] = test_scores
    metrics = evaluate_ranking(df_test, score_col="score", label_col="match")

    # Save model bundle (preprocessor + ranker)
    bundle = {"preprocessor": pre, "ranker": ranker, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}
    model_path = MODEL_DIR / "pairwise_logreg_stage_a.joblib"
    joblib.dump(bundle, model_path)

    lines = []
    lines.append("Stage A Pairwise Logistic Regression (RankLogReg) — wave split")
    lines.append(f"Test waves: {sorted(test_waves)}")
    lines.append(f"Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")
    lines.append(f"Train match rate: {df_train['match'].mean():.4f} | Test match rate: {df_test['match'].mean():.4f}")
    lines.append(f"Pairwise examples: {X_pair.shape[0]:,} (NEG_PER_POS={NEG_PER_POS}, MAX_POS_PER_USER={MAX_POS_PER_USER})")
    lines.append("")
    lines.append("Ranking metrics on TEST waves (mutual match):")
    for k, vals in metrics.items():
        lines.append(
            f"K={k:>2} | Recall@K={vals['Recall@K']:.4f} | NDCG@K={vals['NDCG@K']:.4f} | MAP@K={vals['MAP@K']:.4f}"
        )
    lines.append("")
    lines.append(f"Saved model: {model_path}")

    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {METRICS_PATH}")


if __name__ == "__main__":
    main()
