"""
14_make_stage_b_figure.py

Generates:
1) A simple NDCG comparison bar plot (random / pid-mean / model)
2) A Top-10 feature table (cleaned feature names using the extracted data dictionary)
3) A SHAP-style feature-importance plot using LightGBM's pred_contrib

Assumes you already ran:
- scripts/13_stage_b_final_frozen.py  -> results/stage_b_final_metrics.csv
- scripts/10_build_stage_b_quality_dataset.py -> results/pairs_stage_b_quality.parquet

Outputs (under results/figures/):
- stage_b_ndcg_comparison.png
- stage_b_top10_features_shap.csv
- stage_b_feature_importance_shap.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from stage_b_lib import build_core_features, compute_groups

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "results" / "stage_b_final_metrics.csv"
DATA_PATH = REPO_ROOT / "results" / "pairs_stage_b_quality.parquet"

FIG_DIR = REPO_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PLOT_NDCG = FIG_DIR / "stage_b_ndcg_comparison.png"
OUT_TOP10_CSV = FIG_DIR / "stage_b_top10_features_shap.csv"
OUT_PLOT_SHAP = FIG_DIR / "stage_b_feature_importance_shap.png"

# Optional: extracted data dictionary + code maps
KEY_MAP_CSV = REPO_ROOT / "reports" / "speed_dating_key_extracted.csv"
CODE_MAP_JSON = REPO_ROOT / "reports" / "speed_dating_code_maps.json"


def _load_key_maps() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Load column->description and code maps if available."""
    desc_map: dict[str, str] = {}
    code_maps: dict[str, dict[str, str]] = {}

    if KEY_MAP_CSV.exists():
        df = pd.read_csv(KEY_MAP_CSV)
        for _, r in df.iterrows():
            desc_map[str(r["column"])] = str(r["description"])

    if CODE_MAP_JSON.exists():
        with open(CODE_MAP_JSON, "r", encoding="utf-8") as f:
            code_maps = json.load(f)

    return desc_map, code_maps


def _role_prefix(var: str) -> str:
    if var.endswith("_u"):
        return "User: "
    if var.endswith("_v"):
        return "Partner: "
    return ""


def _strip_role(var: str) -> tuple[str, str]:
    """Return (base_var, role_prefix)."""
    if var.endswith("_u"):
        return var[:-2], "User: "
    if var.endswith("_v"):
        return var[:-2], "Partner: "
    return var, ""


def clean_feature_name(raw: str, desc_map: dict[str, str], code_maps: dict[str, dict[str, str]]) -> str:
    """
    Convert sklearn/lgbm feature names to human-readable labels:
    - remove 'num__' / 'cat__'
    - collapse one-hot names like race_u_2 -> 'User: Race = 2 (Asian)' if mapping exists
    - map known engineered features to descriptions
    """
    name = raw
    name = re.sub(r"^(num__|cat__)", "", name)

    # One-hot pattern: <var>_<code> where <var> is known categorical base
    # We treat trailing _<number> as category code if the part before it looks like a variable.
    m = re.match(r"^(.+)_(-?\d+(?:\.\d+)?)$", name)
    if m:
        var_full, code = m.group(1), m.group(2)

        # normalize "2.0" -> "2"
        if code.endswith(".0"):
            code = code[:-2]

        base_var, role = _strip_role(var_full)

        label = None
        if base_var in code_maps and code in code_maps[base_var]:
            label = code_maps[base_var][code]

        desc = desc_map.get(base_var, base_var)
        if label:
            return f"{role}{desc} = {code} ({label})"
        return f"{role}{desc} = {code}"


    # Non one-hot vars
    base_var, role = _strip_role(name)
    if base_var in desc_map:
        # Keep short; many descriptions in the original key are long
        short = desc_map[base_var]
        short = short.split(".")[0].strip()
        return f"{role}{short}"

    # Engineered fallbacks
    engineered = {
        "abs_age_diff": "Absolute age difference",
        "same_race": "Same race indicator",
        "pref_dot_self_v": "Preference alignment (user prefs · partner self)",
        "pref_v_dot_self_u": "Preference alignment (partner prefs · user self)",
    }
    if base_var in engineered:
        return engineered[base_var]

    return f"{role}{base_var}"

def shorten_label(s: str) -> str:
    # normalize role prefix first
    s = s.replace("Partner: ", "").replace("User: ", "")

    # exact-ish rewrites (order matters)
    rules = [
        (r"What they think the OTHER person values: fun", "V:Partner fun-ness"),
        (r"Overall, on a scale of 1-10, how happy do you expect to be with the people you meet during the speed-dating event\?", "V:Excitement for event"),
        (r"How important is it to you \(on a scale of 1-10\) that a person you date be of the same religious background\?", "V:Same religion"),
        (r"Absolute age difference between \(iid, pid\)", "U:Absolute age difference"),
        (r"How important is it to you \(on a scale of 1-10\) that a person you date be of the same racial/ethnic background\?", "V:Same ethnicity"),
        (r"Stated preference weight on shared interests \(Time 1\)", "V:Shared interests"),
        (r"What they think the OTHER person values: ambition \(Time 1\)", "V:Partner ambitiousness"),
        (r"Age \(years\)", "V:Age"),
        (r"Stated preference weight on intelligence \(Time 1\)", "V:Partner intelligence"),
        (r"Stated preference weight on sincerity \(Time 1\)", "V:Partner sincerity"),
    ]

    for pat, rep in rules:
        s = re.sub(pat, rep, s)

    # remove Time markers + extra whitespace
    s = s.replace(" (Time 1)", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_ndcg_comparison_plot() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}. Run Script 13 first.")

    dfm = pd.read_csv(METRICS_PATH)
    # Expect columns as written by script 13
    # Summarize NDCG@5 and NDCG@10
    out_rows = []
    for k in [5, 10]:
        sub = dfm[dfm["k"] == k]
        out_rows.append({
            "k": k,
            "random": sub["baseline_random_ndcg"].mean(),
            "pid_mean": sub["baseline_pidmean_ndcg"].mean(),
            "model": sub["model_test_ndcg"].mean(),
        })
    s = pd.DataFrame(out_rows)

    # Plot: grouped bars for K=5 and K=10
    x = np.arange(len(s))
    w = 0.25

    plt.figure()
    plt.bar(x - w, s["random"], width=w, label="Random")
    plt.bar(x, s["pid_mean"], width=w, label="Pid-mean (train-only)")
    plt.bar(x + w, s["model"], width=w, label="LambdaMART (model)")
    plt.xticks(x, [f"K={k}" for k in s["k"]])
    plt.ylabel("NDCG")
    plt.title("Stage B: NDCG comparison (outer-test mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT_NDCG, dpi=200)
    plt.close()


def compute_shap_top10() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run Script 10 first.")

    df = pd.read_parquet(DATA_PATH).copy()
    df = df.dropna(subset=["wave", "iid", "pid", "quality"]).copy()
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype(int)

    # IMPORTANT: keep stable query grouping
    df = df.sort_values(["wave", "iid", "pid"]).reset_index(drop=True)

    # Label for training the interpretability model
    y = pd.to_numeric(df["quality"], errors="coerce").to_numpy(float)
    y = np.clip(np.rint(np.nan_to_num(y, nan=0.0)), 0, 10).astype(int)

    # Build “core” features (matches Stage 13/12 feature set)
    X_df, num_cols, cat_cols = build_core_features(df)

    # Preprocess exactly like Stage B training
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        [("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
    )

    Z = pre.fit_transform(X_df)
    feature_names = list(pre.get_feature_names_out())

    groups = compute_groups(df)

    # Train a single full-data model for interpretability only
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "verbosity": -1,
        "seed": 7,
        "label_gain": list(range(0, 11)),

        # Frozen config (same as Stage 13)
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 150,
        "lambda_l2": 20.0,
        "max_depth": 8,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
    }

    dtrain = lgb.Dataset(Z, label=y, group=groups, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=200)

    # SHAP-style contributions: (n_rows, n_features + 1 bias)
    # To keep memory bounded, sample rows
    n = Z.shape[0]
    sample_n = min(2000, n)
    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=sample_n, replace=False)

    Zs = Z[idx]
    contrib = np.asarray(booster.predict(Zs, pred_contrib=True))
    contrib_feat = contrib[:, :-1]  # drop bias term

    mean_abs = np.mean(np.abs(contrib_feat), axis=0)

    df_imp = (
        pd.DataFrame({"feature_raw": feature_names, "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True)
    )

    # Clean names using the extracted key (if present)
    desc_map, code_maps = _load_key_maps()
    df_imp["feature_clean"] = df_imp["feature_raw"].apply(
        lambda s: clean_feature_name(str(s), desc_map, code_maps)
    )

    top10 = df_imp.head(10).copy()
    return top10




def plot_shap_top10(top10: pd.DataFrame) -> None:
    top10_rev = top10.iloc[::-1].copy()

    top10_rev["label_short"] = top10_rev["feature_clean"].apply(shorten_label)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10_rev["label_short"], top10_rev["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP contribution| (LightGBM pred_contrib)")
    ax.set_title("Stage B: Top features (SHAP-style importance)")

    fig.tight_layout()
    fig.savefig(OUT_PLOT_SHAP, dpi=200, bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    make_ndcg_comparison_plot()
    top10 = compute_shap_top10()
    top10["feature_short"] = top10["feature_clean"].apply(shorten_label)
    top10.to_csv(OUT_TOP10_CSV, index=False)
    plot_shap_top10(top10)

    print("Wrote:")
    print(" -", OUT_PLOT_NDCG)
    print(" -", OUT_TOP10_CSV)
    print(" -", OUT_PLOT_SHAP)


if __name__ == "__main__":
    main()
