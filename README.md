# Predicting Match Quality from Early Interaction Signals (Speed Dating)

A learning-to-rank recommender system built on the **UCI Speed Dating** dataset.  
We rank candidates **within an event** (`wave`) for a given participant (`iid`), using only **pre-date/profile/preference** information.

This repo implements two stages:

- **Stage A — Mutual Match Ranking:** rank who you’ll mutually match with (binary, noisy).
- **Stage B — Mutual Date Quality Ranking:** rank who you’ll have the best date with (graded, much stronger supervision).

---

## TL;DR Results

### Stage B (primary, strongest result): Mutual Date Quality Ranking

**Protocol:** wave-based nested CV (outer held-out waves for TEST; inner train/val split over remaining waves).  
**Model:** LightGBM LambdaMART (LambdaRank).  
**NDCG uses** `quality_grade` (0–10 integer).

**Final model (selected by inner-VAL mean NDCG@5):**
- **K=5  | TEST NDCG = 0.7972 ± 0.0262 | Recall@5 (quality ≥ 7) = 0.6924 | Recall@5 (quality ≥ 8) = 0.3169**
- **K=10 | TEST NDCG = 0.8572 ± 0.0263**

**Baselines (proper CV, no leakage):**
- Random:
  - K=5  | TEST NDCG **0.7717 ± 0.0235**
  - K=10 | TEST NDCG **0.8406 ± 0.0280**
- Partner mean-quality prior (computed on TRAIN only, applied to TEST):
  - K=5  | TEST NDCG **0.7674 ± 0.0213**
  - K=10 | TEST NDCG **0.8379 ± 0.0272**

**Frozen config (Stage B):**
- learning_rate = 0.03
- num_leaves = 31
- min_data_in_leaf = 150
- lambda_l2 = 20.0
- max_depth = 8
- feature_fraction = 0.85
- bagging_fraction = 0.85
- bagging_freq = 1

---

### Stage A (secondary): Mutual Match Ranking (harder/noisier)

Stage A is harder because the label is a noisy binary outcome; performance gains were smaller and more split-sensitive than Stage B.

One representative Stage A result (outer-test, wave-based CV, core features):
- K=5  | TEST NDCG ≈ 0.296 (std ≈ 0.057)
- K=10 | TEST NDCG ≈ 0.394 (std ≈ 0.056)

---

## Dataset & Query Definition (Recommender Framing)

A “query” is a participant in a specific event:
- **Query:** `(wave, iid)`
- **Candidates:** all partners `pid` that `iid` met in that wave
- **Task:** rank candidates for each query

Empirical diagnostics (Stage B dataset):
- Avg candidates/query: **14.56**
- Avg tie ratio within query (1 − unique_grades / candidates): **0.593**
- Pct queries with ≥1 “great date” (quality ≥ 8): **0.546**
- Mean count of (quality ≥ 8) per query: **1.016**

> Why random NDCG can look “high”: with many ties (same grade shared by many candidates), lots of orderings have similar DCG. The key is **lift vs baselines under the same protocol**.

---

## Labels

### Stage A — mutual match
- `match ∈ {0,1}`

### Stage B — mutual date quality (proxy)
- `quality = min(like, like_o)`  
  Enforces mutual satisfaction (one-sided interest ≠ high quality).
- For ranking: `quality_grade = round(quality)` clipped to `0..10` (integer relevance grades for LambdaRank).

---

## Methods

### Why learning-to-rank (not just classification)?
Classification asks “will this pair match?” independently.  
A recommender needs **good ordering**: best options should appear at the top.

We use:
- **LightGBM LambdaMART (LambdaRank objective)** to optimize ranking swaps directly.

### Splitting to avoid leakage
We split by **wave** so the model can’t learn event-specific quirks from the same event it’s evaluated on.

---

## Repo structure (suggested)

```
.
├── data/                 # raw dataset files (CSV, key doc)
├── scripts/              # pipeline scripts
├── results/              # generated artifacts (gitignored)
├── reports/              # committed summaries
├── requirements.txt
└── README.md
```

Recommended: keep `results/` gitignored, commit `reports/`.

---

## How to run (typical)

1) Create env, install deps
- Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- macOS/Linux/Git Bash:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Build Stage B dataset
```bash
python scripts/10_build_stage_b_quality_dataset.py
```

3) Run Stage B nested CV + baselines (the script you used to get final numbers)
```bash
python scripts/12_stage_b_baselines_and_label_variants.py
```

---

## Report
- See `reports/stage_b_final_summary.md`

---

## Future work
- Better data (larger scale, fewer ties, clearer online analogue) → consider **two-tower retrieval + re-ranker**.
- Add diversity constraints and “top-of-list” product metrics (e.g., satisfaction@K, exposure fairness).
- Robustness: test across different cohorts / waves / feature subsets.
