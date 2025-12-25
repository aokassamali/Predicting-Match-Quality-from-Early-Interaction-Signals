# Stage B Final Summary — Date Quality Ranking

## Goal
Given a user (iid) and the set of partners they met in the same speed-dating event (wave), rank candidates by **mutual date quality** using only **pre-date / profile / preference features**.

## Dataset & Query Structure
- Rows (candidate pairs) loaded: **8,368**
- Kept (non-null mutual quality): **8,022**
- Queries (wave, iid): **551**
- Avg candidates per query: **14.56** (median 16)

### Why random NDCG can look high
- Avg tie ratio within a query (1 − unique_grades / candidates): **0.593**
  - Many candidates share the same quality grade, so many rankings produce similar DCG.
- “Great date” prevalence (mutual quality ≥ 8):
  - Pct queries with ≥1 item: **0.546**
  - Mean count per query: **1.02**
  - This creates a meaningful but sparse “top-of-list” target.

## Label (Mutual Quality Proxy)
We define mutual quality as:

- `quality = min(like, like_o)`  (both sides must rate the interaction highly)
- For ranking metrics we use `quality_grade = round(quality)` clipped to **0..10**.

## Modeling Approach
### Model
- **LightGBM LambdaMART (LambdaRank / learning-to-rank)**

### Validation Protocol
- **Wave-based nested cross-validation** to prevent event leakage:
  - Outer folds = held-out waves (test)
  - Inner split = train vs val waves
- Selection metric: **INNER-VAL mean NDCG@5**
- Reported metrics: OUTER-TEST mean ± std across repeats/folds

### Frozen “Final” Hyperparameters
Selected by INNER-VAL NDCG@5:

- learning_rate = **0.03**
- num_leaves = **31**
- min_data_in_leaf = **150**
- lambda_l2 = **20.0**
- max_depth = **8**
- feature_fraction = **0.85**
- bagging_fraction = **0.85**
- bagging_freq = **1**

## Results (Outer-TEST; mean ± std)
### Baselines (no leakage; computed per fold)
- **Random**
  - K=5  NDCG **0.7717 ± 0.0235** | Recall@5 (≥7) **0.6550** | Recall@5 (≥8) **0.2768**
  - K=10 NDCG **0.8406 ± 0.0280**
- **Partner Mean Quality Prior (train-only)**
  - K=5  NDCG **0.7674 ± 0.0213** | Recall@5 (≥7) **0.6315** | Recall@5 (≥8) **0.2755**
  - K=10 NDCG **0.8379 ± 0.0272**

### Final Model (LambdaMART; pre-date features only)
- **Selected best config:** `leaves=31_minleaf=150_l2=20.0`
  - K=5  NDCG **0.7972 ± 0.0262** | Recall@5 (≥7) **0.6924** | Recall@5 (≥8) **0.3169**
  - K=10 NDCG **0.8572 ± 0.0263**

## Interpretation
- Stage B is substantially more learnable than Stage A (mutual match) because it provides **graded supervision** rather than a noisy binary outcome.
- Even with a high tie rate, the model provides consistent lift over baselines on **top-of-list quality** (NDCG@5 and Recall@5 for high-quality dates).

## Notes / Caveats
- Random NDCG is high due to heavy grade ties within queries; interpret results primarily as **lift vs baselines** under the same protocol.
- The dataset is limited to speed-dating events (waves) and may not reflect modern online dating dynamics.
