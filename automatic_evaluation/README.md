**Repository: Automatic Entity Evaluation**

Evaluate predicted entity labels against ground truth using rule-based, fuzzy, and semantic matching.

---

### Examples

Assume for the entity **Victim** the tuning process yields:

- **Tuned threshold**: 0.75
- **Lower bound (5th percentile)**: 0.60

| Strategy | Final threshold | Rationale                                                                  |
| -------- | --------------- | -------------------------------------------------------------------------- |
| tuned    | 0.75            | Maximizes historical accuracy (precision+recall).                          |
| lower    | 0.60            | Ensures at least 95% recall of true positives.                             |
| balanced | 0.75            | Meets recall requirement while retaining precision (max of 0.75 and 0.60). |

---

## 🚀 Features

- **Multi-stage resolution**:
  1. **Rule**: exact or missing match
  2. **Fuzzy**: token-level similarity (RapidFuzz)
  3. **Semantic**: embedding-based cosine similarity (SentenceTransformer)
- **Threshold tuning** (or load saved): select per-entity fuzzy/semantic cutoffs
- **Metrics & plots**: precision, recall, F1; similarity distributions; resolution breakdown

---

## ⚙️ Requirements

- Python 3.8+
- pandas, numpy, rapidfuzz, sentence-transformers, matplotlib

```bash
pip install pandas numpy rapidfuzz sentence-transformers matplotlib
```

---

## 🔧 Configuration

Edit at top of script:

```python
EVAL_METHOD              # human_evaluation name (e.g. "parachute")
INPUT_CSV                # path to input CSV (gold + predictions)
BASE_OUTPUT_DIR          # root for outputs
FUZZY_SELECTION_STRATEGY # "tuned", "lower", or "balanced"
SEMANTIC_SELECTION_STRATEGY
LOAD_SAVED_THRESHOLDS    # True to skip recomputing
THRESHOLDS_JSON          # path to saved thresholds (if loading)
```

Customize `ENTITY_CONFIG` to specify columns:

```python
ENTITY_CONFIG = [
  { "col":"Victim",         "pred":"Victim_P",         "flags":[..] },
  { "col":"Perpetrator",    "pred":"Perpetrator_P",    "flags":[..] },
  { "col":"Mode of Demise", "pred":"Mode of Demise_P", "flags":[..] },
]
```

### Selection Strategies

- **tuned**: Use the threshold that maximizes accuracy on the tuning grid for each entity (optimizes overall precision and recall balance based on historical data).
- **lower**: Set the threshold to the lower-bound (5th percentile) of similarity scores among true-positive examples. This ensures that at least 95% of correctly matched pairs meet or exceed the threshold, prioritizing recall even if precision may decrease.
- **balanced**: For each entity, choose the higher value between the **tuned** and **lower** thresholds. This guarantees you meet the minimum recall requirement (`lower`) while still leveraging the accuracy-optimized threshold (`tuned`) to maintain precision.

---

## 📈 Workflow

1. **Load CSV** (expects `Gold` & `Pred` columns per entity).
2. **Initialize flags** and empty “Resolution” column.
3. **Compute similarity metrics**:
   - Normalize text (lowercasing, punctuation removal).
   - Fuzzy ratio and cosine similarity for each non‐missing pair.
   - Save to `*_similarity_metrics.csv`.
4. **Plot cosine distributions** (`*_cosine_distributions.png`).
5. **Threshold selection**:
   - **Load** saved JSON or **tune** over grids (`0.0–1.0`).
   - Plot tuning curves (`*_fuzzy_tuning_plot.png`, `*_semantic_tuning_plot.png`).
   - Derive final thresholds per‐entity and overall.
   - Save thresholds JSON.
6. **Full resolution**: apply multi‐stage matching using final thresholds; populate TP/FP/FN/TN flags.
7. **Generate merged report** (`*_merged_report.csv`):
   - Counts resolved by rule/fuzzy/semantic.
   - Manual vs. auto precision/recall/F1.
   - Thresholds and deltas.
8. **Plot**:
   - Resolution breakdown (`*_resolution_plot.png`, `*_overall_resolution.png`).
   - PRF comparison with Δ labels (`*_prf_precision.png`, `_recall.png`, `_f1.png`).

---

## 📂 Outputs

All files are saved under:

```
$BASE_OUTPUT_DIR/<input_basename>/
```

- `parachute_<basename>.csv` – raw flags + resolutions
- `*_merged_report.csv` – summary metrics
- `*_similarity_metrics.csv` – raw sim scores
- `*_tuning.csv` – tuning results
- `*.png` – plots

---

## ▶️ Usage

```bash
python3 evaluate_entities.py
```

Ensure paths and strategies are set in script header. Upon completion, review CSVs & plots in the output folder.

