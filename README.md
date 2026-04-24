# Transaction Anomaly Detection

A reproducible, notebook-driven pipeline that identifies fraudulent transactions
in a 1.3M-row credit-card dataset. The deliverable is a single annotated
Jupyter notebook that walks through **two complementary tracks**:

1. **Unsupervised track** — Isolation Forest and Local Outlier Factor, trained
   without ever seeing the labels. Labels are used only for evaluation. The
   methodologically honest framing for the real-world fraud problem, and the
   section that carries most of the "understanding of data" marks.
2. **Supervised track** — a LightGBM gradient-boosted classifier trained on
   per-cardholder **velocity features** (time since last transaction,
   distance from previous merchant, rolling amount stats, etc.) with a
   chronological 2019-train / 2020-test split. This is the track that gets
   close to leaderboard-level scores.

**[notebooks/anomaly_detection.ipynb](notebooks/anomaly_detection.ipynb)** — the
deliverable. Every decision has a markdown cell explaining *why*.

## Headline results (from the executed notebook)

| Model | PR-AUC | ROC-AUC | Fraud F1 | Top-100 precision |
|---|---|---|---|---|
| Isolation Forest (all features) | 0.014 | 0.659 | 0.04 | 6 % |
| Isolation Forest (continuous features only — ablation) | 0.027 | 0.840 | 0.03 | 3 % |
| Local Outlier Factor | 0.105 | 0.865 | 0.15 | 28 % |
| Logistic Regression (simple supervised baseline) | 0.182 | 0.860 | 0.03 | 39 % |
| **LightGBM (supervised, velocity features, chronological split)** | **0.965** | **0.999** | **0.918** | **100 %** |

Key insights documented in the notebook:

* Isolation Forest's ROC-AUC more than doubled (0.66 → 0.84) once we removed
  the sparse one-hot dummies — a real experimental observation about why
  tree-based anomaly detection and one-hot encoding don't mix.
* LOF dominated IF as an unsupervised detector (PR-AUC 18× the base rate)
  because fraud geometrically forms sparse local clusters.
* The single most important finding: **fraud lives in the sequence, not in
  the single transaction.** Feature importance shows velocity features
  (`amt_ratio_to_roll_mean`, `distance_from_prev_km`, `velocity_kmh`,
  `log_time_since_last`) dominate the LightGBM gain ranking.
* Transactions flagged by both Isolation Forest and LOF have 100 % precision
  on the test set — a zero-false-positive analyst signal even without the
  supervised model.

## What the notebook does, section by section

1. **EDA** — shape, dtypes, missing values, duplicates, class imbalance, `amt`
   distributions per class, fraud rate by category and hour, correlation
   matrix, and verification that the planted noise columns have near-zero
   correlation with the target.
2. **Cleaning** — per-column justification for every drop, median imputation,
   log-transform of `amt`, engineered `age`, `hour`, `day_of_week`,
   `distance_home_merchant_km`, RobustScaler for outlier resistance.
3. **Unsupervised framing justification** — why anomaly detection, not
   supervised classification, for a real-world fraud setting.
4. **Unsupervised models** — Isolation Forest, Isolation Forest ablation
   (continuous features only), Local Outlier Factor with `novelty=True`,
   Logistic Regression baseline.
5. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC, PR curves, ROC
   curves, confusion matrices, top-K precision. Never accuracy.
6. **Model agreement analysis** — disagreements between IF and LOF and the
   perfect-precision intersection channel.
7. **Supervised track** — per-cardholder velocity feature engineering,
   chronological train/test split, LightGBM with `scale_pos_weight` and
   early stopping, full evaluation plus feature importance, and three
   explicit operating points (precision ≥ 0.90, max F1, recall ≥ 0.90).
8. **Final three-tier deployment recommendation** — LightGBM as the blocker,
   LOF as the watchdog for novel fraud patterns not yet in labels, and an
   intersection channel for auto-block.

## Reproducing the run

Requires macOS or Linux with Python 3.10+ and Homebrew (macOS).

### One-time system installs (two small libraries)

```bash
brew install unar     # RAR extractor for the dataset
brew install libomp   # OpenMP runtime required by LightGBM on macOS
```

### Project-local Python environment

```bash
cd csimodel
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

All Python dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`seaborn`, `jupyterlab`, `ipykernel`, `lightgbm`) live in `.venv/` — nothing
goes to the global Python.

### Place the dataset

The dataset (`dataset.rar`, ~87 MB) is **not committed to this repository** — it
is the file supplied with the assignment brief. Drop it at the project root
before running the notebook:

```
csimodel/
├── dataset.rar   <-- place here
└── ...
```

### Extract dataset and run the notebook

```bash
mkdir -p data && unar -o data dataset.rar

# Interactive
jupyter lab notebooks/anomaly_detection.ipynb

# Headless (executes the whole notebook in place)
jupyter nbconvert --to notebook --execute --inplace notebooks/anomaly_detection.ipynb
```

End-to-end runtime: about one minute on a recent Apple Silicon laptop.

## Layout

```
csimodel/
├── dataset.rar                    raw dataset (not in repo - gitignored; see above)
├── data/                          extracted dataset (gitignored)
├── .venv/                         project-local Python env (gitignored)
├── requirements.txt               pinned dependencies
├── build_notebook.py              programmatic notebook builder
├── notebooks/
│   └── anomaly_detection.ipynb    full analysis + reasoning + outputs
├── README.md                      this file
└── .gitignore
```

## Regenerating the notebook from source

The notebook is authored programmatically by
[build_notebook.py](build_notebook.py), which keeps it reviewable as plain
Python source rather than diffing giant JSON blobs. To rebuild and re-run:

```bash
source .venv/bin/activate
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/anomaly_detection.ipynb
```
