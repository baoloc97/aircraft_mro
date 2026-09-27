# Unscheduled Component Removal Prediction (MRO PoC)

Predicts whether an aircraft component will have an **unscheduled removal within the next 30 flight cycles**,
under the operational constraint of **≥ 80% of removals detected with ≤ 5 alerts per 100 active components**.

**Result (XGBoost, held-out aircraft and units never seen in training):** 83.1% of removals detected
(95% CI 76–90%), at most 4.7 alerts per 100 components in any scoring run (mean 3.1), median warning 21 flight
cycles before removal. Two rule baselines reach 71% and 14% at the same alert budget.

All data is simulated. The numbers show the method works end to end, not the performance to expect on real
fleet data. Full write-up: [`reports/report.md`](reports/report.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp                  # macOS only: LightGBM / XGBoost need the OpenMP runtime

# Data: download the 9 CSV files from the OneDrive folder (https://1drv.ms/f/c/da4fc7aee9f85e92/IgAhczYtPwdoT7cqYtK6JjbuAe6e1tPrppsmppOnedM8PJc?e=c02Brs) into data/raw/,
# or regenerate the identical files (seeded):
python -m src.generate_data

python -m src.build_dataset && python -m src.features && python -m src.split
python -m pytest tests/ -q           # 15 tests: point-in-time features, preprocessing, reasons, API

uvicorn api.main:app --reload        # uses the committed models/risk_scorer.joblib; docs at /docs
```

Full rerun of every step, including model tuning, takes about 20 minutes (see the steps below).

## Project structure

```
data/raw/                 mock source tables (CSV, not in git: OneDrive or src.generate_data)
data/processed/           snapshot table, features, split (not in git: rebuilt by the pipeline)
src/                      pipeline, one module per step
  generate_data.py        1  fleet simulator
  build_dataset.py        3  snapshots and 30 FC label
  features.py             4  point-in-time features (+ raw_as_of: the data known at a given day)
  split.py                5  temporal / aircraft / component split
  preprocess.py           6  missing data, outliers, categorical, class weights
  imbalance_experiment.py 6b imbalance ablation
  models.py               7  model registry (one entry per candidate model) + rule baselines
  train.py                7  tuning and comparison
  ensemble.py             7b voting / ensemble test
  evaluate.py             8  threshold selection and test evaluation
  scoring.py                 RiskScorer: model + calibration + thresholds, used by every consumer
  explain.py              9  SHAP
  reasons.py              9  plain-language reason per feature (table-driven)
  predict.py              10 daily batch scoring
  metrics.py, viz.py, config.py
api/                      11 FastAPI service + sample export
tests/                    leakage, preprocessing, reason and API tests
notebooks/01_eda.ipynb    2  EDA with outputs
models/                   risk_scorer.joblib (final, in git); tuned candidates (not in git)
reports/                  report.md, figures/, results/ (summary tables cited in the report)
artifacts/                logs and intermediate files (not in git: regenerated)
docs/                     sample API responses, slide notes
```

## Code design

- **One responsibility per module**: each pipeline step is a module with a `main()`; shared concerns live in
  `config` (paths, constants), `metrics` (evaluation), `scoring` (the production scorer) and `reasons`
  (explanation text).
- **Open for extension**: models are declared once in `models.MODEL_REGISTRY` (preprocessor, estimator factory,
  search space); reason sentences in `reasons.EXACT` / `SUFFIX_RULES`. Adding a model or a feature explanation
  is one new entry; training, the ablation, the scorer and the API do not change.
- **One code path for training and serving**: the same `build_features` and `raw_as_of` are used by the
  evaluation, the batch job and the API, and a test checks that live and offline scores match.
- **No secrets or personal data**: the dataset is synthetic; nothing in the repository needs credentials.

## Step 1 - Mock dataset

```bash
python -m src.generate_data     # ~5 s, writes 9 CSV files (~36 MB) to data/raw/
```

**Or download them:** the OneDrive folder [`raw`](https://1drv.ms/f/c/da4fc7aee9f85e92/IgAhczYtPwdoT7cqYtK6JjbuAe6e1tPrppsmppOnedM8PJc?e=c02Brs) is this repository's `data/raw/` folder.
Put its 9 CSV files in `data/raw/` under their original names (the code reads `data/raw/<table>.csv`, see
`RAW_DIR` in `src/config.py`):

```
data/raw/
  aircraft.csv  parts_catalog.csv  components.csv  installations.csv  flight_usage.csv
  sensor_readings.csv  fault_codes.csv  maintenance_history.csv  removals.csv
```

The generator is seeded, so the downloaded files and the regenerated ones are byte-identical.

A day-by-day fleet simulation (40 aircraft, 13 part numbers, 1,160 installed positions, 2024-01-01 to 2025-12-31).

| Table | Grain | Notes |
|---|---|---|
| `aircraft` | tail | type, manufacture year, home base, climate zone |
| `parts_catalog` | part number | ATA chapter, MTBUR, primary health sensor, soft-time limit |
| `components` | serial number | CSN/TSN at start, prior shop visits |
| `installations` | install event | which serial sat in which tail/position and when |
| `flight_usage` | tail × day | flight cycles, flight hours, OAT, heavy-check flag |
| `sensor_readings` | tail × position × day | vibration / temperature / pressure (keyed by position, not serial) |
| `fault_codes` | fault message | CMS-style codes, related and nuisance |
| `maintenance_history` | event | installation, removal, troubleshooting, MEL deferral |
| `removals` | removal | type (scheduled/unscheduled), reason, shop finding |

Built-in realism: Weibull wear-out per part number, climate and aircraft-age effects, latent rogue units,
precursor drift before most failures, sudden failures and NFF removals with no clear precursor,
rotables moving between tails, ~7.7% missing sensor values, spikes and sensor glitches (0, -999, 9999).
Resulting positive rate: **~2%** of component-days.

## Step 2 - Exploratory data analysis

```bash
python notebooks/build_eda.py   # rebuilds and executes notebooks/01_eda.ipynb (~15 s)
```

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) ships with outputs. Each finding ends in a pipeline decision:
removal rates by ATA / climate, three missing-data mechanisms, sentinel glitch codes and spikes, the precursor
signal (sensor drift from ~90 FC, strongly abnormal inside the 30 FC window; related faults ~19x baseline in the
final week), life-used distribution, and the leakage traps. Figures are saved to `reports/figures/`.

## Step 3 - Snapshot table and label

```bash
python -m src.build_dataset     # writes data/processed/snapshots.parquet
```

- **Row** = one installed serial number at the end of a snapshot day (every 3 days, ~1,160 active units per snapshot).
- **Label** = 1 if that installation ends in an **unscheduled** removal (confirmed failure or NFF) within the next
  **30 flight cycles** of its aircraft. Scheduled removals are negatives.
- **Censoring**: snapshots with < 30 FC of follow-up before the end of the data are dropped (label unknown).
- **Cadence choice**: with weekly snapshots, 19.5% of removals have no positive snapshot at all, because 30 FC is
  only ~6 days. Even a perfect model would be capped near 80% event recall. Every 3 days gives 100% coverage.

Result: 269,642 rows, 5,305 positives (**1.97%**).

## Step 4 - Point-in-time features

```bash
python -m src.features              # writes data/processed/features.parquet
python -m pytest tests/ -q          # leakage test
```

42 numeric + 5 categorical features ([`src/features.py`](src/features.py)):

| Group | Examples |
|---|---|
| Usage / age | `csi`, `csn`, `cso`, `csi_over_mtbur`, `fc_per_day_30d`, `low_history` |
| Sensor | 7-day mean, `*_z_vs_ref` (last 7 days vs the unit's own days −60…−15), 14-day slope, deviation from part baseline, `primary_z_signed` (the part's health sensor, signed in the degradation direction), `sensor_missing_ratio_7d`, `days_since_last_reading` |
| Fault codes | related faults over 3 / 7 / 30 days, week-over-week trend, caution+ count, nuisance count, days since last related fault |
| Maintenance | troubleshooting and MEL deferrals during this installation, prior unscheduled / NFF removals of the serial on any tail, shop visits |
| Context | part number, ATA chapter, primary sensor, aircraft type, climate zone, aircraft age, ambient temperature |

Point-in-time rules: ACMS records are attributed to the serial installed on that day (never to the previous
unit in the same position), windows only look backwards, part baselines and utilisation averages come from
the first quarter only. `tests/test_point_in_time.py` rebuilds features on raw data truncated at day *t*
and checks that every feature is identical to the full-history version. It caught one leak during development
(average utilisation over the full period used to back-fill CSI for legacy installations).

## Step 5 - Leakage-safe split

```bash
python -m src.split     # writes data/processed/model_table.parquet and reports/figures/split_timeline.png
```

| Set | Period | Rows | Positives |
|---|---|---|---|
| train | 2024-01-29 → 2025-02-22 | 97,993 | 2,012 (2.05%) |
| valid | 2025-03-09 → 2025-06-25 | 29,344 | 492 (1.68%) |
| test_strict | 2025-07-07 → 2025-12-25, 8 held-out tails | 13,253 | 263 (1.98%) |
| test_fleet | 2025-07-07 → 2025-12-25, all tails | 66,642 | 1,264 (1.90%) |

- **Temporal**: rows whose 30 FC label window crosses a period boundary are purged (measured in flight cycles),
  plus a 7-day embargo.
- **Aircraft**: 8 tails, stratified by aircraft type and climate zone, never appear in train/valid.
- **Component**: train/valid rows of any serial tested on a held-out tail are dropped, so the model never saw a
  unit it is tested on.
- Hyper-parameter tuning uses GroupKFold by tail inside the training period. Leakage assertions run on every build.

## Step 6 - Data preparation

[`src/preprocess.py`](src/preprocess.py) builds two sklearn preprocessors, fitted on train only and embedded in
each model pipeline:

| Issue | Linear branch | Tree branch |
|---|---|---|
| Missing data | train median by part number + missing flags | native NaN handling |
| Outliers | (sentinels/spikes removed in features) winsorise p0.1/p99.9, log1p counts | (sentinels/spikes removed in features) |
| Categorical | one-hot, unseen → zeros | ordinal, unseen → NaN |
| Imbalance | no resampling; class weight tuned in {1, ≈7} | `scale_pos_weight` tuned in {1, ≈7} |

`tests/test_preprocess.py` checks train-only fitting and that a new part number with no sensor data is still scored.

### Step 6b - Class-imbalance ablation

```bash
python -m src.imbalance_experiment     # ~1 min; reports/results/imbalance_experiment.csv + figure
```

3 models × 5 strategies (none, weight ×7, weight ×48, undersample 1:10, SMOTE 1:10) on the validation set.
Recall at the 5% alert budget moves by ±1 point at most, while full weighting inflates the mean predicted
probability from 2% to up to 28% and pushes the alert rate at a 0.5 threshold to 14.7%. The threshold choice
matters far more than the resampling choice. Decision: no resampling, positive weight tuned in {1, ≈7}, calibrate
on validation, threshold from the alert budget.

## Step 7 - Models

```bash
python -m src.feature_iteration   # v1 vs v2 features (grouped CV)
python -m src.train               # 2 rule baselines + 5 tuned models (~15 min)
python -m src.ensemble            # voting / ensemble test
```

Validation (alerts = top 5% per run, 95% cluster-bootstrap CI in `reports/results/model_comparison.csv`):

| Model | CV recall @5% | Valid recall @5% | Valid event recall | PR-AUC |
|---|---|---|---|---|
| **XGBoost (champion)** | 71.2% | 72.4% | 84.3% | 0.627 |
| HistGradientBoosting | 70.8% | 72.4% | 84.7% | 0.620 |
| LightGBM | 71.0% | 72.0% | 83.9% | 0.626 |
| Random Forest | 70.2% | 71.1% | 83.1% | 0.608 |
| Logistic Regression | 69.5% | 70.5% | 83.5% | 0.600 |
| Rule: repeat faults | - | 56.5% | 70.6% | 0.401 |
| Rule: age (CSI/MTBUR) | - | 12.8% | 14.1% | 0.033 |

Champion rule: among models statistically tied on validation, the best grouped-CV recall. Ensembles (hard, soft and
rank voting) did not beat XGBoost reliably (best: +0.8 pt, P = 0.88 < 0.90), so a single model is kept.

## Step 8 - Threshold and test results

```bash
python -m src.evaluate            # writes models/risk_scorer.joblib, reports/results/test_results.csv, figures
```

Threshold chosen on validation by an MRO value model under the hard 5% cap; test sets evaluated once.

| | test_strict (unseen tails + units) | test_fleet |
|---|---|---|
| **Event recall** (target ≥ 80%) | **83.1%** (95% CI 76.2–89.7%) | **82.8%** (79.8–85.9%) |
| Max alert rate per run (target ≤ 5%) | **4.7%** (mean 3.1%) | **4.6%** (mean 3.1%) |
| Precision / HIGH-tier precision | 47% / 74% | 44% / 76% |
| Median lead time | 22 FC | 21 FC |

## Step 9 - Explainability

```bash
python -m src.explain             # SHAP global + example cards (reports/results/example_explanations.json)
```

## Step 10 - Batch scoring

```bash
python -m src.predict --date 2025-10-15
```

## Step 11 - API

```bash
uvicorn api.main:app --reload     # http://127.0.0.1:8000/docs
python -m api.export_samples      # refresh docs/sample_responses/*.json
```

```bash
curl "localhost:8000/fleet/risk?date=2025-10-15&level=HIGH&limit=3"
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"as_of_date": "2025-10-15", "components": [{"serial_no": "FBP-00028"}]}'
```

Example responses: [`docs/sample_responses/`](docs/sample_responses/).
