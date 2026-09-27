# Slide notes: cases and evidence by step

Running log kept while building the pipeline. Each case = problem → what we did → evidence → figure.
Used later to build the slide deck and the report.

---

## Framing (title / problem slides)

- **Task:** predict whether a component has an **unscheduled removal within the next 30 flight cycles**.
- **Constraint = one metric:** recall ≥ 80% **and** ≤ 5 alerts per 100 active components → *Recall @ 5% alert rate*.
- **Why it is hard:** 2% positives → the top 5% of scores must hold 80% of removals → ~16x lift over random,
  and the precision floor is 0.8 × 2 / 5 = **32%**.
- **Alert definition:** at any scoring run, ≤ 5% of active components are flagged. Independent of scoring cadence.

---

## Step 1: Mock data

**Case 1.1: No real data → build a simulator, not random tables**
- Problem: real MRO data is confidential. Random tables would give a model nothing to learn and nothing to clean.
- Did: day-by-day fleet simulation. 40 aircraft (A320neo / A321neo / B737-800), 6 bases in 3 climate zones,
  13 part numbers across 8 ATA chapters, 1,160 positions, 2 years.
- Mechanisms: Weibull wear-out per part number, climate and aircraft-age effects, latent rogue units,
  precursor drift before most failures, 8% sudden failures + NFF removals (caps achievable recall),
  rotables moving between tails via a shop/spares pool.
- Evidence: 9 CSV tables, 730k sensor rows, 2,767 unscheduled removals, positive rate **1.97%**.

**Case 1.2: Data is deliberately dirty so data preparation can be shown**
- ~7.7% missing sensor values (three mechanisms: random NaN, multi-day unit dropouts, whole-aircraft outages).
- Sentinel glitch codes 0 / −999 / 9999 (~750 per channel) + multiplicative spikes.
- Nuisance fault codes unrelated to health; NFF removals.

**Case 1.3: ACMS realism: sensor and fault data are keyed by tail + position, not serial**
- Forces the join through `installations` to know which serial produced a reading → sets up Case 4.1.

**Case 1.4: Planted leakage traps**
- `shop_finding` / `removal_reason` exist only after removal. 82% of serials move between tails.

---

## Step 2: EDA  (figures in `reports/figures/eda_*.png`)

**Case 2.1: Failure rates vary by system and climate** → `eda_removal_rates.png`
- ATA 36 pneumatic 1.07 and APU 0.97 per 1,000 FC vs landing gear 0.32. Hot/sandy bases +25% vs temperate.
- → part number, ATA, climate as features.

**Case 2.2: The precursor is real but late** → `eda_precursor_sensor.png`, `eda_precursor_faults.png`
- Primary sensor starts drifting ~90 FC before removal, passes 1σ at ~35 FC, reaches 2–3.5σ inside the 30 FC window.
- Related fault codes reach **19x** the fleet average in the final week.
- → features = recent change vs the unit's *own* baseline, short windows.

**Case 2.3: Outliers must not be clipped blindly** → `eda_outliers_example.png`
- Degradation itself produces extreme values. Remove only impossible values (sentinels) and single-point spikes.

**Case 2.4: Age alone is a weak predictor** → `eda_life_used.png`
- 47% of unscheduled removals happen before 50% of MTBUR; only 11% after MTBUR.
- → an age-based rule (CSI / MTBUR) is the baseline ML has to beat.

---

## Step 3: Snapshot table and label

**Case 3.1: Unit of prediction and label definition**
- Row = serial × snapshot day. Label = 1 if the installation ends in an unscheduled removal (failure or NFF)
  within the next 30 FC of its aircraft. Scheduled removals = 0.
- NFF counted as positive: the aircraft still took an unplanned removal (cost, delay).

**Case 3.2: Snapshot cadence caps recall (strong slide)**
| Cadence | Rows | Removals with ≥ 1 positive snapshot |
|---|---|---|
| Weekly | 116k | **80.5%** |
| Every 3 days | 270k | 100% |
| Daily | 808k | 100% |
- 30 FC ≈ 6 days. With weekly scoring, 1 in 5 removals falls between two scoring runs → even a perfect model
  would sit at ~80% event recall. → train on 3-day snapshots; recommend **daily scoring** in production.

**Case 3.3: Right censoring**
- Last snapshots with < 30 FC of follow-up have unknown labels → dropped (positives and negatives alike, so the
  end of the period is not biased toward positives).

**Case 3.4: Mild prior shift over time**
- Positive rate 2.55% (2024 Q1) → ~1.9% (2025). → pick the threshold by alert rate on validation, not a fixed
  probability; calibrate on validation.

---

## Step 4: Features

**Case 4.1: Attribute ACMS data to the right serial**
- Readings/faults are per position. Attribute to the serial installed that day; skip the installation-day reading
  (it may belong to either unit). Otherwise the new unit inherits the failed unit's drift → false alerts.

**Case 4.2: Features relative to the unit's own baseline**
- `z_vs_ref` = mean of last 7 days vs the unit's own days −60…−15. Removes part-to-part and unit-to-unit offsets.
- `primary_z_signed`: the part's health sensor signed in the degradation direction (pressure drops, temp/vibration
  rise) = engineering knowledge encoded as a feature.
- Top univariate AUC (2024): `primary_z_signed` 0.82, `primary_slope_signed` 0.79, `days_since_related_fault` 0.79.
  No feature near 1.0.

**Case 4.3: Missing data kept as information**
- Young installations have no reference window → `z_vs_ref` NaN (12.6%). Kept NaN (trees handle it) and
  flagged with `low_history`, `sensor_missing_ratio_7d`, `days_since_last_reading`.

**Case 4.4: Point-in-time test caught a real leak (strong slide)**
- Test: rebuild all 47 features on raw data truncated at day *t* and compare with the full-history version.
- Found: CSI back-fill for legacy installations used average utilisation over the **whole** two years
  → `csi`, `csn`, `cso`, `csi_over_mtbur`, `csn_over_mtbur` changed when the future was removed.
- Fixed (average from the first quarter only). Test passes at 3 cut dates.
- Message: leakage prevention is verified by a test, not asserted.

---

## Step 5: Leakage-safe split  → `split_timeline.png` (key slide)

**Case 5.1: Three layers of leakage protection**
| Layer | Risk | What we did |
|---|---|---|
| Temporal | adjacent snapshots are near-duplicates; a label window can reach into the next period | consecutive periods: train Jan 2024–Feb 2025, valid Mar–Jun 2025, test Jul–Dec 2025. **Purge** rows whose 30 FC window crosses the boundary + **7-day embargo** |
| Aircraft | tail-specific behaviour memorised | **8 held-out tails** (20%) never used in train/valid |
| Component | rotables move between tails, so a tail split alone still leaks the unit's history | drop all train/valid rows of any serial that is tested on a held-out tail |

**Case 5.2: Purge is measured in flight cycles, not days**
- The label window is 30 FC, so the purge is too. Tails in a heavy check fly 0 cycles/day → their purge gap
  is longer (visible as uneven grey edges in the timeline figure).

**Case 5.3: Filter the training side, not the test side (design trade-off)**
- First attempt: keep only test rows whose serial was never seen in training → test shrank to **39 positives**,
  and it was biased toward brand-new, low-risk units.
- Final: remove the *training* history of serials that appear in the test set → train loses ~18% of rows,
  test keeps **263 positives**. Same guarantee (the model never saw a tested unit), usable test size.

**Case 5.4: Held-out tails stratified by aircraft type AND climate**
- First random draw: 5 of 8 held-out tails at temperate bases (lowest failure rate) → flattering test.
- Final: quota sampling so held-out mix matches the fleet: 3 A320 / 3 A321 / 2 B737 and
  3 hot-sandy / 3 temperate / 2 hot-humid.

**Case 5.5: Two test sets, two questions**
| Set | Rows | Positives | Question it answers |
|---|---|---|---|
| train | 97,993 | 2,012 (2.05%) | |
| valid | 29,344 | 492 (1.68%) | threshold + calibration + model selection |
| **test_strict** | 13,253 | 263 (1.98%) | does it generalise to unseen aircraft and unseen units? (headline) |
| test_fleet | 66,642 | 1,264 (1.90%) | how does it perform in production on the fleet it knows? |
- Hyper-parameter tuning inside train uses **GroupKFold by tail** (4 folds × 8 tails).
- Automatic assertions: no held-out tail or tested serial in train/valid; periods ordered; embargo respected.

---

## Step 6: Data preparation  (brief criterion: missing data, outliers, categorical, class imbalance)

**Case 6.1: One table that answers the data-preparation criterion (key slide)**
| Issue | Linear branch (Logistic Regression) | Tree branch (RF, HistGB, LightGBM, XGBoost) | Why |
|---|---|---|---|
| Missing (17 of 42 features, max 13%) | median **by part number** from train + `_missing` flags | keep NaN, learned natively | a pump's typical value ≠ a valve's; the flag tells the model the value was guessed |
| Outliers | sentinels + spikes removed at feature time; then winsorise at train p0.1/p99.9, log1p on counts | sentinels + spikes removed at feature time; nothing else (split-based) | top of the drift distribution *is* the failing units → tame only the extreme tail |
| Categorical (5 features) | one-hot (30 cols), unseen → all zeros | ordinal codes, unseen → NaN | low cardinality here; in a real fleet with thousands of P/Ns use target encoding (out-of-fold) |
| Imbalance (2,012 pos vs 95,981 neg) | class weight, tuned in {1, ≈7} | `scale_pos_weight`, tuned in {1, ≈7} | see Cases 6.2 and 6.5 |
- Every statistic is fitted on **train only** and lives inside the model Pipeline → same transformation at scoring time.

**Case 6.2: Why class weights, not SMOTE or undersampling**
- SMOTE interpolates between snapshots of different units → invents sensor histories that never happened.
- Undersampling throws away healthy-unit variety (parts, climates, ages).
- Weights keep every real row; only the cost of missing a removal changes.
- Validation/test keep the natural ~2% rate → recall and alert rate are measured honestly.
- **Updated after the experiment in Case 6.5:** full weighting (×48) brought no recall gain and broke calibration.
  Final choice: positive weight is a tuned hyper-parameter in {1, √(neg/pos) ≈ 7}, never the full ratio.

**Case 6.3: New components and missing data at scoring time (brief requirement)**
- New part number never seen in training → linear: falls back to global median, one-hot all zeros;
  trees: category = NaN. Pipeline still returns a score (tested).
- Unit with no sensor data at all → imputed / NaN + missing flags → still scored, and the flags feed the
  data-quality indicator shown in the API response.
- Young installation (< 30 days) → `low_history` = 1, no own-baseline features yet → falls back on
  part-number baselines, fault codes and serial history.

**Case 6.4: Guarantees are tested** (`tests/test_preprocess.py`, 5 tests)
- imputer medians equal train medians and do not move after transforming test data
- unseen part number + unseen aircraft type + all sensors missing → finite output
- linear branch has no NaN; tree branch keeps NaN
- winsor limits fixed after fit; class weight equals the label ratio

**Case 6.5: Imbalance ablation: what does each strategy change? (key slide)** → `imbalance_experiment.png`
- Setup: 3 models (Logistic Regression, LightGBM, XGBoost) × 5 strategies, same features and settings, validation
  set at its natural 1.68% positive rate, 3 seeds for the stochastic runs. Test sets untouched.
  Script: `src/imbalance_experiment.py`, results: `reports/results/imbalance_experiment.csv`.

| Strategy | LGBM recall @ 5% | LGBM event recall @ 5% | LGBM PR-AUC | Alert rate @ p ≥ 0.5 (LR / LGBM / XGB) | Mean predicted prob (LR / LGBM / XGB), actual 1.7% |
|---|---|---|---|---|---|
| none | 71.1% | 83.1% | 0.615 | 1.1% / 1.2% / 1.2% | 2.3% / 2.0% / 2.1% |
| weight ×7 (√ratio) | 71.3% | 83.4% | 0.613 | 2.1% / 1.9% / 1.7% | 8.2% / 4.8% / 5.2% |
| weight ×48 (full) | 70.2% | 82.2% | 0.610 | **14.7%** / 4.5% / 4.4% | **28.4%** / 11.4% / 13.8% |
| undersample 1:10 | 70.7% | 82.7% | 0.605 | 1.7% / 1.9% / 1.8% | 6.4% / 4.8% / 5.3% |
| SMOTE 1:10 | 70.6% | 83.1% | 0.610 | 1.7% / 1.2% / 1.3% | 6.1% / 2.8% / 3.0% |

Findings (say these on the slide):
1. **Ranking barely moves.** Recall @ 5% budget stays within ±1 point across all 5 strategies for every model
   (LR 68–69%, LGBM 70–71%, XGB 70–71%). Seed-to-seed noise is about the same size.
   Imbalance handling does not create signal; the features do.
2. **What it really changes is the probability scale.** Full weighting inflates the mean predicted probability
   from 2% to 11–28% (actual 1.7%). With a fixed 0.5 threshold the alert rate jumps from 1% to **14.7%** for LR,
   **3x over the 5% budget**, while the same model at the budget catches the same removals.
3. **So the threshold matters more than the resampling.** If the threshold is set by the alert budget (Step 8),
   the strategy choice is almost irrelevant for recall. If a default 0.5 is used, the strategy decides whether the
   hangar is flooded with alerts (weight ×48) or misses ~45% of removals (none).
4. **Full weighting is the worst option here**: slightly lower PR-AUC for every model, worst calibration, and the
   Brier score is 4–12x worse. SMOTE adds cost (slowest fit) and synthetic histories for no gain.
5. **Decision:** keep all real rows, no resampling. Positive weight tuned in {1, ≈7} by grouped CV;
   probabilities calibrated on validation; threshold from the alert budget.

Also visible here (carry to Step 7/8): on validation, event-level recall is already ~83% (above the 80% target)
but row-level recall is ~71%. The brief says "detects 80% of actual removals" → the event is the unit that
matters, but Step 7 should try to lift both.

---

## Step 7: Modelling

**Case 7.1: Bootstrap: used for confidence, never for splitting**
- Not for the split: resampling rows with replacement puts near-identical consecutive snapshots of the same unit
  in both train and test = exactly the leakage Step 5 prevents.
- Used inside Random Forest (bagging) and for **confidence intervals**: cluster bootstrap that resamples whole
  serial numbers (rows of one unit are correlated), alerts held fixed as in operation, 1,000 resamples.
- Paired: the same resamples for every model → "is model A really better than B, or is it noise?"
  (`p_best_better` = share of resamples where the champion beats the other model).

**Case 7.2: First comparison: the model family is not the lever** (fast run, validation)
- 5 ML models all at 69–71% row recall @ 5%, overlapping 95% CIs; event recall 80–84%.
- Rules far behind: repeat-fault rule 57% row / 71% event; age rule 13% / 14%.
- → invest in features, not in more model types.

**Case 7.3: Feature iteration driven by the EDA** (`src/feature_iteration.py`, `reports/results/feature_iteration.csv`)
- EDA showed the drift *accelerates* near removal. v1 features only had a 7-day mean and a 14-day slope.
- Added: 3-day mean vs own baseline (reacts to the steep end), 30-day slope (catches the slow start),
  acceleration = 14-day slope − 30-day slope. Point-in-time test re-run: passes.
- Same LightGBM settings, grouped CV on train + validation:
| Feature set | CV recall @ 5% | CV event recall | CV PR-AUC | Valid recall @ 5% | Valid event recall | Valid PR-AUC |
|---|---|---|---|---|---|---|
| v1 (42 numeric) | 69.6% | 79.7% | 0.612 | 70.9% | 83.5% | 0.614 |
| v2 (48 numeric) | **70.8%** | **80.6%** | **0.627** | **72.4%** | **84.7%** | **0.632** |
- Small but consistent gain on every metric, on CV and on validation → kept.

**Case 7.4: Full tuning: 5 models, random search, GroupKFold by tail** → `model_comparison.png`
| Model | CV recall @ 5% (train) | Valid recall @ 5% | Valid event recall | Valid PR-AUC | P(top valid model better) |
|---|---|---|---|---|---|
| HistGradientBoosting | 70.8% | 72.4% | 84.7% | 0.620 |  |
| XGBoost ★ | 71.2% | 72.4% | 84.3% | 0.627 | 44% |
| LightGBM | 71.0% | 72.0% | 83.9% | 0.626 | 70% |
| RandomForest | 70.2% | 71.1% | 83.1% | 0.608 | 97% |
| LogisticRegression | 69.5% | 70.5% | 83.5% | 0.600 | 97% |
| Rule: repeat faults |  | 56.5% | 70.6% | 0.401 | 100% |
| Rule: age (CSI/MTBUR) |  | 12.8% | 14.1% | 0.033 | 100% |
- Tuning picked **no class weight** for 4 of 5 models (LightGBM picked ×7) → consistent with the ablation.
- **Champion rule (fixed in code):** among models statistically tied with the top validation model (it beats them
  in < 90% of paired resamples: HistGB, XGBoost, LightGBM), take the best grouped-CV recall → **XGBoost**.
  CV is independent of validation, which is reused for calibration and the threshold.
- Right panel: event recall plateaus at ~85–88% even at 15% alerts → ~12% of removals have no usable precursor
  (sudden failures, NFF). More alerts cannot fix that; better data could.

**Case 7.5: Majority voting / ensembles** (`src/ensemble.py`, `reports/results/ensemble_experiment.csv`)
- Rule fixed before the run: adopt an ensemble only if it beats the single model in ≥ 90% of paired resamples.
| Candidate | Event recall | Δ vs XGBoost | P(better) |
|---|---|---|---|
| single XGBoost | 84.3% | – | – |
| hard majority vote (5) | 83.9% | −0.4 pt | 20% |
| soft vote top 3 | 83.9% | −0.4 pt | 24% |
| rank vote top 3 | 85.1% | +0.8 pt | 88% |
| rank vote all 5 | 83.9% | −0.4 pt | 20% |
- Why so little gain: boosting models' scores correlate 0.91–0.93 (Spearman) → they miss the same removals.
- Hard voting: ~37 of 40 alert slots per run are unanimous; the remaining 3 slots are decided among ~7 tied units.
- **Decision: single XGBoost.** Best ensemble +0.8 pt is not reliable (88% < 90%) and would cost SHAP clarity,
  3x models to monitor, slower scoring.

---

## Step 8: Evaluation and threshold  → `eval_recall_vs_alert_rate.png`, `eval_value_curve.png`

**Case 8.1: Threshold chosen by MRO value, under the hard 5% cap (validation)**
- Value model (illustrative, stated): **$30,000** saved per anticipated removal (avoided AOG/delay/expedited spare),
  **$600** per inspection episode (consecutive alerts on one unit = one inspection).
- Scan thresholds on validation, always capping at 5% of active components per run → value peaks at a mean alert
  rate of **3.7%** (event recall 83.9%), then declines: extra alerts beyond that mostly add inspections, not catches.
- Insight for the business: the 5% capacity is **not binding**; the model needs ~3–4 alerts per 100 components.
- Tiers: **HIGH** = top ~1.5% per run (inspect within 5 FC, pre-position spare), **MEDIUM** = rest of alerts.
- Calibrated probability (isotonic on validation) shown to engineers; alerts decided on the raw score
  (isotonic output is a step function → ties).

**Case 8.2: Test results: evaluated once, never tuned on** (key slide)
| | test_strict (unseen tails + units) | test_fleet (all tails) |
|---|---|---|
| Removal events | 124 | 605 |
| **Event recall** (brief: ≥ 80%) | **83.1%** (95% CI 76.2%–89.7%) | **82.8%** (79.8%–85.9%) |
| Mean / max alert rate per run (brief: ≤ 5%) | 3.1% / **4.7%** | 3.1% / **4.6%** |
| Row recall | 73.8% | 73.4% |
| Precision (alerts that are real) | 47.1% | 44.4% |
| HIGH-tier precision | 74.3% | 75.6% |
| Median lead time (first alert → removal) | 22 FC | 21 FC |
| PR-AUC / ROC-AUC | 0.685 / 0.932 | 0.674 / 0.922 |
| Net value (illustrative) | $3.0M | $14.3M (6 months, 40 aircraft) |
- **Both targets met on both test sets.** No drop from validation to unseen aircraft → no sign of tail/unit memorisation.
- **Honest caveat:** test_strict has only 124 removals; its 95% CI lower bound (76.2%) is below 80%.
  The point estimate meets the target; certainty needs more data → shadow-mode period before go-live.
- Precision ~45% vs the 32% floor implied by the brief; HIGH tier ~75% → engineers can trust HIGH alerts.

**Case 8.3: Where the model fails** → `eval_segments.png`, `eval_lead_time.png`
- Confirmed failures: 86% caught. **NFF removals: 9%** (23 events): NFF has no physical degradation to detect,
  by definition. Counting NFF as positive (Case 3.1) costs ~3 pt of headline recall; worth stating.
- By ATA: weakest ATA 34 air data (76%) and ATA 28 fuel pumps (79%); strongest ATA 49 APU (94%).
- Temperate bases 78% vs hot/sandy 86% (fewer precursor signals where stress is lower).
- Lead time: median 21 FC (~4 days); 95% of catches have the first alert ≥ 7 FC before removal (98% ≥ 5 FC) → enough to pre-position a spare.
- Calibration (test_fleet): close to the diagonal; the top decile is slightly under-predicted (13% vs 15%).

---

## Step 9: Explainability  → `shap_global.png`, `shap_examples.png`

**Case 9.1: Global drivers (mean |SHAP|, XGBoost)**
- 1 `primary_z3_signed` (health sensor, 3-day vs own baseline) · 2 `days_since_related_fault` · 3 `caution_faults_7d`
  · 4 `csi_over_mtbur` · 5 `primary_slope30_signed`. Direction plots match engineering sense (drift up → risk up;
  recent faults → risk up; older unit → risk up). The new v2 feature is #1.

**Case 9.2: Reasons in maintenance language (what the engineer sees)**
- Caught (ACM, ATA 21): "Vibration 5.2σ above this unit's own baseline over the last 3 days · Last related fault
  message 0 days ago · 4 related fault messages (ATA 21) in 30 days" → removed 4 FC later.
- Caught (bleed PRV, ATA 36): "3 CAUTION/WARNING-level related faults in 7 days · Pressure 1.9σ below own baseline"
  → removed 11 FC later.
- **False alert (PCV, ATA 36): temperature 13σ above baseline, but 43% of sensor data missing and no fault message
  in a year** → looks like a *sensor* problem, not a component problem. → Production rule: an extreme sensor-only
  alert with poor data quality is routed to avionics/sensor check first.
- **Missed (EHP, ATA 29): 2 FC before removal the sensor was stable (0.3σ) and no related fault for 142 days** →
  sudden failure with no precursor. No model on this data could catch it.

---

## Step 10: Batch scoring (daily production job)

**Case 10.1: Scoring as of any date uses only what was known that day**
- `python -m src.predict --date 2025-10-15` truncates every raw table at the date, scores all 1,160 active units,
  writes a CSV with risk, level, action and top-3 reasons. ~7 s for the fleet.
- Example run (2025-10-15): 11 HIGH, 16 MEDIUM → 2.3% alerted. Top: fuel boost pump AC038 R1, "pressure 2.8σ below
  own baseline over 3 days · 2 CAUTION/WARNING faults in 7 days · 3 related ATA 28 faults in 30 days".
- **Live = offline check:** on a test date, live scores (truncated data) vs evaluation scores: 1,160 / 1,160 units
  matched, max score difference 0.0, identical alerts → the test numbers are what production would produce.

## Step 11: API  (`api/main.py`, samples in `docs/sample_responses/`)

**Case 11.1: Three endpoints, JSON built for the MRO workflow**
- `GET /health`: model version, scorer, thresholds (flag ≥ 2.1% risk, HIGH ≥ 21.7%), 5% cap, train period.
- `GET /fleet/risk?date=&level=`: ranked alerts for the planning/MCC view (daily job output).
- `POST /predict`: by serial, or by tail + position (how line maintenance identifies a unit).
- Every component carries: calibrated `risk_score`, `risk_level`, `recommended_action`, `top_factors`
  (feature, value, SHAP impact, plain-language reason) and `data_quality` (`low_history`, missing-sensor ratio,
  days since last reading → `confidence: normal | reduced`).
- New component example (installed < 30 days): scored, flagged `low_history: true`, `confidence: "reduced"`.
- Explicit errors: 404 unit not installed on that date, 422 bad request / date outside the data window.
- `tests/test_api.py`: alerts ≤ 5% per run, every alert explained, serial and position lookups agree.
