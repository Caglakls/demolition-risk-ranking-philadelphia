# Demolition Risk Ranking, Philadelphia

A parcel-level machine learning model that ranks every building in Philadelphia by its relative risk of demolition in the following year, together with a citywide ranking for risk year 2025 (demolitions starting in 2026).

The ranking is built to feed a material flow analysis (MFA) of the future supply of building materials from demolition. It answers one question: *which buildings are most likely to come down next?* It does **not** predict how many will.

---

## Contents

- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Method](#method)
- [Model comparison](#model-comparison)
- [The 2025 citywide ranking](#the-2025-citywide-ranking)
- [How to interpret the scores](#how-to-interpret-the-scores)
- [Reproducing the results](#reproducing-the-results)
- [Data sources](#data-sources)
- [Limitations](#limitations)
- [License](#license)

---

## Key results

Primary model: **`combined_strict` XGBoost**. Results are out-of-sample from rolling-origin validation over six test years (2019–2024), covering 3,180,298 parcel-years and 2,460 demolitions.

| Metric | Value | Meaning |
|---|---|---|
| ROC-AUC | **0.943** | A demolished parcel outranks a surviving one 94.3% of the time |
| Top-10% capture | **84.0%** | The highest-risk 10% of parcels contains 84% of the demolitions that actually happened |
| Top-20% capture | 90.3% | Widening to 20% adds only 6 points: risk is concentrated |
| PR-AUC | 0.0697 | Against a base rate of 0.077%, i.e. **90× better than random** |
| Yearly ROC-AUC range | 0.922–0.961 | Stable across all six test years |

For 2025, all **535,000** Philadelphia parcels that have a building are scored and ranked citywide.

---

## Repository structure

```
demolition-risk-ranking-philadelphia/
│
├── machine learning model/
│   ├── model_demolition_philly.py          training + rolling-origin evaluation
│   ├── philly_panel_v2.parquet             training panel, 2015–2024 (see note on file sizes)
│   └── *.joblib                            14 fitted models (7 feature sets × 2 algorithms)
│
├── model outputs/
│   ├── rolling_origin_pooled_metrics.csv   performance per model, all test years pooled
│   ├── rolling_origin_yearly_metrics.csv   performance per model per test year
│   ├── rolling_origin_predictions.csv      every out-of-sample prediction (see note on file sizes)
│   ├── final_model_ranking_comparison.csv  logistic vs XGBoost ranking agreement
│   ├── final_logistic_standardized_coefficients.csv
│   ├── selected_model_summary.txt          model selection notes
│   ├── model_configuration.json            feature sets and hyperparameters
│   ├── data_quality_summary.csv            missingness and event/non-event means per feature
│   ├── events_by_risk_year.csv             rows, events and prevalence per year
│   ├── feature_availability_by_year.csv    % of rows populated per feature per year
│   └── shap_outputs/                       SHAP importance + beeswarm plots (4 XGBoost models)
│
├── risk ranking scripts/
│   ├── preprocess_07_scoring_table_2025.py builds the 2025 feature table
│   └── score_08_risk_ranking_2025.py       applies the frozen models, ranks citywide
│
└── risk ranking outputs/
    ├── risk_ranking_2025_citywide.csv      the final ranking, 535,000 parcels
    └── risk_ranking_2025_summary.txt       headline figures for the 2025 ranking
```

### Pipeline

```
philly_panel_v2.parquet ──► model_demolition_philly.py ──► fitted models (*.joblib)
   (2015–2024, labelled)        rolling-origin CV              + model outputs/
                                                                       │
raw city data ──► preprocess_07_scoring_table_2025.py                  │
   (cut at 31 Dec 2025)   └─► risk_year_2025_scoring_table ──► score_08_risk_ranking_2025.py
                                                                       │
                                                       risk_ranking_2025_citywide.csv
```

---

## Method

### Prediction target

Each row is a parcel in a **risk year** *t*. Features use only information available by **31 December of year *t***, and the target (`event_next_year`) is whether a demolition starts during year *t* + 1.

Only parcels with an assessed building (`improvement_value > 0`) are modelled, since vacant land cannot lose a building. Parcels whose demolition has already begun are removed from later years.

| Risk year | Parcels | Demolitions in *t*+1 | Prevalence |
|---|---:|---:|---:|
| 2015 | 523,201 | 726 | 0.139% |
| 2016 | 523,652 | 710 | 0.136% |
| 2017 | 524,631 | 624 | 0.119% |
| 2018 | 525,761 | 580 | 0.110% |
| 2019 | 526,867 | 384 | 0.073% |
| 2020 | 528,368 | 509 | 0.096% |
| 2021 | 529,366 | 358 | 0.068% |
| 2022 | 530,374 | 393 | 0.074% |
| 2023 | 532,172 | 436 | 0.082% |
| 2024 | 533,151 | 380 | 0.071% |
| **Total** | **5,277,543** | **5,100** | **0.097%** |

### Features

Features come in five groups. Neighbourhood features are computed citywide using 300 ft and 1,000 ft radii and the 25 nearest neighbours.

| Group | Features |
|---|---|
| **Value** | log market value, land share, land-to-improvement ratio, 1-year value change |
| **Relative value** | value ratio to the 25 nearest neighbours, neighbourhood value growth (3 yr), 3-year market and land value change |
| **Neighbourhood** | prior demolitions within 300 ft and 1,000 ft (previous 3 years), new-construction permits and severe violations within 300 ft |
| **Permits** | permit count, investment permits, years since last permit, make-safe and lot-line permits |
| **Violations** | violation cases, severe / vacant-structure / structural / sanitation violations, years since last violation, consecutive years with a vacant-structure violation, years since the first one |

Count features use a 3-year window. Recency features (`years_since_*`) are filled with a fixed value of 99 when no event has ever been recorded, plus an explicit `ever_*` flag. Median filling would wrongly imply the parcel *has* a history.

### Specification ladder

Seven feature sets are compared, building from a value-only baseline up to the full model:

| Specification | Contents |
|---|---|
| `baseline_3` | value + prior demolitions within 300 ft |
| `value_extended` | baseline + relative value |
| `permit_compact` | baseline + permits |
| `violation_compact` | baseline + violations |
| `combined_compact` | baseline + permits + violations |
| `combined_extended` | everything |
| **`combined_strict`** | everything **except** `new_construction`, `zoning_use` and `expired` permit counts |

`combined_strict` is the **primary model**. The three permit types it drops either sit too close to a redevelopment decision that has already been made (new construction, zoning/use) or are not time-safe (`expired` is derived from a permit's *current* status). Dropping them costs almost nothing (ROC-AUC 0.943 vs 0.945, top-10% capture 84.0% vs 84.1%). In exchange, none of the model's performance can be attributed to features that partly read the outcome.

### Algorithms

- **XGBoost**: 400 trees, depth 4, learning rate 0.05, `scale_pos_weight` = negatives / positives
- **Logistic regression**: L2, `class_weight="balanced"`, lbfgs solver, median imputation with missingness indicators, standardised inputs. Used as an interpretable companion and to measure ranking uncertainty.

All imputation and scaling is fitted inside each training fold.

### Validation: rolling origin

The model is always trained on the past and tested on the next unseen year:

| Fold | Train | Test |
|---|---|---|
| 1 | 2015–2018 | 2019 |
| 2 | 2015–2019 | 2020 |
| 3 | 2015–2020 | 2021 |
| 4 | 2015–2021 | 2022 |
| 5 | 2015–2022 | 2023 |
| 6 | 2015–2023 | 2024 |

Scores are converted to within-year percentiles before pooling, because six separately trained models do not share a common score scale. The final deployed models are then refitted on all labelled years, 2015–2024.

---

## Model comparison

Pooled out-of-sample results, 2019–2024 (from `model outputs/rolling_origin_pooled_metrics.csv`):

| Specification | Model | ROC-AUC | PR-AUC | Lift over prevalence | Top-10% capture | Top-20% capture |
|---|---|---:|---:|---:|---:|---:|
| `baseline_3` | XGBoost | 0.811 | 0.0089 | 11.5× | 52.0% | 67.5% |
| `value_extended` | XGBoost | 0.843 | 0.0090 | 11.6× | 54.4% | 71.6% |
| `permit_compact` | XGBoost | 0.845 | 0.0175 | 22.6× | 59.3% | 73.5% |
| `violation_compact` | XGBoost | 0.902 | 0.0551 | 71.2× | 74.1% | 84.0% |
| `combined_compact` | XGBoost | 0.923 | 0.0607 | 78.5× | 79.9% | 87.8% |
| `combined_extended` | XGBoost | 0.945 | 0.0733 | 94.8× | 84.1% | 90.7% |
| **`combined_strict`** | **XGBoost** | **0.943** | **0.0697** | **90.1×** | **84.0%** | **90.3%** |
| `combined_strict` | Logistic | 0.908 | 0.0204 | 26.4× | 78.9% | 87.4% |

Violation history adds the most predictive power. The neighbourhood and vacancy features introduced in `combined_extended` give a further clear gain over `combined_compact`.

### Primary model, year by year

| Test year | Demolitions | ROC-AUC | PR-AUC | Top-10% capture |
|---|---:|---:|---:|---:|
| 2019 | 384 | 0.951 | 0.082 | 86.7% |
| 2020 | 509 | 0.961 | 0.102 | 88.4% |
| 2021 | 358 | 0.938 | 0.065 | 82.7% |
| 2022 | 393 | 0.942 | 0.075 | 83.0% |
| 2023 | 436 | 0.937 | 0.075 | 81.9% |
| 2024 | 380 | 0.922 | 0.056 | 79.5% |

### Most important features (SHAP, `combined_strict` XGBoost)

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | Prior demolitions within 1,000 ft (3 yr) | 0.740 |
| 2 | Years since last violation | 0.726 |
| 3 | Years since first vacant-structure violation | 0.488 |
| 4 | Market value ratio to neighbours | 0.475 |
| 5 | Severe violation cases (3 yr) | 0.312 |
| 6 | Years since last permit | 0.242 |
| 7 | Log market value | 0.209 |

Beeswarm plots are in [`model outputs/shap_outputs/`](model%20outputs/shap_outputs/).

---

## The 2025 citywide ranking

### Building the scoring table

`preprocess_07_scoring_table_2025.py` rebuilds every model feature for risk year 2025 using the training panel's exact code path. Everything is cut at **31 December 2025**. The raw extracts run later than that, so 25,849 permits and 54,073 violations dated 2026, and all 2026–2027 assessment values, are excluded.

**Parity check:** the same code also rebuilds risk year 2024 and compares it with the training panel. Across 569,461 parcels, 25 of 26 features reproduce exactly and the 26th (`prior_demos_300ft_3yr`) agrees on 99.99% of rows. This confirms the 2025 table matches what the model was trained on.

### Scoring

`score_08_risk_ranking_2025.py` applies the frozen models without refitting:

| Model | Role |
|---|---|
| `combined_strict` XGBoost | **primary**: sets the rank and percentile |
| `combined_extended` XGBoost | sensitivity |
| `combined_strict` logistic | interpretable companion and ranking-uncertainty measure |

Features and scores are computed citywide *before* any geographic subset. This way parcels near the edge of a study area keep their full 300 ft and 1,000 ft neighbourhoods.

### Results

| | |
|---|---|
| Parcels scored | 535,000 (all parcels with a building) |
| Logistic vs XGBoost, Spearman ρ | 0.70 |
| Logistic vs XGBoost, top-1% overlap | 42.8% |

The disagreement between the two algorithms should be carried into the MFA as model uncertainty.

**Extrapolation check.** Philadelphia's citywide reassessment pushed the mean 1-year value change to 0.207 in 2025, compared with 0.003 in 2024. Tree models extrapolate poorly beyond their training range, so the primary model was re-scored with that feature blanked out. The ranking barely moves (Spearman 0.992, top-1% overlap 93.7%), so the 2025 ranking is not being driven by the reassessment.

**Concentration of risk:**

| Top slice | Parcels | Minimum score |
|---|---:|---:|
| 0.1% | 535 | 0.983 |
| 0.5% | 2,675 | 0.940 |
| 1% | 5,350 | 0.889 |
| 5% | 26,750 | 0.556 |
| 10% | 53,500 | 0.369 |

### Output columns (`risk_ranking_2025_citywide.csv`)

| Column | Description |
|---|---|
| `parcel_number` | OPA account number |
| `risk_year` | 2025 (predicts demolition starting in 2026) |
| `score_combined_strict_xgboost` | primary model score |
| `pct_combined_strict_xgboost` | its citywide percentile |
| `score_combined_extended_xgboost`, `pct_…` | sensitivity model score and percentile |
| `score_combined_strict_logistic`, `pct_…` | logistic companion score and percentile |
| `score_primary_no_value_change` | primary model with `market_value_change_1yr` blanked (extrapolation check) |
| `citywide_rank` | 1 = highest risk |
| `citywide_percentile` | percentile of the primary score (1.0 = highest) |
| `in_fishtown` | 1 if the parcel lies inside the Fishtown boundary polygon |
| `in_fishtown_old_id_set` | 1 if the parcel was in the earlier ID-based Fishtown panel |

---

## How to interpret the scores

> **Scores are relative rankings, not calibrated probabilities.**

XGBoost was trained with `scale_pos_weight` so it could learn from about 5,100 demolitions among 5.3 million parcel-years. That reweighting inflates the raw outputs. A parcel scoring 0.9 is near the top of the ranking. It does **not** have a 90% chance of demolition; the true annual rate even among the highest-risk parcels is a few percent.

- ✅ Use the **rank or percentile** to select the highest-risk buildings (e.g. the top 1% or top 5%).
- ✅ Use explicit low / medium / high scenarios for how many of them come down.
- ❌ Do not sum scores to estimate a demolition count. This would overstate it enormously.
- ❌ Do not report a score as a percentage chance.

---

## Reproducing the results

### Requirements

```
python >= 3.10
pandas, numpy, scipy, pyarrow
scikit-learn >= 1.8
xgboost
joblib
matplotlib
shap            # optional, for the SHAP outputs
```

The `.joblib` models should be loaded with the same scikit-learn and XGBoost versions used to fit them.

### Steps

1. **Train and evaluate** (`machine learning model/model_demolition_philly.py`): reads `philly_panel_v2.parquet` from its own folder and writes everything under `model_outputs_philly/all/`.
2. **Build the 2025 scoring table** (`risk ranking scripts/preprocess_07_scoring_table_2025.py`).
3. **Score and rank** (`risk ranking scripts/score_08_risk_ranking_2025.py`). It imports `model_demolition_philly.py` to reuse the feature sets and recency handling, then loads the fitted models.

> **Note on paths.** The scripts were run inside the original research directory, and steps 2 and 3 use a hard-coded `BASE` path. Before running them, point `BASE` (and `FITTED` in step 3) at your local copy.
>
> Step 2 also needs cleaned input tables that are not in this repository: `assessments_clean.csv`, `permits_clean.csv`, `violations_clean.csv`, `demolitions_clean.csv`, `demolitions_events_all.csv`, `demolitions_no_opa_account.csv`, and the OPA properties extract. These are produced by upstream preprocessing scripts, also not included here.

### Note on file sizes

GitHub rejects files over 100 MB. `philly_panel_v2.parquet` (~171 MB) and `rolling_origin_predictions.csv` (~3.6 GB) exceed this. They are therefore excluded from version control and are available from the author on request.

---

## Data sources

All inputs are public City of Philadelphia data, published through [OpenDataPhilly](https://opendataphilly.org/):

- **Office of Property Assessment (OPA)**: property characteristics, locations, and yearly market, land and improvement values
- **Department of Licenses & Inspections (L&I)**: building and zoning permits, code violations, and demolitions

---

## Limitations

- **Relative, not absolute.** The model ranks parcels within a year; it does not estimate how many demolitions will occur.
- **Mixed demolition processes.** About 66% of citywide demolitions are city-led (typically imminently dangerous buildings), a different process from private redevelopment. The deployed model pools both. The training script can fit private-only and city-only models (`POPULATIONS`).
- **Short value history.** Assessment records start in 2015, so the 3-year value change features are missing for risk years 2015–2017 (see `feature_availability_by_year.csv`).
- **No building age.** Building age was excluded: 68% of demolished parcels have been retired from the current OPA snapshot, so its missingness was strongly linked to the outcome. Relative-value features take its place.
- **Logistic coefficients are collinear.** Read them for direction only. SHAP on the XGBoost model is the better guide to feature importance.
- **Reassessment year.** 2025 values follow a citywide reassessment. The extrapolation check shows the ranking is robust to it, but a 2025 score is still applied to a value distribution the model never saw during training.

---

## License

Released under the [MIT License](LICENSE).
