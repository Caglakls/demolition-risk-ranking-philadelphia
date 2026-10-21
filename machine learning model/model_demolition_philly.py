"""
Rolling-origin demolition-risk modelling for Philadelphia.

Successor to model_demolition_with_permits_violations.py (Fishtown). The
validation design, metric set and specification-ladder logic are unchanged so
the two chapters stay comparable; the differences below are all forced by
scale or by defects found while rebuilding the data.

Input
-----
philly_panel_v2.parquet
    5,655,164 parcel-year rows, 574,330 parcels, 5,249 events, 2015-2024.

Differences from the Fishtown script, and why
---------------------------------------------
1. building_age is gone. 68% of demolished parcels have been retired from the
   current OPA snapshot, so every OPA-sourced attribute carried a ~50x
   missingness-to-target correlation. Its role is taken by the relative-value
   group (market_value_ratio_to_neighbors, neighbor_value_growth_3yr).

2. Logistic solver is lbfgs, not liblinear. liblinear is single-threaded and
   does not scale to millions of rows. Same L2 penalty, same class_weight.

3. Recency features are sentinel-filled, not median-imputed. A missing
   years_since_last_violation means NO violation history; the median asserts
   the opposite. Filling with a large constant plus an explicit ever_* flag
   is the fix requested in section 18 step 2 of the development notes. The
   sentinel is a fixed constant, so it learns nothing from the fold.

4. Remaining imputation uses add_indicator=True, so missingness enters the
   logistic model as an explicit term instead of being silently replaced.

5. combined_strict drops new_construction, zoning_use AND expired. The
   Fishtown strict spec dropped only new_construction, but zoning_use has a
   near-identical near-event profile, and expired is derived from CURRENT
   permit status so it is not time-safe at all.

6. Populations. City-led demolition is 66% of citywide events and is a
   different process from private redevelopment. Set POPULATIONS to fit the
   private-only and city-only models as well; the private model is the one
   comparable to the Fishtown result.

Timing structure is unchanged: a row with risk_year=t uses information
available by 31 December of year t and predicts demolition during t+1.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "philly_panel_v2.parquet"
OUTPUT_DIR = BASE / "model_outputs_philly"

RANDOM_STATE = 42
FIRST_TEST_YEAR = 2019

TARGET = "event_next_year"
YEAR_COL = "risk_year"
ID_COL = "parcel_number"

# Only parcels carrying an assessed building can lose one.
REQUIRE_BUILDING = True

# "all"      every demolition
# "private"  private redevelopment only; parcels later demolished by the city
#            are removed rather than counted as negatives
# "city"     city-led demolition only, same logic reversed
# Fitting all three roughly triples runtime. Start with ["all"].
POPULATIONS = ["all"]

# ---- feature blocks --------------------------------------------------------

VALUE = [
    "log_market_value",
    "land_share",
    "land_to_improvement_ratio",
    "market_value_change_1yr",
]

# NOTE: the 3-year change features are undefined for risk years 2015-2017
# because the assessment history starts in 2015. Their missingness therefore
# encodes "early year", and early years have higher prevalence. They are kept
# out of the compact specs for that reason and appear only in the extended
# spec, where feature_availability_by_year.csv makes the gap visible.
RELATIVE_VALUE = [
    "market_value_ratio_to_neighbors",
    "neighbor_value_growth_3yr",
    "market_value_change_3yr",
    "land_value_change_3yr",
]

NEIGHBORHOOD = ["prior_demos_300ft_3yr"]
NEIGHBORHOOD_EXTRA = [
    "prior_demos_1000ft_3yr",
    "neighbor_new_construction_300ft_3yr",
    "neighbor_severe_violations_300ft_3yr",
]

PERMIT_COMPACT = [
    "permit_count_3yr",
    "investment_count_3yr",
    "zoning_use_count_3yr",
    "new_construction_count_3yr",
    "expired_count_3yr",
    "years_since_last_permit",
]
PERMIT_STRICT = [
    "permit_count_3yr",
    "investment_count_3yr",
    "years_since_last_permit",
]
PERMIT_EXTRA = ["make_safe_count_3yr", "lot_line_count_3yr"]

VIOLATION_COMPACT = [
    "violation_case_count_3yr",
    "severe_violation_case_count_3yr",
    "years_since_last_violation",
]
VIOLATION_EXTRA = [
    "vacant_structure_violation_case_count_3yr",
    "structural_violation_case_count_3yr",
    "sanitation_violation_case_count_3yr",
    "consecutive_years_vacant_violation",
    "years_since_first_vacant_violation",
    "ever_violation_flag",
]

BASELINE = VALUE + NEIGHBORHOOD

FEATURE_SETS: dict[str, list[str]] = {
    "baseline_3": BASELINE,
    "value_extended": BASELINE + RELATIVE_VALUE,
    "permit_compact": BASELINE + PERMIT_COMPACT,
    "violation_compact": BASELINE + VIOLATION_COMPACT,
    "combined_compact": BASELINE + PERMIT_COMPACT + VIOLATION_COMPACT,
    "combined_extended": (
        BASELINE + RELATIVE_VALUE + NEIGHBORHOOD_EXTRA
        + PERMIT_COMPACT + PERMIT_EXTRA
        + VIOLATION_COMPACT + VIOLATION_EXTRA
    ),
    "combined_strict": (
        BASELINE + RELATIVE_VALUE + NEIGHBORHOOD_EXTRA
        + PERMIT_STRICT + PERMIT_EXTRA
        + VIOLATION_COMPACT + VIOLATION_EXTRA
    ),
}

# Missing means "no such event has ever been recorded", not "unknown".
RECENCY_FEATURES = {
    "years_since_last_permit": ("ever_permit_flag", 99.0),
    "years_since_last_violation": ("ever_violation_flag", 99.0),
    "years_since_first_vacant_violation": ("ever_vacant_violation_flag", 99.0),
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

SHAP_SAMPLE_SIZE = 30000
SHAP_SPECS = ["baseline_3", "combined_compact", "combined_extended",
              "combined_strict"]


# ============================================================
# 2. HELPERS
# ============================================================

@dataclass
class ModelBundle:
    name: str
    specification: str
    features: list[str]
    model: Any


def ensure_directories(population: str) -> Path:
    out = OUTPUT_DIR / population
    (out / "fitted_models").mkdir(parents=True, exist_ok=True)
    (out / "shap_outputs").mkdir(parents=True, exist_ok=True)
    return out


def all_features() -> list[str]:
    return sorted({f for fs in FEATURE_SETS.values() for f in fs})


def prepare_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Sentinel-fill the recency features and add their paired flags.

    A fixed constant is used, so nothing is learned from the data and this is
    safe to do once outside the fold loop.
    """
    for col, (flag, sentinel) in RECENCY_FEATURES.items():
        if col not in df.columns:
            continue
        if flag not in df.columns:
            df[flag] = df[col].notna().astype("int8")
        df[col] = df[col].fillna(sentinel)
    return df


def validate_input(df: pd.DataFrame) -> None:
    required = {ID_COL, YEAR_COL, TARGET}
    required.update(all_features())
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Panel is missing required columns:\n" + "\n".join(missing))

    if df.duplicated([ID_COL, YEAR_COL]).any():
        raise ValueError("Panel contains duplicated parcel-year rows.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1.")

    # A leaked outcome column would silently destroy the whole analysis.
    forbidden = {"demo_start_year", "demo_complete_year", "building_age",
                 "parcel_absent_from_opa", "age_missing"}
    used = forbidden.intersection(all_features())
    if used:
        raise ValueError(f"Outcome or OPA-derived columns in feature sets: {used}")


def make_logistic_pipeline(features: list[str]) -> Pipeline:
    """Median-impute (with an explicit missingness indicator) then standardise.

    Both steps are fitted inside the training fold only, so no test-year
    information reaches preprocessing.
    """
    preprocessing = ColumnTransformer(
        transformers=[(
            "numeric",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]),
            features,
        )],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # L2 is the default in sklearn >= 1.8; passing penalty= explicitly is
    # deprecated there, and n_jobs no longer has any effect.
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",          # liblinear does not scale to 5M rows
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])


def make_xgboost(y_train: pd.Series) -> XGBClassifier:
    positives = int(y_train.sum())
    if positives == 0:
        raise ValueError("Training fold contains zero positive events.")
    negatives = int(len(y_train) - positives)
    return XGBClassifier(**XGB_PARAMS, scale_pos_weight=negatives / positives)


def safe_roc_auc(y_true, scores) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def top_fraction_metrics(y_true, scores, fraction: float):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_sel = max(1, int(math.ceil(len(y) * fraction)))
    sel = np.argsort(-s, kind="stable")[:n_sel]
    hits = int(y[sel].sum())
    total = int(y.sum())
    return (hits / total if total else np.nan), hits / n_sel, n_sel, hits


def calculate_metrics(y_true, scores) -> dict[str, float | int]:
    prevalence = float(np.mean(y_true))
    pr_auc = float(average_precision_score(y_true, scores))
    c10, p10, n10, e10 = top_fraction_metrics(y_true, scores, 0.10)
    c20, p20, n20, e20 = top_fraction_metrics(y_true, scores, 0.20)
    c05, p05, n05, e05 = top_fraction_metrics(y_true, scores, 0.05)
    return {
        "n_rows": int(len(y_true)),
        "n_events": int(np.sum(y_true)),
        "prevalence": prevalence,
        "pr_auc": pr_auc,
        "pr_auc_lift_over_prevalence": pr_auc / prevalence if prevalence else np.nan,
        "roc_auc": safe_roc_auc(y_true, scores),
        "top_5pct_capture": c05,
        "top_10pct_capture": c10,
        "top_20pct_capture": c20,
        "precision_at_top_5pct": p05,
        "precision_at_top_10pct": p10,
        "precision_at_top_20pct": p20,
        "events_in_top_5pct": e05,
        "events_in_top_10pct": e10,
        "events_in_top_20pct": e20,
    }


def fit_predict(model_name: str, features: list[str],
                train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = train_df[features], train_df[TARGET].astype(int)
    model = (make_logistic_pipeline(features) if model_name == "logistic"
             else make_xgboost(y_train))
    model.fit(X_train, y_train)
    return model, model.predict_proba(test_df[features])[:, 1]


# ============================================================
# 3. POPULATION AND DATA QUALITY
# ============================================================

def select_population(df: pd.DataFrame, population: str) -> pd.DataFrame:
    """Restrict to one demolition process.

    A parcel later demolished by the city is REMOVED from the private
    population rather than kept as a negative: it did get demolished, so
    labelling it 0 would be wrong.
    """
    if population == "all":
        return df
    keep_agent = "PRIVATE" if population == "private" else "CITY"
    drop_agent = "CITY" if population == "private" else "PRIVATE"
    drop_parcels = set(df.loc[df["demo_agent"] == drop_agent, ID_COL].unique())
    return df.loc[~df[ID_COL].isin(drop_parcels)].copy()


def data_quality_summary(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for f in all_features():
        s = pd.to_numeric(df[f], errors="coerce")
        rows.append({
            "variable": f,
            "missing_n": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "unique_n": int(s.nunique(dropna=True)),
            "min": s.min(), "median": s.median(), "max": s.max(),
            "mean_event": float(s[df[TARGET] == 1].mean()),
            "mean_nonevent": float(s[df[TARGET] == 0].mean()),
        })
    pd.DataFrame(rows).to_csv(out / "data_quality_summary.csv", index=False)

    df.groupby(YEAR_COL)[TARGET].agg(
        n_rows="size", n_events="sum", prevalence="mean"
    ).to_csv(out / "events_by_risk_year.csv")

    # Which features are actually populated in which year. This is where the
    # 2015-2017 gap in the 3-year change features shows up.
    avail = df.groupby(YEAR_COL)[all_features()].apply(
        lambda g: 100 * g.notna().mean()
    ).round(2)
    avail.to_csv(out / "feature_availability_by_year.csv")


# ============================================================
# 4. ROLLING-ORIGIN EVALUATION
# ============================================================

def run_rolling_origin(df: pd.DataFrame):
    years = sorted(int(y) for y in df[YEAR_COL].unique())
    test_years = [y for y in years if y >= FIRST_TEST_YEAR]

    yearly_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for spec, features in FEATURE_SETS.items():
        for model_name in ["logistic", "xgboost"]:
            print(f"\n  {spec} | {model_name}")
            for test_year in test_years:
                train_df = df.loc[df[YEAR_COL] < test_year]
                test_df = df.loc[df[YEAR_COL] == test_year]
                if train_df.empty or test_df.empty:
                    continue

                _, scores = fit_predict(model_name, features, train_df, test_df)
                metrics = calculate_metrics(test_df[TARGET].astype(int), scores)
                print(f"    test {test_year}: train {len(train_df):,}"
                      f"/{int(train_df[TARGET].sum())} ev  ->  "
                      f"PR-AUC {metrics['pr_auc']:.4f}  "
                      f"ROC {metrics['roc_auc']:.3f}  "
                      f"top10 {metrics['top_10pct_capture']:.3f}")

                yearly_rows.append({
                    "specification": spec, "model": model_name,
                    "test_year": test_year,
                    "train_start_year": int(train_df[YEAR_COL].min()),
                    "train_end_year": int(train_df[YEAR_COL].max()),
                    "train_rows": int(len(train_df)),
                    "train_events": int(train_df[TARGET].sum()),
                    **metrics,
                })

                fold = test_df[[ID_COL, YEAR_COL, TARGET]].copy()
                fold["specification"] = spec
                fold["model"] = model_name
                fold["predicted_probability"] = scores
                # Raw scores from six separately trained models are not on a
                # common scale; the within-year percentile is (section 11.5).
                fold["risk_percentile_within_year"] = pd.Series(
                    scores, index=fold.index
                ).rank(method="average", pct=True)
                prediction_frames.append(fold)

    yearly = pd.DataFrame(yearly_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    pooled_rows = []
    for (spec, model_name), g in predictions.groupby(["specification", "model"],
                                                     sort=False):
        raw = calculate_metrics(g[TARGET].astype(int),
                                g["predicted_probability"].to_numpy())
        pct = calculate_metrics(g[TARGET].astype(int),
                                g["risk_percentile_within_year"].to_numpy())
        sub = yearly[(yearly.specification == spec) & (yearly.model == model_name)]
        pooled_rows.append({
            "specification": spec, "model": model_name,
            "first_test_year": int(g[YEAR_COL].min()),
            "last_test_year": int(g[YEAR_COL].max()),
            **raw,
            "pr_auc_within_year_percentile": pct["pr_auc"],
            "roc_auc_within_year_percentile": pct["roc_auc"],
            "top_10pct_capture_within_year": pct["top_10pct_capture"],
            "mean_yearly_pr_auc": sub["pr_auc"].mean(),
            "median_yearly_pr_auc": sub["pr_auc"].median(),
            "std_yearly_pr_auc": sub["pr_auc"].std(),
            "mean_yearly_roc_auc": sub["roc_auc"].mean(),
            "min_yearly_roc_auc": sub["roc_auc"].min(),
            "mean_yearly_top_10pct_capture": sub["top_10pct_capture"].mean(),
        })

    pooled = (pd.DataFrame(pooled_rows)
              .sort_values(["pr_auc_within_year_percentile", "pr_auc"],
                           ascending=False)
              .reset_index(drop=True))
    return yearly, pooled, predictions


# ============================================================
# 5. FINAL MODELS, COEFFICIENTS, RANKING AGREEMENT, SHAP
# ============================================================

def fit_final_models(df: pd.DataFrame, out: Path) -> dict:
    fitted = {}
    for spec, features in FEATURE_SETS.items():
        X, y = df[features], df[TARGET].astype(int)
        for model_name in ["logistic", "xgboost"]:
            model = (make_logistic_pipeline(features) if model_name == "logistic"
                     else make_xgboost(y))
            model.fit(X, y)
            joblib.dump(model, out / "fitted_models" / f"{spec}_{model_name}.joblib")
            fitted[(spec, model_name)] = ModelBundle(model_name, spec, features, model)
        print(f"    final models fitted: {spec}")
    return fitted


def save_logistic_coefficients(fitted: dict, out: Path) -> None:
    rows = []
    for (spec, model_name), bundle in fitted.items():
        if model_name != "logistic":
            continue
        clf = bundle.model.named_steps["classifier"]
        names = bundle.model.named_steps["preprocess"].get_feature_names_out()
        for name, coef in zip(names, clf.coef_[0]):
            rows.append({
                "specification": spec,
                "feature": name,
                "standardized_coefficient": float(coef),
                "odds_ratio_per_1_sd": float(np.exp(coef)),
                "direction": "higher risk" if coef > 0 else "lower risk",
                "intercept": float(clf.intercept_[0]),
            })
    d = pd.DataFrame(rows)
    d["absolute_coefficient"] = d["standardized_coefficient"].abs()
    d.sort_values(["specification", "absolute_coefficient"],
                  ascending=[True, False]).to_csv(
        out / "final_logistic_standardized_coefficients.csv", index=False)


def compare_rankings(predictions: pd.DataFrame, out: Path) -> None:
    rows = []
    for spec in FEATURE_SETS:
        lg = predictions[(predictions.specification == spec)
                         & (predictions.model == "logistic")][
            [ID_COL, YEAR_COL, "predicted_probability"]].rename(
            columns={"predicted_probability": "logistic_probability"})
        xg = predictions[(predictions.specification == spec)
                         & (predictions.model == "xgboost")][
            [ID_COL, YEAR_COL, "predicted_probability"]].rename(
            columns={"predicted_probability": "xgboost_probability"})
        m = lg.merge(xg, on=[ID_COL, YEAR_COL], how="inner", validate="one_to_one")
        rho, p = spearmanr(m["logistic_probability"], m["xgboost_probability"])
        for frac in [0.10, 0.20]:
            n_top = max(1, int(math.ceil(len(m) * frac)))
            overlap = len(set(m.nlargest(n_top, "logistic_probability").index)
                          & set(m.nlargest(n_top, "xgboost_probability").index))
            rows.append({
                "specification": spec, "spearman_rho": float(rho),
                "spearman_p_value": float(p), "top_fraction": frac,
                "n_top": n_top, "top_overlap_n": overlap,
                "top_overlap_pct": overlap / n_top,
            })
    pd.DataFrame(rows).to_csv(out / "final_model_ranking_comparison.csv", index=False)


def run_shap(df: pd.DataFrame, fitted: dict, out: Path) -> None:
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP not installed; skipping. pip install shap")
        return

    rng = np.random.default_rng(RANDOM_STATE)
    # Oversample events so the explanation is not all negatives: at a 0.1%
    # base rate a uniform sample of 30,000 would contain about 30 events.
    events = df.index[df[TARGET] == 1].to_numpy()
    others = df.index[df[TARGET] == 0].to_numpy()
    n_event = min(len(events), SHAP_SAMPLE_SIZE // 3)
    n_other = SHAP_SAMPLE_SIZE - n_event
    idx = np.concatenate([
        rng.choice(events, size=n_event, replace=False),
        rng.choice(others, size=min(n_other, len(others)), replace=False),
    ])

    for spec in SHAP_SPECS:
        if (spec, "xgboost") not in fitted:
            continue
        bundle = fitted[(spec, "xgboost")]
        X = df.loc[idx, bundle.features]
        explainer = shap.TreeExplainer(bundle.model)
        values = explainer(X)

        pd.DataFrame({
            "specification": spec,
            "feature": bundle.features,
            "mean_absolute_shap": np.abs(values.values).mean(axis=0),
        }).sort_values("mean_absolute_shap", ascending=False).to_csv(
            out / "shap_outputs" / f"{spec}_xgboost_shap_importance.csv",
            index=False)

        plt.figure()
        shap.plots.beeswarm(values, max_display=len(bundle.features), show=False)
        plt.tight_layout()
        plt.savefig(out / "shap_outputs" / f"{spec}_xgboost_shap_beeswarm.png",
                    dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    SHAP done: {spec}")


def write_selection_summary(pooled: pd.DataFrame, out: Path) -> None:
    ranked = pooled.sort_values(
        ["pr_auc_within_year_percentile", "top_10pct_capture", "roc_auc"],
        ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    lines = [
        "ROLLING-ORIGIN MODEL COMPARISON (Philadelphia)",
        "=" * 50, "",
        "Ranked on PR-AUC after converting each test year to a common",
        "within-year percentile scale. Raw pooled PR-AUC mixes six",
        "separately trained models whose score scales are not comparable;",
        "the research use is ranking parcels within a year.", "",
        f"  Specification: {best['specification']}",
        f"  Model: {best['model']}",
        f"  PR-AUC (within-year percentile): {best['pr_auc_within_year_percentile']:.6f}",
        f"  PR-AUC (raw pooled): {best['pr_auc']:.6f}",
        f"  PR-AUC lift over prevalence: {best['pr_auc_lift_over_prevalence']:.2f}",
        f"  ROC-AUC: {best['roc_auc']:.4f}",
        f"  Top-10% capture: {best['top_10pct_capture']:.3f}",
        f"  Top-20% capture: {best['top_20pct_capture']:.3f}",
        f"  Yearly PR-AUC sd: {best['std_yearly_pr_auc']:.6f}", "",
        "Do not select the final model from this line alone. Check:",
        "1. yearly fold stability (std_yearly_pr_auc, min_yearly_roc_auc);",
        "2. whether combined_extended beats combined_compact, i.e. whether",
        "   the new neighbourhood and vacancy features earn their place;",
        "3. whether the gain survives combined_strict, which removes",
        "   new_construction, zoning_use and expired;",
        "4. coefficient and SHAP plausibility;",
        "5. logistic-XGBoost ranking overlap.", "",
        "Reminder: scores are relative rankings, not calibrated",
        "probabilities. Do not report a score of 0.7 as a 70% chance.",
    ]
    (out / "selected_model_summary.txt").write_text("\n".join(lines),
                                                    encoding="utf-8")


# ============================================================
# 6. MAIN
# ============================================================

def main() -> None:
    print(f"Reading {DATA_PATH.name}")
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(DATA_PATH).schema.names)
    wanted = {ID_COL, YEAR_COL, TARGET, "has_building", "demo_agent"}
    wanted.update(all_features())
    # ever_* flags are created by prepare_recency() if the panel lacks them.
    wanted.update(f for f, _ in RECENCY_FEATURES.values())
    panel = pd.read_parquet(DATA_PATH, columns=sorted(wanted & available))

    if REQUIRE_BUILDING:
        before = len(panel)
        panel = panel[panel["has_building"] == 1].copy()
        print(f"  has_building filter: {before:,} -> {len(panel):,} rows")

    panel[YEAR_COL] = panel[YEAR_COL].astype(int)
    panel[TARGET] = panel[TARGET].astype(int)
    panel = prepare_recency(panel)
    validate_input(panel)

    for population in POPULATIONS:
        out = ensure_directories(population)
        df = select_population(panel, population).reset_index(drop=True)
        print("\n" + "=" * 66)
        print(f"POPULATION: {population}   rows {len(df):,}   "
              f"events {int(df[TARGET].sum()):,}   "
              f"prevalence {100 * df[TARGET].mean():.4f}%")
        print("=" * 66)

        data_quality_summary(df, out)
        yearly, pooled, predictions = run_rolling_origin(df)

        yearly.to_csv(out / "rolling_origin_yearly_metrics.csv", index=False)
        pooled.to_csv(out / "rolling_origin_pooled_metrics.csv", index=False)
        predictions.to_csv(out / "rolling_origin_predictions.csv", index=False)

        print("\nPooled comparison (ranked on within-year percentile PR-AUC):")
        print(pooled[["specification", "model", "n_events", "prevalence",
                      "pr_auc", "pr_auc_within_year_percentile",
                      "pr_auc_lift_over_prevalence", "roc_auc",
                      "top_10pct_capture", "top_20pct_capture",
                      "std_yearly_pr_auc"]].to_string(index=False))

        print("\nFitting final models on all labelled years ...")
        fitted = fit_final_models(df, out)
        save_logistic_coefficients(fitted, out)
        compare_rankings(predictions, out)
        run_shap(df, fitted, out)
        write_selection_summary(pooled, out)

        (out / "model_configuration.json").write_text(json.dumps({
            "data_path": str(DATA_PATH),
            "population": population,
            "require_building": REQUIRE_BUILDING,
            "first_test_year": FIRST_TEST_YEAR,
            "target": TARGET,
            "feature_sets": FEATURE_SETS,
            "recency_sentinel": {k: v[1] for k, v in RECENCY_FEATURES.items()},
            "xgboost_parameters": XGB_PARAMS,
            "logistic_solver": "lbfgs",
            "random_state": RANDOM_STATE,
        }, indent=2), encoding="utf-8")

        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
