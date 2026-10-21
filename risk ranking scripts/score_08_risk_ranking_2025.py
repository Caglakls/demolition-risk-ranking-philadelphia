"""
Step 8: score the 2025 building stock and produce the demolition-risk ranking.

Applies the FROZEN deployment models to risk_year_2025_scoring_table.parquet.
Nothing is refitted here.

Population
----------
Parcels carrying a building (improvement_value > 0), matching the training
filter. Vacant land cannot experience a building demolition.

Order of operations
-------------------
Features were computed citywide BEFORE any geographic subset, so parcels near
a neighbourhood boundary keep their full 300 ft and 1000 ft neighbourhoods.
Scoring is likewise citywide; the Fishtown crop happens last. Cropping first
would truncate the neighbourhood of every parcel within 300 ft of the edge --
about 17% of Fishtown.

Models
------
combined_strict   XGBoost   primary ranking
combined_extended XGBoost   sensitivity: adds the three near-event features
combined_strict   Logistic  interpretable companion / ranking-uncertainty

Extrapolation check
-------------------
market_value_change_1yr has a 2025 mean of 0.207 against 0.003 in 2024 -- the
citywide reassessment. Tree models extrapolate poorly outside the training
range, so the primary model is also scored with that feature blanked, and the
top-1% overlap between the two rankings is reported.

Outputs
-------
risk_ranking_2025_citywide.parquet / .csv
risk_ranking_2025_fishtown.csv
risk_ranking_2025_summary.txt
"""

from pathlib import Path
import importlib.util
import sys

import joblib
import numpy as np
import pandas as pd

BASE = Path(
    r"C:\Users\ck976\OneDrive\PHD\Circular economy"
    r"\chapter 3 supply demand prediction\datasets\philly model"
)
FITTED = BASE / "model_outputs_philly" / "all" / "fitted_models"
FISHTOWN_PANEL = (
    BASE.parent / "violations"
    / "historical_demolition_risk_panel_with_permits_violations.csv"
)

MODELS = [
    ("combined_strict", "xgboost", "primary"),
    ("combined_extended", "xgboost", "sensitivity"),
    ("combined_strict", "logistic", "companion"),
]


def load_model_module():
    """Reuse the training script's feature sets and recency handling verbatim."""
    spec = importlib.util.spec_from_file_location(
        "mdp", BASE / "model_demolition_philly.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mdp"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    print("=" * 72)
    print("STEP 8  RISK RANKING, RISK YEAR 2025")
    print("=" * 72)

    m = load_model_module()

    s = pd.read_parquet(BASE / "risk_year_2025_scoring_table.parquet")
    print(f"  scoring table rows            : {len(s):,}")

    s = s.loc[s["has_building"] == 1].reset_index(drop=True)
    print(f"  with a building (scored)      : {len(s):,}")

    # Identical sentinel treatment to training. The constant is fixed, so
    # this learns nothing from the scoring data.
    s = m.prepare_recency(s)
    for col, (flag, sentinel) in m.RECENCY_FEATURES.items():
        print(f"    {col}: sentinel {sentinel} applied, "
              f"{int((s[col] == sentinel).sum()):,} rows")

    out = s[["parcel_number"]].copy()
    out["risk_year"] = 2025

    for spec, algo, role in MODELS:
        path = FITTED / f"{spec}_{algo}.joblib"
        model = joblib.load(path)
        feats = m.FEATURE_SETS[spec]

        missing = [f for f in feats if f not in s.columns]
        if missing:
            raise ValueError(f"{spec}: scoring table missing {missing}")

        X = s[feats]
        scores = model.predict_proba(X)[:, 1]
        name = f"{spec}_{algo}"
        out[f"score_{name}"] = scores
        out[f"pct_{name}"] = pd.Series(scores).rank(method="average", pct=True)
        print(f"  scored {name:<32s} ({role:<11s}) "
              f"mean {scores.mean():.5f}  max {scores.max():.5f}")

    # ---------------------------------------------------- extrapolation ----
    spec, algo = "combined_strict", "xgboost"
    model = joblib.load(FITTED / f"{spec}_{algo}.joblib")
    feats = m.FEATURE_SETS[spec]
    X_blank = s[feats].copy()
    X_blank["market_value_change_1yr"] = np.nan
    blank_scores = model.predict_proba(X_blank)[:, 1]
    out["score_primary_no_value_change"] = blank_scores

    primary = out[f"score_{spec}_{algo}"].to_numpy()
    n_top = max(1, int(round(0.01 * len(out))))
    top_a = set(np.argsort(-primary, kind="stable")[:n_top])
    top_b = set(np.argsort(-blank_scores, kind="stable")[:n_top])
    overlap = len(top_a & top_b) / n_top
    rho = pd.Series(primary).corr(pd.Series(blank_scores), method="spearman")
    print(f"\n  extrapolation check: blanking market_value_change_1yr")
    print(f"    top-1% overlap {overlap:.3f}   spearman {rho:.4f}")

    # ---------------------------------------------------- rankings ---------
    prim = f"score_{spec}_{algo}"
    out = out.sort_values(prim, ascending=False).reset_index(drop=True)
    out["citywide_rank"] = np.arange(1, len(out) + 1)
    out["citywide_percentile"] = out[prim].rank(method="average", pct=True)

    # logistic-vs-xgboost agreement, carried into the MFA as model uncertainty
    lg = out["score_combined_strict_logistic"].to_numpy()
    xg = out[prim].to_numpy()
    rank_rho = pd.Series(lg).corr(pd.Series(xg), method="spearman")
    lo = set(np.argsort(-lg, kind="stable")[:n_top])
    xo = set(np.argsort(-xg, kind="stable")[:n_top])
    algo_overlap = len(lo & xo) / n_top

    # ---------------------------------------------------- fishtown ---------
    fish = pd.read_csv(FISHTOWN_PANEL, usecols=["parcel_number"], low_memory=False)
    fish_ids = set(fish["parcel_number"].unique())
    out["in_fishtown"] = out["parcel_number"].isin(fish_ids).astype("int8")

    fz = out.loc[out["in_fishtown"] == 1].copy()
    fz["fishtown_rank"] = np.arange(1, len(fz) + 1)
    fz["fishtown_percentile"] = fz[prim].rank(method="average", pct=True)
    print(f"\n  Fishtown parcels with a building: {len(fz):,}")

    out.to_parquet(BASE / "risk_ranking_2025_citywide.parquet", index=False)
    out.to_csv(BASE / "risk_ranking_2025_citywide.csv", index=False)
    fz.to_csv(BASE / "risk_ranking_2025_fishtown.csv", index=False)

    # ---------------------------------------------------- summary ----------
    lines = [
        "DEMOLITION RISK RANKING, RISK YEAR 2025",
        "=" * 52, "",
        f"Scored population           : {len(out):,} parcels with a building",
        f"Primary model               : combined_strict XGBoost (frozen)",
        f"Fishtown parcels            : {len(fz):,}",
        "",
        "Model agreement (carry into the MFA as uncertainty)",
        f"  logistic vs XGBoost spearman   : {rank_rho:.4f}",
        f"  top-1% overlap                 : {algo_overlap:.3f}",
        "",
        "Extrapolation check, market_value_change_1yr blanked",
        f"  spearman vs primary            : {rho:.4f}",
        f"  top-1% overlap                 : {overlap:.3f}",
        "",
        "Concentration of risk, citywide",
    ]
    for frac in [0.001, 0.005, 0.01, 0.05, 0.10]:
        k = max(1, int(round(frac * len(out))))
        lines.append(f"  top {frac*100:>5.1f}%  = {k:>7,} parcels   "
                     f"min score {out[prim].iloc[k-1]:.5f}")
    lines += ["", "Fishtown share of the citywide high-risk groups"]
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(round(frac * len(out))))
        n_f = int(out.iloc[:k]["in_fishtown"].sum())
        lines.append(f"  top {frac*100:>5.1f}%  contains {n_f:>5,} Fishtown parcels "
                     f"({100*n_f/len(fz):.2f}% of Fishtown stock)")
    lines += ["", "Scores are relative rankings, not calibrated probabilities.",
              "Do not sum them to estimate a demolition count."]
    (BASE / "risk_ranking_2025_summary.txt").write_text("\n".join(lines),
                                                        encoding="utf-8")
    print("\n" + "\n".join(lines[2:]))
    print(f"\nWrote 4 files to:\n{BASE}")


if __name__ == "__main__":
    main()
