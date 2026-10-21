"""
Step 7: build the risk-year-2025 feature table for the current building stock.

This is the population the trained model is applied to. It carries no target:
whether a demolition starts in 2026 is what we are predicting.

Time boundary
-------------
Everything is cut at 31 DECEMBER 2025. The raw extracts run further -- permits
to August 2026, violations to July 2026, assessments to 2027 -- and none of
that may enter a risk-year-2025 row. Specifically excluded:

    25,849 permits dated 2026
    54,073 violations dated 2026
    assessment values for 2026 and 2027
    demolitions dated 2025 or 2026

Demolitions enter only through the neighbourhood feature, whose window is
[t-3, t-1] = 2022-2024, so nothing from 2025 onward is used there either.

Parity check
------------
The same code path also rebuilds risk year 2024 and compares it against
philly_panel_v2.parquet. If the 2024 rebuild does not reproduce the training
panel, the 2025 table is not comparable to what the model was fitted on and
the scores would be meaningless. Results go to scoring_table_parity_check.csv.

Outputs
-------
risk_year_2025_scoring_table.parquet   the population to score
risk_year_2025_scoring_table.csv       same, model columns only
scoring_table_parity_check.csv         2024 rebuild vs training panel
scoring_table_build_report.csv         audit trail
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(
    r"C:\Users\ck976\OneDrive\PHD\Circular economy"
    r"\chapter 3 supply demand prediction\datasets\philly model"
)

TARGET_YEAR = 2025
PARITY_YEAR = 2024
YEARS = [PARITY_YEAR, TARGET_YEAR]

WINDOWS = [1, 3, 5]
K_NEIGHBORS = 25
RADIUS_NEAR = 300.0
RADIUS_WIDE = 1000.0
DEMO_WINDOW = (-3, -1)
MAX_RUN_YEARS = 12

PERMIT_CATS = ["alteration", "addition", "systems", "investment", "zoning_use",
               "change_of_use", "new_construction", "expired", "abandoned",
               "cancelled_refused", "make_safe", "lot_line", "foundation",
               "envelope", "partial_change"]

VIOL_CATS = ["severe", "construction", "severe_extended", "vacant_structure",
             "vacant_lot", "structural", "sanitation", "license"]

# Spatial features are undefined, NOT zero, for a parcel with no coordinates.
# The training panel leaves them NaN and XGBoost learned a branch for that, so
# they must never be filled here.
SPATIAL_FEATURES = ["prior_demos_300ft_3yr", "prior_demos_1000ft_3yr",
                    "neighbor_new_construction_300ft_3yr",
                    "neighbor_severe_violations_300ft_3yr"]

audit: list[dict] = []


def note(step: str, n, detail: str = "") -> None:
    audit.append({"step": step, "value": n, "detail": detail})
    shown = f"{n:,}" if isinstance(n, (int, np.integer)) else n
    print(f"{shown:>12}  {step}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------

def build_population(assess: pd.DataFrame, demo: pd.DataFrame,
                     geo: pd.DataFrame, year: int) -> pd.DataFrame:
    """Parcels still standing and at risk at the end of `year`."""
    pop = assess.loc[assess["year"] == year].copy()
    pop = pop.rename(columns={"year": "risk_year"})
    pop = pop.merge(demo, on="parcel_number", how="left")

    already_gone = pop["demo_start_year"].notna() & (
        pop["risk_year"] >= pop["demo_start_year"]
    )
    note(f"{year}: excluded, demolition already begun", int(already_gone.sum()))
    pop = pop.loc[~already_gone].copy()

    pop["has_building"] = (pop["improvement_value"] > 0).astype("int8")
    pop = pop.merge(geo, on="parcel_number", how="left")
    note(f"{year}: population rows", len(pop))
    note(f"{year}: of which carry a building", int(pop["has_building"].sum()))
    return pop


def assessment_change_3yr(assess: pd.DataFrame, year: int) -> pd.DataFrame:
    """Three-year log change, computed from values no later than `year`."""
    a = assess.loc[assess["year"] <= year].sort_values(
        ["parcel_number", "year"]
    ).copy()
    a["_log_land"] = np.log(a["land_value"].where(a["land_value"] > 0))
    out = {}
    for src, name in [("log_market_value", "market_value_change_3yr"),
                      ("_log_land", "land_value_change_3yr")]:
        prev = a.groupby("parcel_number", sort=False)[src].shift(3)
        prev_year = a.groupby("parcel_number", sort=False)["year"].shift(3)
        a[name] = np.where((a["year"] - prev_year) == 3, a[src] - prev, np.nan)
        out[name] = name
    sel = a.loc[a["year"] == year,
                ["parcel_number", "market_value_change_3yr",
                 "land_value_change_3yr"]]
    return sel


def permit_features(perm: pd.DataFrame, parcels: pd.Index, year: int) -> pd.DataFrame:
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    prior = perm.loc[perm["permit_dt"] <= year_end]

    frame = pd.DataFrame(index=parcels)
    g = prior.groupby("parcel_number", sort=False)
    frame["prior_permit_count_since_2007"] = (
        frame.index.map(g.size()).fillna(0).astype("int32")
    )
    last = frame.index.map(g["permit_dt"].max())
    frame["years_since_last_permit"] = (
        (year_end - pd.Series(last, index=frame.index)).dt.days / 365.25
    )

    for w in WINDOWS:
        start = pd.Timestamp(f"{year - w + 1}-01-01")
        recent = prior.loc[prior["permit_dt"] >= start]
        rg = recent.groupby("parcel_number", sort=False)
        frame[f"permit_count_{w}yr"] = (
            frame.index.map(rg.size()).fillna(0).astype("int32")
        )
        sums = rg[PERMIT_CATS].sum()
        for c in PERMIT_CATS:
            frame[f"{c}_count_{w}yr"] = (
                frame.index.map(sums[c]).fillna(0).astype("int32")
            )
    return frame.reset_index()


def violation_features(viol: pd.DataFrame, parcels: pd.Index, year: int) -> pd.DataFrame:
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    prior = viol.loc[viol["violation_dt"] <= year_end]

    frame = pd.DataFrame(index=parcels)
    g = prior.groupby("parcel_number", sort=False)
    last = frame.index.map(g["violation_dt"].max())
    last = pd.Series(last, index=frame.index)
    frame["prior_violation_case_count_since_2007"] = (
        frame.index.map(g["case_code"].nunique()).fillna(0).astype("int32")
    )
    frame["years_since_last_violation"] = (year_end - last).dt.days / 365.25
    frame["ever_violation_flag"] = last.notna().astype("int8")

    for w in WINDOWS:
        start = pd.Timestamp(f"{year - w + 1}-01-01")
        recent = prior.loc[prior["violation_dt"] >= start]
        rg = recent.groupby("parcel_number", sort=False)
        frame[f"violation_case_count_{w}yr"] = (
            frame.index.map(rg["case_code"].nunique()).fillna(0).astype("int32")
        )
        frame[f"violation_record_count_{w}yr"] = (
            frame.index.map(rg["violationnumber"].nunique()).fillna(0).astype("int32")
        )
        frame[f"violation_active_years_{w}yr"] = (
            frame.index.map(rg["violation_year"].nunique()).fillna(0).astype("int32")
        )
        for c in VIOL_CATS:
            sub = recent.loc[recent[c]]
            cnt = sub.groupby("parcel_number", sort=False)["case_code"].nunique()
            frame[f"{c}_violation_case_count_{w}yr"] = (
                frame.index.map(cnt).fillna(0).astype("int32")
            )
    return frame.reset_index()


def neighbour_median_value(pop: pd.DataFrame) -> pd.Series:
    sub = pop.loc[pop["x"].notna(), ["parcel_number", "x", "y", "market_value"]]
    ref = sub.loc[sub["market_value"].notna()].reset_index(drop=True)
    tree = cKDTree(ref[["x", "y"]].to_numpy())
    _, idx = tree.query(sub[["x", "y"]].to_numpy(), k=K_NEIGHBORS + 1, workers=-1)
    med = np.median(ref["market_value"].to_numpy()[idx[:, 1:]], axis=1)
    return pd.Series(med, index=sub["parcel_number"].to_numpy())


def radius_count(pop: pd.DataFrame, pts: pd.DataFrame, lo: int, hi: int,
                 radius: float, year: int, self_counts=None) -> pd.Series:
    sub = pop.loc[pop["x"].notna(), ["parcel_number", "x", "y"]]
    w = pts[(pts["year"] >= year + lo) & (pts["year"] <= year + hi)]
    if w.empty:
        return pd.Series(0.0, index=sub["parcel_number"].to_numpy())
    tree = cKDTree(w[["x", "y"]].to_numpy())
    counts = tree.query_ball_point(sub[["x", "y"]].to_numpy(), r=radius,
                                   return_length=True).astype("float64")
    if self_counts is not None:
        own = (self_counts[(self_counts["year"] >= year + lo)
                           & (self_counts["year"] <= year + hi)]
               .groupby("parcel_number")["n"].sum())
        counts -= sub["parcel_number"].map(own).fillna(0).to_numpy()
        counts = np.clip(counts, 0, None)
    return pd.Series(counts, index=sub["parcel_number"].to_numpy())


def vacancy_duration(viol: pd.DataFrame, parcels: pd.Index, year: int) -> pd.DataFrame:
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    vac = viol.loc[viol["vacant_structure"] & (viol["violation_dt"] <= year_end)]

    first = vac.groupby("parcel_number")["violation_dt"].min()
    presence = (vac.drop_duplicates(["parcel_number", "violation_year"])
                .assign(flag=1)
                .pivot(index="parcel_number", columns="violation_year", values="flag"))
    years = list(range(2007, year + 1))
    presence = presence.reindex(columns=years, fill_value=0).fillna(0)
    arr = presence.to_numpy(dtype="int16")
    run = np.zeros_like(arr)
    run[:, 0] = arr[:, 0]
    for j in range(1, arr.shape[1]):
        run[:, j] = np.where(arr[:, j] > 0, run[:, j - 1] + 1, 0)
    run = np.clip(run, 0, MAX_RUN_YEARS)

    frame = pd.DataFrame(index=parcels)
    frame["consecutive_years_vacant_violation"] = (
        frame.index.map(pd.Series(run[:, -1], index=presence.index))
        .fillna(0).astype("float32")
    )
    frame["years_since_first_vacant_violation"] = frame.index.map(
        (year_end - first).dt.days / 365.25
    )
    return frame.reset_index()


# ---------------------------------------------------------------------------

def build_year(year: int, assess, demo, geo, perm, viol, dpts, ppts, pself,
               spts, sself, nbr_lag: pd.Series) -> pd.DataFrame:
    print(f"\n  ---- risk year {year} ----")
    pop = build_population(assess, demo, geo, year)
    parcels = pd.Index(np.sort(pop["parcel_number"].unique()), name="parcel_number")

    pop = pop.merge(assessment_change_3yr(assess, year), on="parcel_number", how="left")
    pop = pop.merge(permit_features(perm, parcels, year), on="parcel_number", how="left")
    pop = pop.merge(violation_features(viol, parcels, year), on="parcel_number", how="left")
    pop = pop.merge(vacancy_duration(viol, parcels, year), on="parcel_number", how="left")

    nbr = neighbour_median_value(pop)
    pop["neighbor_median_value"] = pop["parcel_number"].map(nbr).astype("float32")
    pop["market_value_ratio_to_neighbors"] = (
        pop["market_value"] / pop["neighbor_median_value"].replace(0, np.nan)
    ).astype("float32")
    lag = pop["parcel_number"].map(nbr_lag)
    pop["neighbor_value_growth_3yr"] = (
        np.log(pop["neighbor_median_value"].where(pop["neighbor_median_value"] > 0))
        - np.log(lag.where(lag > 0))
    ).astype("float32")

    pop["prior_demos_300ft_3yr"] = pop["parcel_number"].map(
        radius_count(pop, dpts, *DEMO_WINDOW, RADIUS_NEAR, year)).astype("float32")
    pop["prior_demos_1000ft_3yr"] = pop["parcel_number"].map(
        radius_count(pop, dpts, *DEMO_WINDOW, RADIUS_WIDE, year)).astype("float32")
    pop["neighbor_new_construction_300ft_3yr"] = pop["parcel_number"].map(
        radius_count(pop, ppts, -2, 0, RADIUS_NEAR, year, pself)).astype("float32")
    pop["neighbor_severe_violations_300ft_3yr"] = pop["parcel_number"].map(
        radius_count(pop, spts, -2, 0, RADIUS_NEAR, year, sself)).astype("float32")

    pop["violation_recent_share_3of5"] = (
        pop["violation_case_count_3yr"] / pop["violation_case_count_5yr"].replace(0, np.nan)
    ).astype("float32")
    pop["permit_recent_share_3of5"] = (
        pop["permit_count_3yr"] / pop["permit_count_5yr"].replace(0, np.nan)
    ).astype("float32")

    count_cols = [c for c in pop.columns if c.endswith("yr") and "years_since" not in c
                  and "share" not in c and "growth" not in c and "change" not in c
                  and c not in SPATIAL_FEATURES]
    pop[count_cols] = pop[count_cols].fillna(0)
    pop["ever_violation_flag"] = pop["ever_violation_flag"].fillna(0).astype("int8")
    pop["consecutive_years_vacant_violation"] = (
        pop["consecutive_years_vacant_violation"].fillna(0).astype("float32")
    )
    return pop


def main() -> None:
    print("=" * 72)
    print("STEP 7  RISK-YEAR-2025 SCORING TABLE")
    print("=" * 72)

    panel = pd.read_parquet(
        BASE / "philly_panel_v2.parquet",
        columns=["parcel_number", "risk_year", "x", "y", "neighbor_median_value"],
    )
    # Panel geometry first, so parcels shared with the training data keep
    # byte-identical coordinates. OPA then fills parcels that entered the
    # stock after 2024 and are therefore absent from the panel.
    geo = (panel[["parcel_number", "x", "y"]]
           .dropna(subset=["x", "y"])
           .drop_duplicates("parcel_number").reset_index(drop=True))
    note("parcel coordinates from the training panel", len(geo))

    opa = pd.read_csv(BASE / "raw datasets" / "opa_properties_public (2).csv",
                      usecols=["parcel_number", "shape"], low_memory=False)
    opa["parcel_number"] = pd.to_numeric(
        opa["parcel_number"], errors="coerce").round().astype("Int64")
    xy = opa["shape"].str.extract(r"POINT\s*\(([-\d.]+)\s+([-\d.]+)\)")
    opa["x"] = pd.to_numeric(xy[0], errors="coerce")
    opa["y"] = pd.to_numeric(xy[1], errors="coerce")
    opa = (opa.dropna(subset=["parcel_number", "x", "y"])[["parcel_number", "x", "y"]]
           .drop_duplicates("parcel_number"))
    opa["parcel_number"] = opa["parcel_number"].astype("int64")

    # Two coordinate tables, deliberately.
    #
    # geo_ref is panel-only and is used to LOCATE the permit and violation
    # reference points for the neighbourhood counts. It must stay frozen,
    # because training built those reference sets from panel geometry alone.
    # Widening it adds reference points that the model never saw, which
    # inflates every neighbourhood count relative to the training
    # distribution -- an earlier attempt to "improve" coverage here dropped
    # neighbor_new_construction parity to 88.2%.
    #
    # geo_query is panel plus OPA and supplies the parcel's OWN location.
    # Extending it is safe: shared parcels keep identical coordinates because
    # the panel is concatenated first, and parcels new since 2024 are absent
    # from the parity comparison by construction.
    geo_ref = geo
    before = len(geo)
    geo_query = (pd.concat([geo, opa], ignore_index=True)
                 .drop_duplicates("parcel_number", keep="first")
                 .reset_index(drop=True))
    note("parcel coordinates for scoring (panel + OPA)", len(geo_query),
         f"{len(geo_query) - before:,} added for parcels new since 2024")
    note("parcel coordinates for reference sets (panel only)", len(geo_ref),
         "frozen to match training")
    geo = geo_query

    assess = pd.read_csv(BASE / "assessments_clean.csv", low_memory=False)
    assess = assess[assess["year"] <= TARGET_YEAR].copy()
    note("assessment rows kept (<= 2025)", len(assess),
         "2026 and 2027 values excluded entirely")

    demo = pd.read_csv(BASE / "demolitions_clean.csv",
                       usecols=["parcel_number", "demo_start_year"]).drop_duplicates("parcel_number")

    perm = pd.read_csv(BASE / "permits_clean.csv", low_memory=False)
    perm["permit_dt"] = pd.to_datetime(perm["permit_dt"])
    n26 = int((perm["permit_year"] >= 2026).sum())
    perm = perm[perm["permit_year"] <= TARGET_YEAR].copy()
    note("permits excluded as dated 2026 or later", n26, "would leak the future")

    viol = pd.read_csv(BASE / "violations_clean.csv", low_memory=False)
    viol["violation_dt"] = pd.to_datetime(viol["violation_dt"])
    v26 = int((viol["violation_year"] >= 2026).sum())
    viol = viol[viol["violation_year"] <= TARGET_YEAR].copy()
    note("violations excluded as dated 2026 or later", v26, "would leak the future")
    viol["case_code"] = pd.factorize(viol["casenumber"])[0]

    # reference point sets
    dpts = pd.concat([
        pd.read_csv(BASE / "demolitions_events_all.csv",
                    usecols=["demo_start_year", "geocode_x", "geocode_y"]),
        pd.read_csv(BASE / "demolitions_no_opa_account.csv",
                    usecols=["demo_start_year", "geocode_x", "geocode_y"]),
    ], ignore_index=True).dropna().rename(
        columns={"demo_start_year": "year", "geocode_x": "x", "geocode_y": "y"})

    # geo_ref, not geo_query -- see the comment where the two are defined.
    nc = perm.loc[perm["new_construction"] == 1, ["parcel_number", "permit_year"]]
    ppts = nc.merge(geo_ref, on="parcel_number", how="inner").dropna(subset=["x", "y"])
    ppts = ppts.rename(columns={"permit_year": "year"})
    pself = ppts.groupby(["parcel_number", "year"]).size().rename("n").reset_index()

    sv = viol.loc[viol["severe"], ["parcel_number", "violation_year"]]
    spts = sv.merge(geo_ref, on="parcel_number", how="inner").dropna(subset=["x", "y"])
    spts = spts.rename(columns={"violation_year": "year"})
    sself = spts.groupby(["parcel_number", "year"]).size().rename("n").reset_index()

    results = {}
    for year in YEARS:
        lag = panel.loc[panel["risk_year"] == year - 3].set_index(
            "parcel_number")["neighbor_median_value"]
        results[year] = build_year(year, assess, demo, geo, perm, viol,
                                   dpts, ppts[["year", "x", "y"]], pself,
                                   spts[["year", "x", "y"]], sself, lag)

    # ---------------------------------------------------- parity ----------
    print("\n" + "=" * 72)
    print("PARITY CHECK: rebuilt 2024 against the training panel")
    print("=" * 72)
    check_feats = [
        "log_market_value", "land_share", "land_to_improvement_ratio",
        "market_value_change_1yr", "market_value_change_3yr",
        "land_value_change_3yr", "market_value_ratio_to_neighbors",
        "neighbor_value_growth_3yr", "prior_demos_300ft_3yr",
        "prior_demos_1000ft_3yr", "neighbor_new_construction_300ft_3yr",
        "neighbor_severe_violations_300ft_3yr", "permit_count_3yr",
        "investment_count_3yr", "years_since_last_permit",
        "make_safe_count_3yr", "lot_line_count_3yr",
        "violation_case_count_3yr", "severe_violation_case_count_3yr",
        "years_since_last_violation", "ever_violation_flag",
        "vacant_structure_violation_case_count_3yr",
        "structural_violation_case_count_3yr",
        "sanitation_violation_case_count_3yr",
        "consecutive_years_vacant_violation",
        "years_since_first_vacant_violation",
    ]
    truth = pd.read_parquet(
        BASE / "philly_panel_v2.parquet",
        columns=["parcel_number", "risk_year"] + check_feats,
    )
    truth = truth[truth["risk_year"] == PARITY_YEAR]
    mine = results[PARITY_YEAR]
    m = truth.merge(mine[["parcel_number"] + check_feats], on="parcel_number",
                    how="inner", suffixes=("_panel", "_rebuilt"))
    rows = []
    for f in check_feats:
        a, b = m[f + "_panel"], m[f + "_rebuilt"]
        both_na = a.isna() & b.isna()
        close = np.isclose(a.fillna(-9e9), b.fillna(-9e9), rtol=1e-4, atol=1e-4)
        agree = int((close | both_na).sum())
        rows.append({"feature": f, "n": len(m), "agree": agree,
                     "agree_pct": round(100 * agree / len(m), 4),
                     "max_abs_diff": float(np.nanmax(np.abs(a - b))) if agree < len(m) else 0.0})
    parity = pd.DataFrame(rows).sort_values("agree_pct")
    parity.to_csv(BASE / "scoring_table_parity_check.csv", index=False)
    print(parity.to_string(index=False))
    print(f"\n  rows compared: {len(m):,}   "
          f"features reproducing exactly: "
          f"{int((parity.agree_pct >= 99.99).sum())}/{len(parity)}")

    # ---------------------------------------------------- output ----------
    out = results[TARGET_YEAR].copy()
    out = out.drop(columns=[c for c in ["demo_start_year"] if c in out.columns])
    out.to_parquet(BASE / "risk_year_2025_scoring_table.parquet", index=False)
    note("2025 scoring table rows", len(out))
    note("2025 rows carrying a building", int(out["has_building"].sum()))

    model_cols = ["parcel_number", "risk_year", "has_building"] + check_feats
    out[[c for c in model_cols if c in out.columns]].to_csv(
        BASE / "risk_year_2025_scoring_table.csv", index=False)

    pd.DataFrame(audit).to_csv(BASE / "scoring_table_build_report.csv", index=False)
    print(f"\nWrote 4 files to:\n{BASE}")


if __name__ == "__main__":
    main()
