from __future__ import annotations

"""
evaluate_week_against_actual.py
-------------------------------------------------------------
Purpose
-------
Create a fair week-level comparison between:
1. actual operations on the same modeled bin subset, and
2. the model's weekly plan + daily routing outputs

This avoids comparing a partial modeled subset against all actual campus pickups.

Inputs
------
Expected files:
- data/interim/collections_merged.parquet
- data/processed/small_instance_service_schedule.csv
- data/processed/small_instance_planning_summary.csv
- data/processed/daily_route_plan.csv

Outputs
-------
- data/processed/weekly_actual_vs_model_summary.csv
- data/processed/weekly_actual_vs_model_modeled_subset.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "interim": interim_dir, "processed": processed_dir}


def canonical_stream(x: object) -> str:
    s = str(x).strip().lower()
    lookup = {
        "compostables": "Compostables",
        "compost": "Compostables",
        "waste": "Waste",
        "landfill": "Waste",
        "bottles/cans": "Bottles/Cans",
        "bottles & cans": "Bottles/Cans",
        "recycling": "Bottles/Cans",
        "single stream": "Bottles/Cans",
    }
    return lookup.get(s, str(x).strip() if str(x).strip() else "Unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare actual weekly operations to model outputs on the same modeled bin subset.")
    parser.add_argument("--actual-start-date", type=str, required=True, help="Actual week start date, YYYY-MM-DD")
    parser.add_argument("--actual-end-date", type=str, required=True, help="Actual week end date, YYYY-MM-DD")
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    actual_fp = paths["interim"] / "collections_merged.parquet"
    sched_fp = paths["processed"] / "small_instance_service_schedule.csv"
    plan_fp = paths["processed"] / "small_instance_planning_summary.csv"
    route_fp = paths["processed"] / "daily_route_plan.csv"

    if not actual_fp.exists():
        raise FileNotFoundError("Missing collections_merged.parquet. Run run_pipeline.py first.")
    if not sched_fp.exists():
        raise FileNotFoundError("Missing small_instance_service_schedule.csv. Run solve_7day_schedule.py first.")
    if not plan_fp.exists():
        raise FileNotFoundError("Missing small_instance_planning_summary.csv. Run solve_7day_schedule.py first.")
    if not route_fp.exists():
        raise FileNotFoundError("Missing daily_route_plan.csv. Run solve_daily_routing.py first.")

    actual = pd.read_parquet(actual_fp)
    sched = pd.read_csv(sched_fp)
    planning = pd.read_csv(plan_fp)
    routes = pd.read_csv(route_fp)

    # -----------------------------
    # Clean actual data
    # -----------------------------
    actual["Collection_Time"] = pd.to_datetime(actual["Collection_Time"], errors="coerce")
    actual = actual[actual["Collection_Time"].notna()].copy()
    actual["date"] = actual["Collection_Time"].dt.floor("D")

    if "Stream_Type" in actual.columns:
        actual["stream"] = actual["Stream_Type"].apply(canonical_stream)
    else:
        actual["stream"] = "Unknown"

    if "Serial" not in actual.columns:
        raise KeyError("collections_merged.parquet must contain Serial for subset comparison.")

    actual["Serial"] = actual["Serial"].astype(str).str.strip()

    # -----------------------------
    # Clean model outputs
    # -----------------------------
    sched["Serial"] = sched["Serial"].astype(str).str.strip()
    sched["stream"] = sched["stream"].apply(canonical_stream)
    routes["stream"] = routes["stream"].apply(canonical_stream)

    start = pd.to_datetime(args.actual_start_date)
    end = pd.to_datetime(args.actual_end_date)

    # All actual operations in the chosen week
    actual_week_all = actual[(actual["date"] >= start) & (actual["date"] <= end)].copy()

    # Modeled bin subset = unique bins that appear in the model schedule
    modeled_bins = set(sched["Serial"].dropna().astype(str).str.strip().unique().tolist())

    # Actual operations restricted to the same modeled bin subset
    actual_week_subset = actual_week_all[actual_week_all["Serial"].isin(modeled_bins)].copy()

    # -----------------------------
    # Summaries: actual subset
    # -----------------------------
    actual_subset_summary = (
        actual_week_subset.groupby("stream")
        .size()
        .rename("actual_pickups_same_subset")
        .reset_index()
    )

    actual_subset_bin_counts = pd.DataFrame(
        [{"modeled_bin_count": len(modeled_bins), "actual_pickups_same_subset_total": len(actual_week_subset)}]
    )

    # -----------------------------
    # Summaries: model schedule
    # -----------------------------
    model_sched_summary = (
        sched.groupby("stream")
        .agg(
            model_pickups=("Serial", "count"),
            model_pickup_gal=("pickup_gal", "sum"),
            model_pickup_lb=("pickup_lb", "sum"),
            modeled_unique_bins=("Serial", "nunique"),
        )
        .reset_index()
    )

    # -----------------------------
    # Summaries: model routing
    # -----------------------------
    model_route_summary = (
        routes.groupby("stream")
        .agg(
            model_route_count=("truck", "count"),
            model_route_gal=("route_gal", "sum"),
            model_route_lb=("route_lb", "sum"),
            model_route_min=("route_minutes", "sum"),
        )
        .reset_index()
    )

    # -----------------------------
    # Merge stream-level comparison
    # -----------------------------
    summary = actual_subset_summary.merge(model_sched_summary, on="stream", how="outer")
    summary = summary.merge(model_route_summary, on="stream", how="outer")
    summary = summary.fillna(0)

    # -----------------------------
    # Add overall row
    # -----------------------------
    overall = pd.DataFrame([
        {
            "stream": "OVERALL",
            "actual_pickups_same_subset": float(summary["actual_pickups_same_subset"].sum()),
            "model_pickups": float(summary["model_pickups"].sum()),
            "model_pickup_gal": float(summary["model_pickup_gal"].sum()),
            "model_pickup_lb": float(summary["model_pickup_lb"].sum()),
            "modeled_unique_bins": float(summary["modeled_unique_bins"].sum()),
            "model_route_count": float(summary["model_route_count"].sum()),
            "model_route_gal": float(summary["model_route_gal"].sum()),
            "model_route_lb": float(summary["model_route_lb"].sum()),
            "model_route_min": float(summary["model_route_min"].sum()),
        }
    ])

    planning_cols = [
        "status",
        "objective_value",
        "horizon_days",
        "num_bins_in_instance",
        "num_trucks",
        "truck_work_min",
        "total_pickups",
        "total_extra_dumps",
        "total_overtime_min",
        "overflow_bin_days",
        "cbc_time_limit_sec",
        "cbc_gap_rel",
    ]
    for col in planning_cols:
        if col not in planning.columns:
            planning[col] = None

    planning_one = planning[planning_cols].head(1).copy()
    for col in planning_cols:
        overall[col] = planning_one.iloc[0][col]

    out = pd.concat([summary, overall], ignore_index=True)

    # -----------------------------
    # Save outputs
    # -----------------------------
    out_fp = paths["processed"] / "weekly_actual_vs_model_summary.csv"
    subset_fp = paths["processed"] / "weekly_actual_vs_model_modeled_subset.csv"
    out.to_csv(out_fp, index=False)
    actual_week_subset.to_csv(subset_fp, index=False)

    print(f"[OK] Wrote: {out_fp}")
    print(f"[OK] Wrote: {subset_fp}")
    print("\nStream-level comparison on modeled bin subset:")
    print(out.to_string(index=False))
    print("\nModeled subset info:")
    print(actual_subset_bin_counts.to_string(index=False))


if __name__ == "__main__":
    main()