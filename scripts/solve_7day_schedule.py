# scripts/solve_7day_schedule.py
# -------------------------------------------------------------
# solve_7day_schedule.py
#
# Purpose
# -------
# Solve a small deterministic 7-day Bigbelly scheduling instance
# using the projected-fill input table produced by
# build_projected_fill.py.
#
# The model decides:
#   1. which bins to service,
#   2. on which day in the 7-day horizon to service them,
#   3. which truck serves each selected bin,
#   4. which waste stream each truck handles on each day.
#
# The formulation enforces:
#   - service by the required deadline,
#   - at most one service per bin within the horizon,
#   - stream compatibility between bins and trucks,
#   - daily truck capacity limits,
#   - daily truck workload limits.
#
# Inputs
# ------
#   data/processed/bin_7day_projection_inputs.csv
#   or
#   data/processed/bin_7day_projection_inputs.parquet
#
# Outputs
# -------
#   data/processed/small_instance_service_schedule.csv
#   data/processed/small_instance_truck_streams.csv
#   data/processed/small_instance_truck_load_check.csv
#
# Notes
# -----
# This script solves a small prototype instance rather than the full
# operational problem. It is intended to demonstrate that the 7-day
# scheduling formulation is implementable and produces feasible
# assignments under capacity, workload, and stream constraints.
# -------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

"""
    Return the project root directory.

    Assumes this script is stored in:
        project_root/scripts/solve_7day_schedule.py
"""
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

 """
    Ensure the processed-data folder exists and return its path.
 """
def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}

"""
    Select a small representative prototype instance from the full
    projected-fill input table.

    Logic
    -----
    - If required_only=True, keep only bins that must be serviced
      within the horizon.
    - Separate bins into:
        a) required bins (finite service deadline)
        b) optional bins (no service deadline in the horizon)
    - If max_bins is provided, spread selections across deadlines
      0 through 6 first so the example illustrates a full 7-day
      schedule rather than only urgent day-0 bins.
    - Fill remaining slots using other required bins first, then
      optional bins if necessary.

    Returns
    -------
    pd.DataFrame
        A reduced input table used for the prototype solve.
"""
def choose_instance(df: pd.DataFrame, max_bins: int | None, required_only: bool) -> pd.DataFrame:
    out = df.copy()
    
    # Optionally restrict to bins that must be serviced in the horizon
    if required_only:
        out = out[out["must_service_within_horizon"]].copy()

    # Ensure service_deadline exists
    if "service_deadline" not in out.columns:
        out["service_deadline"] = np.nan

    # Separate required and optional bins
    required = out[out["service_deadline"].notna()].copy()
    optional = out[out["service_deadline"].isna()].copy()

    # If no required bins exist, fall back to optional bins
    if required.empty:
        chosen = optional.copy()
        if max_bins is not None:
            chosen = chosen.head(max_bins)
        return chosen.reset_index(drop=True)

    required["service_deadline"] = required["service_deadline"].astype(int)

    picks = []

    # If no size limit is provided, return all required bins sorted by urgency
    if max_bins is None:
        return required.sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

    # Spread selections across deadlines 0..6 first
    per_deadline = max(1, max_bins // 7)

    for d in range(7):
        bucket = required[required["service_deadline"] == d].sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True]
        )
        picks.append(bucket.head(per_deadline))

    chosen = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["Serial"])

    # Fill remaining slots with other required bins first
    if len(chosen) < max_bins:
        remaining_required = required[~required["Serial"].isin(chosen["Serial"])].sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True]
        )
        need = max_bins - len(chosen)
        chosen = pd.concat([chosen, remaining_required.head(need)], ignore_index=True)

    # Only if still short, add optional bins
    if len(chosen) < max_bins:
        remaining_optional = optional.sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True]
        )
        need = max_bins - len(chosen)
        chosen = pd.concat([chosen, remaining_optional.head(need)], ignore_index=True)

    return chosen.reset_index(drop=True)

 """
    Solve the 7-day prototype scheduling instance.

    Workflow
    --------
    1. Parse runtime arguments.
    2. Load the projected-fill input table.
    3. Validate required columns.
    4. Select a small prototype instance.
    5. Build the optimization model.
    6. Solve the model with CBC.
    7. Export and print the resulting schedule, truck-stream plan,
       and truck load checks.
 """

def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a small 7-day Bigbelly scheduling instance.")
    parser.add_argument("--input", type=str, default=None, help="Path to bin_7day_projection_inputs.parquet or .csv")
    parser.add_argument("--max-bins", type=int, default=12, help="Small instance size for prototype solve")
    parser.add_argument("--num-trucks", type=int, default=2)
    parser.add_argument("--truck-capacity-gal", type=float, default=300.0)
    parser.add_argument("--truck-work-min", type=float, default=180.0)
    parser.add_argument("--required-only", action="store_true", help="Only solve for bins with a finite deadline")
    args = parser.parse_args()

    # Resolve project folders
    root = repo_root()
    paths = ensure_dirs(root)

    # Choose the projected-fill input file
    if args.input:
        input_fp = Path(args.input)
    else:
        input_fp = paths["processed"] / "bin_7day_projection_inputs.parquet"

    # Fallback to CSV if Parquet is not available
    if not input_fp.exists():
        csv_fp = paths["processed"] / "bin_7day_projection_inputs.csv"
        if csv_fp.exists():
            input_fp = csv_fp
        else:
            raise FileNotFoundError(
                "Missing bin_7day_projection_inputs. Run python scripts/build_projected_fill.py first."
            )

    # Load input file
    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)
    
    # Verify required modeling columns are present 
    required_cols = [
        "Serial",
        "stream",
        "threshold_pct",
        "days_since_last_service",
        "bin_capacity_gal",
        "avg_service_min",
        "avg_travel_proxy_min",
        "must_service_within_horizon",
    ] + [f"fill_day_{d}" for d in range(7)]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input file missing columns: {missing}")

    # Ensure deadline and required flag exist in usable form
    if "service_deadline" not in df.columns:
        df["service_deadline"] = np.nan

    # Select the small prototype instance to solve
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)

    instance = choose_instance(df, max_bins=args.max_bins, required_only=args.required_only)

    if instance.empty:
        raise ValueError("No bins available for solve after filtering.")

    # Sets used by the optimization model
    bins = instance["Serial"].astype(str).tolist()
    streams = sorted(instance["stream"].dropna().astype(str).unique().tolist())
    days = list(range(7))
    trucks = [f"T{k+1}" for k in range(args.num_trucks)]

    # Parameter dictionaries for easier use in the optimization model
    bin_stream = dict(zip(instance["Serial"], instance["stream"]))
    service_deadline = dict(zip(instance["Serial"], instance["service_deadline"]))
    service_min = dict(zip(instance["Serial"], instance["avg_service_min"]))
    travel_min = dict(zip(instance["Serial"], instance["avg_travel_proxy_min"]))
    bin_capacity_gal = dict(zip(instance["Serial"], instance["bin_capacity_gal"]))

    # Build projected fill and projected pickup volume by bin and day
    projected_fill = {}
    pickup_gal = {}

    for _, row in instance.iterrows():
        i = row["Serial"]
        for d in days:
            pct = float(row[f"fill_day_{d}"])
            projected_fill[(i, d)] = pct
            pickup_gal[(i, d)] = float(bin_capacity_gal[i]) * pct / 100.0
    
    # Create the optimization model
    model = LpProblem("Bigbelly_7Day_Schedule", LpMinimize)

    # Decision variables
    # x[i,d] = 1 if bin i is serviced on day d
    # y[i,k,d] = 1 if truck k services bin i on day d
    # z[k,s,d] = 1 if truck k is assigned to stream s on day d
    x = LpVariable.dicts("x", (bins, days), cat=LpBinary)
    y = LpVariable.dicts("y", (bins, trucks, days), cat=LpBinary)
    z = LpVariable.dicts("z", (trucks, streams, days), cat=LpBinary)

    # Objective:
    # minimize the number of pickups, with a very small tie-breaker
    # that prefers later service when multiple choices are feasible
    epsilon = 0.001
    model += (
        lpSum(x[i][d] for i in bins for d in days)
        + epsilon * lpSum((6 - d) * x[i][d] for i in bins for d in days)
    )

    # Constraint 1:
    # At most one service per bin within the 7-day horizon
    for i in bins:
        model += lpSum(x[i][d] for d in days) <= 1, f"at_most_once_{i}"

    # Constraint 2:
    # If a bin has a deadline, it must be serviced by that day
    for i in bins:
        deadline = service_deadline[i]
        if pd.notna(deadline):
            deadline = int(deadline)
            model += lpSum(x[i][d] for d in range(deadline + 1)) >= 1, f"deadline_{i}"

    # Constraint 3:
    # If a bin is serviced on a day, exactly one truck must be assigned
    for i in bins:
        for d in days:
            model += lpSum(y[i][k][d] for k in trucks) == x[i][d], f"truck_assign_{i}_{d}"

    # Constraint 4:
    # A truck may be assigned to at most one stream on a given day
    for k in trucks:
        for d in days:
            model += lpSum(z[k][s][d] for s in streams) <= 1

    # Constraint 5:
    # A truck can only serve a bin if that truck is assigned to the bin's stream
    for i in bins:
        s_i = bin_stream[i]
        for k in trucks:
            for d in days:
                model += y[i][k][d] <= z[k][s_i][d], f"compat_{i}_{k}_{d}"

    # Constraint 6:
    # Daily truck capacity limit in gallons
    for k in trucks:
        for d in days:
            model += (
                lpSum(pickup_gal[(i, d)] * y[i][k][d] for i in bins) <= args.truck_capacity_gal
            ), f"capacity_{k}_{d}"

    # Constraint 7:
    # Daily truck workload limit in minutes
    for k in trucks:
        for d in days:
            model += (
                lpSum((service_min[i] + travel_min[i]) * y[i][k][d] for i in bins) <= args.truck_work_min
            ), f"workload_{k}_{d}"

    # Solve the model using CBC
    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = LpStatus[model.status]
    print(f"[STATUS] {status}")

    if status != "Optimal":
        print("Model did not solve to optimality. Try increasing truck capacity, work minutes, or reducing max_bins.")
        return
    # ---------------------------------------------------------
    # Build output table 1: service schedule by bin
    # ---------------------------------------------------------
    schedule_rows = []
    for i in bins:
        picked = False
        for d in days:
            if value(x[i][d]) > 0.5:
                picked = True
                truck = None
                for k in trucks:
                    if value(y[i][k][d]) > 0.5:
                        truck = k
                        break
                schedule_rows.append(
                    {
                        "Serial": i,
                        "stream": bin_stream[i],
                        "service_day": d,
                        "truck": truck,
                        "projected_fill_pct_that_day": round(projected_fill[(i, d)], 2),
                        "projected_pickup_gal": round(pickup_gal[(i, d)], 2),
                        "service_deadline": None if pd.isna(service_deadline[i]) else int(service_deadline[i]),
                    }
                )
        if not picked:
            schedule_rows.append(
                {
                    "Serial": i,
                    "stream": bin_stream[i],
                    "service_day": None,
                    "truck": None,
                    "projected_fill_pct_that_day": None,
                    "projected_pickup_gal": None,
                    "service_deadline": None if pd.isna(service_deadline[i]) else int(service_deadline[i]),
                }
            )
    # ---------------------------------------------------------
    # Build output table 2: truck stream assignments by day
    # ---------------------------------------------------------
    truck_stream_rows = []
    for d in days:
        for k in trucks:
            chosen_stream = None
            for s in streams:
                if value(z[k][s][d]) > 0.5:
                    chosen_stream = s
                    break
            truck_stream_rows.append({"day": d, "truck": k, "assigned_stream": chosen_stream})
    # ---------------------------------------------------------
    # Build output table 3: truck load / workload checks
    # ---------------------------------------------------------
    load_rows = []
    for d in days:
        for k in trucks:
            total_gal = sum(pickup_gal[(i, d)] * value(y[i][k][d]) for i in bins)
            total_min = sum((service_min[i] + travel_min[i]) * value(y[i][k][d]) for i in bins)
            load_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "pickup_gal_used": round(total_gal, 2),
                    "pickup_gal_capacity": args.truck_capacity_gal,
                    "minutes_used": round(total_min, 2),
                    "minutes_capacity": args.truck_work_min,
                }
            )
    # Convert outputs to dataframes
    schedule_df = pd.DataFrame(schedule_rows).sort_values(["service_day", "Serial"], na_position="last")
    truck_stream_df = pd.DataFrame(truck_stream_rows)
    load_df = pd.DataFrame(load_rows)

    # Output file paths
    schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
    stream_fp = paths["processed"] / "small_instance_truck_streams.csv"
    load_fp = paths["processed"] / "small_instance_truck_load_check.csv"

    # Save output files
    schedule_df.to_csv(schedule_fp, index=False)
    truck_stream_df.to_csv(stream_fp, index=False)
    load_df.to_csv(load_fp, index=False)

    # Print summary information
    print(f"[OK] Objective = {value(model.objective):.3f}")
    print(f"[OK] Wrote: {schedule_fp}")
    print(f"[OK] Wrote: {stream_fp}")
    print(f"[OK] Wrote: {load_fp}")

    # Print detailed outputs
    print("\nService schedule:")
    print(schedule_df.to_string(index=False))

    print("\nTruck stream assignments:")
    print(truck_stream_df.to_string(index=False))

    print("\nTruck load check:")
    print(load_df.to_string(index=False))


if __name__ == "__main__":
    main()
