from __future__ import annotations

"""
solve_7day_schedule.py
-------------------------------------------------------------
Purpose
-------
Solve a 7-day Bigbelly scheduling instance without geographic zoning.

This version is designed to reflect Professor Yano's feedback:
1. Do NOT divide campus into fixed truck zones.
2. Allow a bin to be serviced more than once within the 7-day horizon.
3. Use inventory-style state variables that:
   - represent the fill in each bin each day,
   - drop to zero when the bin is picked up,
   - then increase again according to the bin's growth rate.
4. Enforce realistic truck constraints:
   - one stream per truck per day,
   - mass limit,
   - volume limit,
   - shift-time limit,
   - extra dump cycles when needed.

Core modeling idea
------------------
For each bin i and day d:
- inventory_gal[i,d] = start-of-day fill in gallons
- if the bin is serviced on day d, that inventory is removed
- post_service_inventory_gal[i,d] = remaining fill after service
- next day's inventory = post-service inventory + daily growth

This makes repeated service possible within the same 7-day horizon.

Inputs
------
Expected input file:
    data/processed/bin_7day_projection_inputs.parquet
or:
    data/processed/bin_7day_projection_inputs.csv

Required columns in the input:
- Serial
- stream
- threshold_pct
- days_since_last_service
- current_fill_pct_est
- daily_fill_growth_pct
- bin_capacity_gal
- density_lb_per_gal
- avg_service_min
- avg_travel_proxy_min
- must_service_within_horizon

Outputs
-------
- small_instance_service_schedule.csv
- small_instance_truck_streams.csv
- small_instance_truck_load_check.csv
- small_instance_inventory_trajectory.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)

# -------------------------------------------------------------
# Global modeling constant
# -------------------------------------------------------------
# We allow inventory to go above 100% up to 125% so the model can
# represent overflow states in a bounded way, consistent with the report.
MAX_FILL_FACTOR = 1.25


# -------------------------------------------------------------
# Helper: repo root
# -------------------------------------------------------------
def repo_root() -> Path:
    """
    Return the project root directory.

    Assumes this file lives in:
        project_root/scripts/solve_7day_schedule.py
    """
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------------------
# Helper: ensure processed folder exists
# -------------------------------------------------------------
def ensure_dirs(root: Path) -> dict[str, Path]:
    """
    Ensure the processed-data folder exists and return its path.
    """
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


# -------------------------------------------------------------
# Helper: clean stream labels
# -------------------------------------------------------------
def canonical_stream(x: object) -> str:
    """
    Standardize stream names to a small consistent set.

    Examples
    --------
    'compost'       -> 'Compostables'
    'landfill'      -> 'Waste'
    'recycling'     -> 'Bottles/Cans'
    'single stream' -> 'Bottles/Cans'
    """
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


# -------------------------------------------------------------
# Helper: compute the day by which the bin must be serviced due
# to the 7-day hygiene rule, based on current days-since-service
# -------------------------------------------------------------
def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    """
    Compute the first day in the horizon by which the bin must be
    serviced due to the 7-day rule, based on the current age since
    last service at the start of day 0.

    Examples
    --------
    days_since_last_service = 6 -> must be serviced by day 1
    days_since_last_service = 7 -> must be serviced by day 0
    days_since_last_service = 1 -> deadline day 6
    """
    remaining = 7 - float(days_since_last_service)
    if remaining <= 0:
        return 0.0
    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))
    return np.nan


def safe_value(var: object) -> float:
    """Return a numeric value from a PuLP variable, defaulting None to 0."""
    v = value(var)
    return 0.0 if v is None else float(v)


# -------------------------------------------------------------
# Helper: pick a small prototype instance for demonstration
# -------------------------------------------------------------
def choose_instance(df: pd.DataFrame, max_bins: int | None, required_only: bool, horizon_days: int) -> pd.DataFrame:
    """
    Select a small representative prototype instance from the full
    projected-fill input table.

    Logic
    -----
    - If required_only=True, keep only bins that must be serviced
      within the current horizon.
    - If max_bins is set, try to spread chosen bins across deadlines
      0..6 so the toy example is more informative.
    """
    out = df.copy()

    if required_only:
        out = out[out["must_service_within_horizon"]].copy()

    if "service_deadline" not in out.columns:
        out["service_deadline"] = np.nan

    required = out[out["service_deadline"].notna()].copy()
    optional = out[out["service_deadline"].isna()].copy()

    # If no required bins exist, just take optional ones.
    if required.empty:
        chosen = optional.copy()
        if max_bins is not None:
            chosen = chosen.head(max_bins)
        return chosen.reset_index(drop=True)

    required["service_deadline"] = required["service_deadline"].astype(int)

    # If no size limit, return all required bins sorted by urgency.
    if max_bins is None:
        return required.sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    picks = []
    per_deadline = max(1, max_bins // max(1, horizon_days))

    # First, try to grab some bins from each deadline bucket.
    for d in range(horizon_days):
        bucket = required[required["service_deadline"] == d].sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        picks.append(bucket.head(per_deadline))

    chosen = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["Serial"])

    # Then fill remaining slots with other required bins.
    if len(chosen) < max_bins:
        remaining_required = required[~required["Serial"].isin(chosen["Serial"])].sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        )
        chosen = pd.concat(
            [chosen, remaining_required.head(max_bins - len(chosen))],
            ignore_index=True,
        )

    # If still short, use optional bins.
    if len(chosen) < max_bins:
        remaining_optional = optional.sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        chosen = pd.concat(
            [chosen, remaining_optional.head(max_bins - len(chosen))],
            ignore_index=True,
        )

    return chosen.reset_index(drop=True)


# -------------------------------------------------------------
# Main solve routine
# -------------------------------------------------------------
def main() -> None:
    """
    Workflow
    --------
    1. Parse runtime arguments.
    2. Load the projected-fill input file.
    3. Validate required columns.
    4. Select a small prototype instance.
    5. Build the MILP.
    6. Solve with CBC.
    7. Export schedules, truck plans, load checks, and inventory paths.
    """
    parser = argparse.ArgumentParser(description="Solve a 7-day Bigbelly scheduling instance.")
    parser.add_argument("--input", type=str, default=None, help="Path to bin_7day_projection_inputs.parquet or .csv")
    parser.add_argument("--max-bins", type=int, default=None, help="Optional instance size cap; default keeps all eligible bins")
    parser.add_argument("--num-trucks", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=7, help="Planning horizon length in days")

    # Truck capacity parameters
    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)

    # Time / labor parameters
    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--dump-turnaround-min", type=float, default=17.0)
    parser.add_argument("--max-extra-dumps", type=int, default=8)
    parser.add_argument("--max-overtime-min", type=float, default=180.0, help="Maximum overtime allowed per truck-day")
    parser.add_argument("--overtime-penalty-per-min", type=float, default=0.02, help="Penalty per overtime minute in the planning objective")
    parser.add_argument("--cbc-time-limit-sec", type=int, default=180, help="CBC solve time limit in seconds")
    parser.add_argument("--cbc-gap-rel", type=float, default=0.05, help="Relative MIP gap tolerance for CBC")

    # Optional filter
    parser.add_argument("--required-only", action="store_true")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Resolve directories and input file
    # ---------------------------------------------------------
    root = repo_root()
    paths = ensure_dirs(root)

    if args.input:
        input_fp = Path(args.input)
    else:
        input_fp = paths["processed"] / "bin_7day_projection_inputs.parquet"

    if not input_fp.exists():
        csv_fp = paths["processed"] / "bin_7day_projection_inputs.csv"
        if csv_fp.exists():
            input_fp = csv_fp
        else:
            raise FileNotFoundError(
                "Missing bin_7day_projection_inputs. Run python scripts/build_projected_fill.py first."
            )

    # ---------------------------------------------------------
    # Load input file
    # ---------------------------------------------------------
    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)

    # ---------------------------------------------------------
    # Validate input schema
    # ---------------------------------------------------------
    required_cols = [
        "Serial",
        "stream",
        "threshold_pct",
        "days_since_last_service",
        "current_fill_pct_est",
        "daily_fill_growth_pct",
        "bin_capacity_gal",
        "density_lb_per_gal",
        "avg_service_min",
        "avg_travel_proxy_min",
        "must_service_within_horizon",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input file missing columns: {missing}")

    # If these fields are missing, reconstruct what we can.
    if "service_deadline" not in df.columns:
        df["service_deadline"] = np.nan
    if "deadline_interval" not in df.columns:
        df["deadline_interval"] = df["days_since_last_service"].apply(lambda x: interval_deadline(x, args.horizon_days))

    df["stream"] = df["stream"].apply(canonical_stream)
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)

    # ---------------------------------------------------------
    # Choose a smaller prototype instance for solve
    # ---------------------------------------------------------
    instance = choose_instance(df, max_bins=args.max_bins, required_only=args.required_only, horizon_days=args.horizon_days)
    if instance.empty:
        raise ValueError("No bins available for solve after filtering.")

    # ---------------------------------------------------------
    # Sets
    # ---------------------------------------------------------
    bins = instance["Serial"].astype(str).tolist()
    streams = [s for s in sorted(instance["stream"].dropna().astype(str).unique().tolist()) if s != "Unknown"]
    days = list(range(args.horizon_days))
    trucks = [f"T{k + 1}" for k in range(args.num_trucks)]

    # ---------------------------------------------------------
    # Parameter dictionaries
    # ---------------------------------------------------------
    bin_stream = dict(zip(instance["Serial"], instance["stream"]))
    threshold_pct = dict(zip(instance["Serial"], instance["threshold_pct"]))
    deadline_interval = dict(zip(instance["Serial"], instance["deadline_interval"]))
    service_min = dict(zip(instance["Serial"], instance["avg_service_min"]))
    travel_min = dict(zip(instance["Serial"], instance["avg_travel_proxy_min"]))
    bin_capacity_gal = dict(zip(instance["Serial"], instance["bin_capacity_gal"]))
    density_lb_per_gal = dict(zip(instance["Serial"], instance["density_lb_per_gal"]))

    # Maximum allowed inventory in a bin, allowing up to 125% to represent overflow.
    max_inventory_gal = {
        i: float(bin_capacity_gal[i]) * MAX_FILL_FACTOR
        for i in bins
    }

    # Current fill in gallons at the start of day 0.
    current_fill_gal = {
        i: float(bin_capacity_gal[i])
        * float(instance.loc[instance["Serial"] == i, "current_fill_pct_est"].iloc[0]) / 100.0
        for i in bins
    }

    # Daily growth rate in gallons/day.
    growth_gal_per_day = {
        i: float(bin_capacity_gal[i])
        * float(instance.loc[instance["Serial"] == i, "daily_fill_growth_pct"].iloc[0]) / 100.0
        for i in bins
    }

    # Convert threshold percentages to threshold gallons.
    # For recyclables, threshold may be NaN, meaning no fill-based threshold rule.
    threshold_gal = {
        i: (
            np.nan
            if pd.isna(threshold_pct[i])
            else float(bin_capacity_gal[i]) * float(threshold_pct[i]) / 100.0
        )
        for i in bins
    }

    # Stream-specific effective volume caps for one truck-load cycle.
    stream_volume_cap = {
        "Waste": args.waste_volume_gal,
        "Compostables": args.compost_volume_gal,
        "Bottles/Cans": args.recycling_volume_gal,
    }

    # ---------------------------------------------------------
    # Build optimization model
    # ---------------------------------------------------------
    model = LpProblem("Bigbelly_7Day_Schedule_No_Zones", LpMinimize)

    # ---------------------------------------------------------
    # Decision variables
    # ---------------------------------------------------------
    # x[i,d] = 1 if bin i is serviced on day d
    x = LpVariable.dicts("x", (bins, days), cat=LpBinary)

    # y[i,k,d] = 1 if truck k services bin i on day d
    y = LpVariable.dicts("y", (bins, trucks, days), cat=LpBinary)

    # z[k,s,d] = 1 if truck k is assigned to stream s on day d
    z = LpVariable.dicts("z", (trucks, streams, days), cat=LpBinary)

    # extra_dumps[k,s,d] = number of additional dump cycles truck k makes on day d
    # while operating on stream s
    extra_dumps = LpVariable.dicts(
        "extra_dumps",
        (trucks, streams, days),
        lowBound=0,
        upBound=args.max_extra_dumps,
        cat=LpInteger,
    )

    # overtime_min[k,d] = overtime minutes used by truck k on day d
    overtime_min = LpVariable.dicts("overtime_min", (trucks, days), lowBound=0, upBound=args.max_overtime_min)

    # inventory_gal[i,d] = start-of-day fill in bin i on day d
    inventory_gal = LpVariable.dicts("inventory_gal", (bins, days), lowBound=0)

    # post_service_inventory_gal[i,d] = fill remaining after any service on day d
    post_service_inventory_gal = LpVariable.dicts("post_service_inventory_gal", (bins, days), lowBound=0)

    # pickup_gal[i,d] = amount removed from bin i on day d
    pickup_gal = LpVariable.dicts("pickup_gal", (bins, days), lowBound=0)

    # pickup_gal_truck[i,k,d] = amount of bin i picked up by truck k on day d
    # this links pickup quantity to truck-specific capacity constraints
    pickup_gal_truck = LpVariable.dicts("pickup_gal_truck", (bins, trucks, days), lowBound=0)

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------
    # Main goal: minimize number of pickups
    # Small tiebreaker: prefer later pickups when feasible
    # Small penalty on extra dumps to avoid unnecessary dump cycles
    epsilon = 0.001
    dump_penalty = 0.01

    model += (
        lpSum(x[i][d] for i in bins for d in days)
        + epsilon * lpSum((days[-1] - d) * x[i][d] for i in bins for d in days)
        + dump_penalty * lpSum(extra_dumps[k][s][d] for k in trucks for s in streams for d in days)
        + args.overtime_penalty_per_min * lpSum(overtime_min[k][d] for k in trucks for d in days)
    )

    # ---------------------------------------------------------
    # Bin inventory logic
    # ---------------------------------------------------------
    for i in bins:
        # Initial start-of-day inventory on day 0
        model += inventory_gal[i][0] == current_fill_gal[i], f"initial_inventory_{i}"
        model += inventory_gal[i][0] <= max_inventory_gal[i], f"initial_cap_{i}"

        # Initial 7-day hygiene rule:
        # if the bin is already close to the 7-day limit, force at least one pickup
        # by its initial due date.
        interval_due = deadline_interval[i]
        if pd.notna(interval_due):
            interval_due = int(interval_due)
            model += lpSum(x[i][d] for d in range(interval_due + 1)) >= 1, f"interval_deadline_{i}"

        for d in days:
            # Inventory must stay within allowed bounded overflow.
            model += inventory_gal[i][d] <= max_inventory_gal[i], f"inventory_cap_{i}_{d}"

            # Post-service inventory can never exceed start-of-day inventory.
            model += (
                post_service_inventory_gal[i][d] <= inventory_gal[i][d]
            ), f"post_le_inventory_{i}_{d}"

            # If x[i,d] = 1, then post-service inventory is forced to zero.
            model += (
                post_service_inventory_gal[i][d] <= max_inventory_gal[i] * (1 - x[i][d])
            ), f"reset_if_pick_{i}_{d}"

            # If x[i,d] = 0, then post-service inventory must equal start inventory.
            model += (
                post_service_inventory_gal[i][d] >= inventory_gal[i][d] - max_inventory_gal[i] * x[i][d]
            ), f"carry_if_no_pick_{i}_{d}"

            # Pickup amount = inventory before service - inventory after service
            model += (
                pickup_gal[i][d] == inventory_gal[i][d] - post_service_inventory_gal[i][d]
            ), f"pickup_def_{i}_{d}"

            # If there is no pickup, pickup_gal must be zero.
            model += (
                pickup_gal[i][d] <= max_inventory_gal[i] * x[i][d]
            ), f"pickup_activate_{i}_{d}"

            # Fill-threshold logic:
            # if the start-of-day fill exceeds the threshold, then x[i,d] must turn on.
            # Rearranged big-M style:
            # inventory <= threshold + M*x
            # So if x = 0, inventory must stay <= threshold.
            if pd.notna(threshold_gal[i]):
                model += (
                    inventory_gal[i][d] <= threshold_gal[i] + max_inventory_gal[i] * x[i][d]
                ), f"threshold_{i}_{d}"

            # Inventory flow from day d to day d+1:
            # next day's start inventory = today's post-service inventory + growth
            if d < days[-1]:
                model += (
                    inventory_gal[i][d + 1] == post_service_inventory_gal[i][d] + growth_gal_per_day[i]
                ), f"inventory_flow_{i}_{d}"

    # ---------------------------------------------------------
    # Truck assignment logic
    # ---------------------------------------------------------
    for i in bins:
        for d in days:
            # If bin i is serviced on day d, exactly one truck must service it.
            model += (
                lpSum(y[i][k][d] for k in trucks) == x[i][d]
            ), f"truck_assign_{i}_{d}"

            for k in trucks:
                # Truck-level picked quantity cannot exceed total picked quantity
                model += (
                    pickup_gal_truck[i][k][d] <= pickup_gal[i][d]
                ), f"truck_pickup_le_total_{i}_{k}_{d}"

                # If truck k is not assigned to bin i on day d, it cannot carry any of its pickup
                model += (
                    pickup_gal_truck[i][k][d] <= max_inventory_gal[i] * y[i][k][d]
                ), f"truck_pickup_activate_{i}_{k}_{d}"

                # If truck k is assigned, pickup quantity must be fully allocated to that truck
                model += (
                    pickup_gal_truck[i][k][d] >= pickup_gal[i][d] - max_inventory_gal[i] * (1 - y[i][k][d])
                ), f"truck_pickup_lower_{i}_{k}_{d}"

            # Total truck-level allocations must sum to total bin pickup
            model += (
                lpSum(pickup_gal_truck[i][k][d] for k in trucks) == pickup_gal[i][d]
            ), f"truck_pickup_balance_{i}_{d}"

    # ---------------------------------------------------------
    # One stream per truck per day
    # ---------------------------------------------------------
    for k in trucks:
        for d in days:
            model += lpSum(z[k][s][d] for s in streams) <= 1, f"one_stream_{k}_{d}"

    # ---------------------------------------------------------
    # Stream compatibility
    # ---------------------------------------------------------
    for i in bins:
        s_i = bin_stream[i]
        for k in trucks:
            for d in days:
                if s_i in streams:
                    model += y[i][k][d] <= z[k][s_i][d], f"compat_{i}_{k}_{d}"
                else:
                    model += y[i][k][d] == 0, f"unknown_stream_block_{i}_{k}_{d}"

    # ---------------------------------------------------------
    # Truck capacity constraints by stream and day
    # ---------------------------------------------------------
    for k in trucks:
        for s in streams:
            for d in days:
                # Total gallons assigned to truck k on stream s on day d
                assigned_gal = lpSum(
                    pickup_gal_truck[i][k][d]
                    for i in bins
                    if bin_stream[i] == s
                )

                # Total pounds assigned to truck k on stream s on day d
                assigned_lb = lpSum(
                    density_lb_per_gal[i] * pickup_gal_truck[i][k][d]
                    for i in bins
                    if bin_stream[i] == s
                )

                # Effective capacity allows 1 base load plus extra dump cycles
                model += (
                    assigned_gal <= stream_volume_cap[s] * (z[k][s][d] + extra_dumps[k][s][d])
                ), f"volcap_{k}_{s}_{d}"

                model += (
                    assigned_lb <= args.truck_mass_lb * (z[k][s][d] + extra_dumps[k][s][d])
                ), f"masscap_{k}_{s}_{d}"

                # Extra dumps only make sense if the truck is actually active on that stream-day
                model += (
                    extra_dumps[k][s][d] <= args.max_extra_dumps * z[k][s][d]
                ), f"dumpactivate_{k}_{s}_{d}"

    # ---------------------------------------------------------
    # Truck shift-time constraint
    # ---------------------------------------------------------
    for k in trucks:
        for d in days:
            # Time spent on bins assigned to this truck-day
            service_and_travel = lpSum(
                (service_min[i] + travel_min[i]) * y[i][k][d]
                for i in bins
            )

            # Time spent on dump turnarounds
            dump_minutes = args.dump_turnaround_min * lpSum(
                extra_dumps[k][s][d] for s in streams
            )

            model += (
                service_and_travel + dump_minutes <= args.truck_work_min + overtime_min[k][d]
            ), f"workload_{k}_{d}"

    # ---------------------------------------------------------
    # Solve the model
    # ---------------------------------------------------------
    solver = PULP_CBC_CMD(msg=False, timeLimit=args.cbc_time_limit_sec, gapRel=args.cbc_gap_rel)
    model.solve(solver)

    status = LpStatus[model.status]
    objective_value = safe_value(model.objective)
    print(f"[STATUS] {status}")
    print(f"[OBJECTIVE] {objective_value:.3f}")

    if status not in {"Optimal", "Not Solved", "Undefined", "Integer Feasible"}:
        print("Model did not find a usable feasible solution. Try increasing resources, overtime, or solver time.")
        return

    # ---------------------------------------------------------
    # Output 1: service schedule
    # ---------------------------------------------------------
    schedule_rows = []
    for i in bins:
        for d in days:
            if safe_value(x[i][d]) > 0.5:
                truck = None
                for k in trucks:
                    if safe_value(y[i][k][d]) > 0.5:
                        truck = k
                        break

                fill_pct_before = 100.0 * safe_value(inventory_gal[i][d]) / float(bin_capacity_gal[i])

                schedule_rows.append(
                    {
                        "Serial": i,
                        "stream": bin_stream[i],
                        "service_day": d,
                        "truck": truck,
                        "inventory_gal_before_service": round(safe_value(inventory_gal[i][d]), 2),
                        "inventory_pct_before_service": round(fill_pct_before, 2),
                        "pickup_gal": round(safe_value(pickup_gal[i][d]), 2),
                        "pickup_lb": round(safe_value(pickup_gal[i][d]) * density_lb_per_gal[i], 2),
                        "interval_deadline": None if pd.isna(deadline_interval[i]) else int(deadline_interval[i]),
                    }
                )

    # If no pickups occur, keep a placeholder row so the CSV is still readable.
    if not schedule_rows:
        schedule_rows.append(
            {
                "Serial": None,
                "stream": None,
                "service_day": None,
                "truck": None,
                "inventory_gal_before_service": None,
                "inventory_pct_before_service": None,
                "pickup_gal": None,
                "pickup_lb": None,
                "interval_deadline": None,
            }
        )

    # ---------------------------------------------------------
    # Output 2: truck stream assignment by day
    # ---------------------------------------------------------
    truck_stream_rows = []
    for d in days:
        for k in trucks:
            chosen_stream = None
            chosen_extra_dumps = 0
            for s in streams:
                if safe_value(z[k][s][d]) > 0.5:
                    chosen_stream = s
                    chosen_extra_dumps = int(round(safe_value(extra_dumps[k][s][d])))
                    break

            truck_stream_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "assigned_stream": chosen_stream,
                    "extra_dumps": chosen_extra_dumps,
                }
            )

    # ---------------------------------------------------------
    # Output 3: truck load / time feasibility checks
    # ---------------------------------------------------------
    load_rows = []
    for d in days:
        for k in trucks:
            total_gal = sum(safe_value(pickup_gal_truck[i][k][d]) for i in bins)
            total_lb = sum(density_lb_per_gal[i] * safe_value(pickup_gal_truck[i][k][d]) for i in bins)
            total_min = sum((service_min[i] + travel_min[i]) * safe_value(y[i][k][d]) for i in bins)

            assigned_stream = None
            extra_dump_count = 0
            volume_cap = None

            for s in streams:
                if safe_value(z[k][s][d]) > 0.5:
                    assigned_stream = s
                    extra_dump_count = int(round(safe_value(extra_dumps[k][s][d])))
                    volume_cap = stream_volume_cap[s] * (1 + extra_dump_count)
                    break

            load_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "assigned_stream": assigned_stream,
                    "pickup_gal_used": round(total_gal, 2),
                    "pickup_gal_capacity_effective": volume_cap,
                    "pickup_lb_used": round(total_lb, 2),
                    "pickup_lb_capacity_effective": args.truck_mass_lb * (1 + extra_dump_count),
                    "extra_dumps": extra_dump_count,
                    "minutes_used_without_dumps": round(total_min, 2),
                    "minutes_used_total": round(total_min + args.dump_turnaround_min * extra_dump_count, 2),
                    "overtime_min": round(safe_value(overtime_min[k][d]), 2),
                    "minutes_capacity_regular": args.truck_work_min,
                    "minutes_capacity_with_overtime": round(args.truck_work_min + safe_value(overtime_min[k][d]), 2),
                }
            )

    # ---------------------------------------------------------
    # Output 4: inventory trajectory by bin and day
    # ---------------------------------------------------------
    inventory_rows = []
    for i in bins:
        for d in days:
            inventory_rows.append(
                {
                    "Serial": i,
                    "stream": bin_stream[i],
                    "day": d,
                    "inventory_gal_start": round(value(inventory_gal[i][d]), 2),
                    "inventory_pct_start": round(
                        100.0 * safe_value(inventory_gal[i][d]) / float(bin_capacity_gal[i]), 2
                    ),
                    "pickup_flag": int(round(safe_value(x[i][d]))),
                    "pickup_gal": round(safe_value(pickup_gal[i][d]), 2),
                    "inventory_gal_after_service": round(safe_value(post_service_inventory_gal[i][d]), 2),
                }
            )

    # ---------------------------------------------------------
    # Output 5: planning summary
    # ---------------------------------------------------------
    total_pickups = sum(safe_value(x[i][d]) for i in bins for d in days)
    total_extra_dumps = sum(safe_value(extra_dumps[k][s][d]) for k in trucks for s in streams for d in days)
    total_overtime = sum(safe_value(overtime_min[k][d]) for k in trucks for d in days)
    overflow_bin_days = sum(1 for i in bins for d in days if safe_value(inventory_gal[i][d]) > float(bin_capacity_gal[i]) + 1e-6)
    planning_summary_df = pd.DataFrame([
        {
            "status": status,
            "objective_value": round(objective_value, 3),
            "horizon_days": args.horizon_days,
            "num_bins_in_instance": len(bins),
            "num_trucks": args.num_trucks,
            "truck_work_min": args.truck_work_min,
            "total_pickups": round(total_pickups, 2),
            "total_extra_dumps": round(total_extra_dumps, 2),
            "total_overtime_min": round(total_overtime, 2),
            "overflow_bin_days": int(overflow_bin_days),
            "cbc_time_limit_sec": args.cbc_time_limit_sec,
            "cbc_gap_rel": args.cbc_gap_rel,
        }
    ])

    # ---------------------------------------------------------
    # Convert outputs to DataFrames
    # ---------------------------------------------------------
    schedule_df = pd.DataFrame(schedule_rows).sort_values(["service_day", "Serial"], na_position="last")
    truck_stream_df = pd.DataFrame(truck_stream_rows)
    load_df = pd.DataFrame(load_rows)
    inventory_df = pd.DataFrame(inventory_rows)

    # ---------------------------------------------------------
    # Output file paths
    # ---------------------------------------------------------
    schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
    truck_stream_fp = paths["processed"] / "small_instance_truck_streams.csv"
    load_fp = paths["processed"] / "small_instance_truck_load_check.csv"
    inventory_fp = paths["processed"] / "small_instance_inventory_trajectory.csv"
    planning_summary_fp = paths["processed"] / "small_instance_planning_summary.csv"

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    schedule_df.to_csv(schedule_fp, index=False)
    truck_stream_df.to_csv(truck_stream_fp, index=False)
    load_df.to_csv(load_fp, index=False)
    inventory_df.to_csv(inventory_fp, index=False)
    planning_summary_df.to_csv(planning_summary_fp, index=False)

    print(f"[OK] Wrote: {schedule_fp}")
    print(f"[OK] Wrote: {truck_stream_fp}")
    print(f"[OK] Wrote: {load_fp}")
    print(f"[OK] Wrote: {inventory_fp}")
    print(f"[OK] Wrote: {planning_summary_fp}")

    print("\nSchedule preview:")
    print(schedule_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
