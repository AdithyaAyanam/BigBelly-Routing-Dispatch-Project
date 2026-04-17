from __future__ import annotations

"""
solve_7day_schedule.py
-------------------------------------------------------------
Approximate 7-day planning heuristic for Bigbelly service.

Why this version exists
-----------------------
This replaces the exact weekly MIP with a rolling-horizon constructive heuristic.
It is designed to be:
1. operationally credible,
2. easier to run on larger instances,
3. consistent with "no fixed campus zones",
4. compatible with the existing Phase 2 daily routing script.

Core idea
---------
For each day in a 7-day planning horizon:
- simulate each bin's fill state day by day,
- determine whether the bin is due because of threshold or 7-day age,
- score bins by urgency / overflow risk / pickup size / tiny-pickup penalty,
- build that day's service set greedily subject to approximate capacity,
- assign trucks by stream,
- reset serviced bins to zero and continue forward.

This script outputs:
- small_instance_service_schedule.csv
- small_instance_truck_streams.csv
- small_instance_truck_load_check.csv
- small_instance_inventory_trajectory.csv
- small_instance_planning_summary.csv

These outputs are designed to remain compatible with the existing routing layer.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_FILL_FACTOR = 1.25


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


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


def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    remaining = 7 - float(days_since_last_service)
    if remaining <= 0:
        return 0.0
    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))
    return np.nan

def load_stop_lookup_serials(root: Path) -> set[str]:
    fp = root / "data" / "processed" / "bin_stop_lookup.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing required file for routable filtering: {fp}"
        )
    lookup = pd.read_csv(fp)
    if "Serial" not in lookup.columns:
        raise KeyError("bin_stop_lookup.csv is missing required column 'Serial'")
    return set(lookup["Serial"].astype(str).str.strip())

def is_routable_for_routing(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    if {"Access_Lat", "Access_Lng"}.issubset(df.columns):
        mask = mask | (
            pd.to_numeric(df["Access_Lat"], errors="coerce").notna()
            & pd.to_numeric(df["Access_Lng"], errors="coerce").notna()
        )

    if {"Lat", "Lng"}.issubset(df.columns):
        mask = mask | (
            pd.to_numeric(df["Lat"], errors="coerce").notna()
            & pd.to_numeric(df["Lng"], errors="coerce").notna()
        )

    return mask


def choose_instance(
    df: pd.DataFrame,
    max_bins: int | None,
    required_only: bool,
    horizon_days: int,
) -> pd.DataFrame:
    out = df.copy()

    if required_only:
        out = out[out["must_service_within_horizon"]].copy()

    if "service_deadline" not in out.columns:
        out["service_deadline"] = np.nan

    required = out[out["service_deadline"].notna()].copy()
    optional = out[out["service_deadline"].isna()].copy()

    if required.empty:
        chosen = optional.copy()
        if max_bins is not None:
            chosen = chosen.head(max_bins)
        return chosen.reset_index(drop=True)

    required["service_deadline"] = required["service_deadline"].astype(int)

    if max_bins is None:
        return required.sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    picks = []
    per_deadline = max(1, max_bins // max(1, horizon_days))

    for d in range(horizon_days):
        bucket = required[required["service_deadline"] == d].sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        picks.append(bucket.head(per_deadline))

    chosen = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["Serial"])

    if len(chosen) < max_bins:
        remaining_required = required[~required["Serial"].isin(chosen["Serial"])].sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        )
        chosen = pd.concat(
            [chosen, remaining_required.head(max_bins - len(chosen))],
            ignore_index=True,
        )

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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def ceil_div_pos(x: float, y: float) -> int:
    if y <= 0:
        return 0
    return int(math.ceil(max(0.0, x) / y))


def earliest_due_day(
    fill_gal: float,
    days_since_service: float,
    growth_gal_per_day: float,
    threshold_gal: float | None,
    bin_capacity_gal: float,
    horizon_days: int,
) -> dict[str, Any]:
    """
    Determine earliest due day within the horizon based on:
    - threshold crossing, or
    - 7-day age rule.
    """
    threshold_day = None
    if threshold_gal is not None and not math.isnan(threshold_gal):
        for d in range(horizon_days):
            proj_fill = fill_gal + d * growth_gal_per_day
            if proj_fill >= threshold_gal - 1e-9:
                threshold_day = d
                break

    age_day = None
    remaining_age = 7.0 - days_since_service
    if remaining_age <= 0:
        age_day = 0
    else:
        age_int = int(math.floor(remaining_age))
        if 0 <= age_int < horizon_days:
            age_day = age_int

    due_candidates = [x for x in [threshold_day, age_day] if x is not None]
    due_day = min(due_candidates) if due_candidates else None

    overflow_day = None
    for d in range(horizon_days):
        proj_fill = fill_gal + d * growth_gal_per_day
        if proj_fill > bin_capacity_gal + 1e-9:
            overflow_day = d
            break

    return {
        "threshold_day": threshold_day,
        "age_day": age_day,
        "due_day": due_day,
        "overflow_day": overflow_day,
    }


def compute_priority(
    current_day: int,
    due_day: int | None,
    overflow_day: int | None,
    projected_pickup_gal: float,
    tiny_pickup_threshold_gal: float,
    days_since_service: float,
    services_in_horizon: float = 0.0,
) -> float:
    """
    Higher score = higher priority.
    """
    due_today = 1 if due_day == current_day else 0
    overdue = 1 if (due_day is not None and due_day < current_day) else 0
    overflow_today = 1 if overflow_day == current_day else 0
    overflow_tomorrow = 1 if overflow_day == current_day + 1 else 0

    if due_day is None:
        days_to_due = 999
    else:
        days_to_due = due_day - current_day

    tiny_pickup = 1 if projected_pickup_gal <= tiny_pickup_threshold_gal + 1e-9 else 0

    score = 0.0
    score += 2000.0 * overdue
    score += 1200.0 * due_today
    score += 1200.0 * overflow_today
    score += 700.0 * overflow_tomorrow

    if overflow_day is not None and overflow_day <= current_day + 1:
        score += 300.0

    score += 120.0 * max(0, 3 - max(days_to_due, 0))
    score += 0.12 * projected_pickup_gal
    score += 8.0 * min(days_since_service, 7.0)

    # discourage tiny pickups unless truly urgent
    score -= 35.0 * tiny_pickup

    # discourage repeated service within the same 7-day simulated run
    score -= 250.0 * services_in_horizon

    return score


def stream_caps(args: argparse.Namespace) -> dict[str, float]:
    return {
        "Waste": args.waste_volume_gal,
        "Compostables": 1.10 * args.compost_volume_gal,
        "Bottles/Cans": args.recycling_volume_gal,
    }


def build_bin_state_table(instance: pd.DataFrame) -> pd.DataFrame:
    out = instance.copy()

    out["Serial"] = out["Serial"].astype(str).str.strip()
    out["stream"] = out["stream"].apply(canonical_stream)

    out["threshold_pct"] = pd.to_numeric(out["threshold_pct"], errors="coerce")
    out["days_since_last_service"] = pd.to_numeric(out["days_since_last_service"], errors="coerce").fillna(0.0)
    out["current_fill_pct_est"] = pd.to_numeric(out["current_fill_pct_est"], errors="coerce").fillna(0.0)
    out["daily_fill_growth_pct"] = pd.to_numeric(out["daily_fill_growth_pct"], errors="coerce").fillna(0.0)
    out["bin_capacity_gal"] = pd.to_numeric(out["bin_capacity_gal"], errors="coerce").fillna(0.0)
    out["density_lb_per_gal"] = pd.to_numeric(out["density_lb_per_gal"], errors="coerce").fillna(1.0)
    out["avg_service_min"] = pd.to_numeric(out["avg_service_min"], errors="coerce").fillna(4.0)
    out["avg_travel_proxy_min"] = pd.to_numeric(out["avg_travel_proxy_min"], errors="coerce").fillna(3.0)

    out["fill_gal"] = out["bin_capacity_gal"] * out["current_fill_pct_est"] / 100.0
    out["growth_gal_per_day"] = out["bin_capacity_gal"] * out["daily_fill_growth_pct"] / 100.0
    out["threshold_gal"] = out["bin_capacity_gal"] * out["threshold_pct"] / 100.0
    out.loc[out["threshold_pct"].isna(), "threshold_gal"] = np.nan
    out["max_inventory_gal"] = out["bin_capacity_gal"] * MAX_FILL_FACTOR

    return out.reset_index(drop=True)


def plan_stream_for_day(
    day: int,
    stream_df: pd.DataFrame,
    num_available_trucks: int,
    args: argparse.Namespace,
    volume_cap_stream: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Greedy stream-level service selection for one day.
    """
    if stream_df.empty or num_available_trucks <= 0:
        return stream_df.iloc[0:0].copy(), {
            "trucks_used": 0,
            "total_gal": 0.0,
            "total_lb": 0.0,
            "total_min": 0.0,
            "extra_dumps_total": 0,
        }

    df = stream_df.copy()

    truck_mass_lb = args.truck_mass_lb
    truck_work_min = args.truck_work_min

    daily_volume_cap_total = num_available_trucks * volume_cap_stream
    daily_mass_cap_total = num_available_trucks * truck_mass_lb
    daily_time_cap_total = num_available_trucks * truck_work_min

    # Required first
    required_mask = (df["due_day"].notna()) & (df["due_day"] <= day)
    required = df[required_mask].copy().sort_values(
        ["priority", "projected_pickup_gal", "days_since_service_state"],
        ascending=[False, False, False],
    )
    optional = df[~required_mask].copy().sort_values(
        ["priority", "projected_pickup_gal", "days_since_service_state"],
        ascending=[False, False, False],
    )
    required_count = len(required)
    required["is_overdue"] = required["due_day"].notna() & (required["due_day"] < day)
    required["is_due_today"] = required["due_day"].notna() & (required["due_day"] == day)
    required["is_overflow_today"] = required["overflow_day"].notna() & (required["overflow_day"] == day)

    optional["is_overdue"] = optional["due_day"].notna() & (optional["due_day"] < day)
    optional["is_due_today"] = optional["due_day"].notna() & (optional["due_day"] == day)
    optional["is_overflow_today"] = optional["overflow_day"].notna() & (optional["overflow_day"] == day)
    selected_rows = []
    used_gal = 0.0
    used_lb = 0.0
    used_min = 0.0

    def can_add(row: pd.Series) -> bool:
        nonlocal used_gal, used_lb, used_min
        g = safe_float(row["projected_pickup_gal"])
        lb = g * safe_float(row["density_lb_per_gal"], 1.0)
        base_mins = safe_float(row["avg_service_min"], 4.0) + safe_float(row["avg_travel_proxy_min"], 3.0)

        if str(row["stream"]) == "Compostables":
            mins = 0.9 * base_mins
        else:
            mins = base_mins

        # Softly allow required bins even if it pushes demand beyond single-pass capacity;
        # later we estimate extra dumps.
        regular_limit_time = 0.9 * daily_time_cap_total
        if used_min + mins > regular_limit_time + 1e-9:
            return False

        # For gallons / pounds, allow some expansion due to dump cycles.
        hard_limit_gal = daily_volume_cap_total * (1 + args.max_extra_dumps)
        hard_limit_lb = daily_mass_cap_total * (1 + args.max_extra_dumps)

        if used_gal + g > hard_limit_gal + 1e-9:
            return False
        if used_lb + lb > hard_limit_lb + 1e-9:
            return False

        return True

    def add_row(row: pd.Series) -> None:
        nonlocal used_gal, used_lb, used_min, selected_rows
        g = safe_float(row["projected_pickup_gal"])
        lb = g * safe_float(row["density_lb_per_gal"], 1.0)
        base_mins = safe_float(row["avg_service_min"], 4.0) + safe_float(row["avg_travel_proxy_min"], 3.0)

        if str(row["stream"]) == "Compostables":
            mins = 0.9 * base_mins
        else:
            mins = base_mins
        used_gal += g
        used_lb += lb
        used_min += mins
        selected_rows.append(row)

    for _, row in required.iterrows():
        pickup_gal = safe_float(row["projected_pickup_gal"])
        tiny = pickup_gal <= args.tiny_pickup_threshold_gal + 1e-9
        forced = bool(row["is_overdue"] or row["is_overflow_today"])

        # skip tiny/zero pickups unless truly forced
        if tiny and not forced:
            continue

        if can_add(row):
            add_row(row)

    for _, row in optional.iterrows():
        pickup_gal = safe_float(row["projected_pickup_gal"])

        if required_count >= 0.75 * len(df):
            continue

        due_soon = pd.notna(row["due_day"]) and row["due_day"] <= day + 1
        overflow_soon = pd.notna(row["overflow_day"]) and row["overflow_day"] <= day + 1
        
        if day == 0:
            if str(row["stream"]) == "Compostables":
                meaningful_pickup = pickup_gal >= 70.0
            elif str(row["stream"]) == "Waste":
                meaningful_pickup = pickup_gal >= 50.0
            else:  # Bottles/Cans
                meaningful_pickup = pickup_gal >= 40.0
        else:
            if str(row["stream"]) == "Compostables":
                meaningful_pickup = pickup_gal >= 50.0
            elif str(row["stream"]) == "Waste":
                meaningful_pickup = pickup_gal >= 35.0
            else:  # Bottles/Cans
                meaningful_pickup = pickup_gal >= 30.0

        # much tighter optional screening
        if not (due_soon or overflow_soon or meaningful_pickup):
            continue

        tiny = pickup_gal <= args.tiny_pickup_threshold_gal + 1e-9
        forced = bool(overflow_soon)

        if tiny and not forced:
            continue

        if can_add(row):
            add_row(row)

    if selected_rows:
        selected = pd.DataFrame(selected_rows).copy()
    else:
        selected = df.iloc[0:0].copy()

    total_gal = selected["projected_pickup_gal"].sum() if not selected.empty else 0.0
    total_lb = (selected["projected_pickup_gal"] * selected["density_lb_per_gal"]).sum() if not selected.empty else 0.0
    total_min = (
        selected["avg_service_min"].sum() + selected["avg_travel_proxy_min"].sum()
        if not selected.empty else 0.0
    )

    trucks_by_vol = ceil_div_pos(total_gal, volume_cap_stream) if volume_cap_stream > 0 else 0
    trucks_by_mass = ceil_div_pos(total_lb, truck_mass_lb) if truck_mass_lb > 0 else 0
    trucks_by_time = ceil_div_pos(total_min, truck_work_min) if truck_work_min > 0 else 0

    trucks_needed = max(trucks_by_vol, trucks_by_mass, trucks_by_time)
    trucks_used = min(num_available_trucks, max(1, trucks_needed)) if not selected.empty else 0

    if trucks_used > 0:
        extra_dumps_by_vol = max(0, ceil_div_pos(total_gal, trucks_used * volume_cap_stream) - 1)
        extra_dumps_by_mass = max(0, ceil_div_pos(total_lb, trucks_used * truck_mass_lb) - 1)
        extra_dumps_total = max(extra_dumps_by_vol, extra_dumps_by_mass)
        extra_dumps_total = min(extra_dumps_total, args.max_extra_dumps)
    else:
        extra_dumps_total = 0

    return selected.reset_index(drop=True), {
        "trucks_used": trucks_used,
        "total_gal": float(total_gal),
        "total_lb": float(total_lb),
        "total_min": float(total_min),
        "extra_dumps_total": int(extra_dumps_total),
    }


def assign_trucks_within_stream(
    selected_df: pd.DataFrame,
    stream: str,
    day: int,
    trucks_for_stream: list[str],
    extra_dumps_total: int,
    args: argparse.Namespace,
    volume_cap_stream: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign selected bins to specific trucks greedily.
    """
    if selected_df.empty or not trucks_for_stream:
        empty_assign = selected_df.copy()
        return empty_assign.iloc[0:0].copy(), pd.DataFrame(
            [{"day": day, "truck": t, "assigned_stream": stream, "extra_dumps": 0} for t in trucks_for_stream]
        )

    df = selected_df.copy().sort_values(
        ["projected_pickup_gal", "priority", "days_since_service_state"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    truck_rows = []
    truck_state = {}
    n_trucks = len(trucks_for_stream)

    base_dump_allocation = [0] * n_trucks
    for j in range(extra_dumps_total):
        base_dump_allocation[j % n_trucks] += 1

    for idx, t in enumerate(trucks_for_stream):
        ed = base_dump_allocation[idx]
        truck_state[t] = {
            "gal_used": 0.0,
            "lb_used": 0.0,
            "min_used": 0.0,
            "vol_cap_eff": volume_cap_stream * (1 + ed),
            "mass_cap_eff": args.truck_mass_lb * (1 + ed),
            "regular_min_cap": args.truck_work_min,
            "max_min_cap": args.truck_work_min + args.max_overtime_min,
            "extra_dumps": ed,
        }
        truck_rows.append({"day": day, "truck": t, "assigned_stream": stream, "extra_dumps": ed})

    assignments = []
    for _, row in df.iterrows():
        g = safe_float(row["projected_pickup_gal"])
        lb = g * safe_float(row["density_lb_per_gal"], 1.0)
        mins = safe_float(row["avg_service_min"], 4.0) + safe_float(row["avg_travel_proxy_min"], 3.0)

        feasible_trucks = []
        for t in trucks_for_stream:
            st = truck_state[t]
            if st["gal_used"] + g <= st["vol_cap_eff"] + 1e-9 \
               and st["lb_used"] + lb <= st["mass_cap_eff"] + 1e-9 \
               and st["min_used"] + mins <= st["max_min_cap"] + 1e-9:
                slack = (
                    (st["vol_cap_eff"] - (st["gal_used"] + g))
                    + 0.15 * (st["mass_cap_eff"] - (st["lb_used"] + lb))
                    + 0.05 * (st["max_min_cap"] - (st["min_used"] + mins))
                )
                feasible_trucks.append((t, slack))

        if not feasible_trucks:
            # If no truck can take it, skip it here.
            # This should be rare because stream-level planning already screened capacity.
            continue

        feasible_trucks.sort(key=lambda x: x[1])
        chosen_truck = feasible_trucks[0][0]

        truck_state[chosen_truck]["gal_used"] += g
        truck_state[chosen_truck]["lb_used"] += lb
        truck_state[chosen_truck]["min_used"] += mins

        out = row.to_dict()
        out["service_day"] = day
        out["truck"] = chosen_truck
        out["pickup_gal"] = round(g, 2)
        out["pickup_lb"] = round(lb, 2)
        out["inventory_gal_before_service"] = round(g, 2)
        cap = safe_float(row["bin_capacity_gal"])
        out["inventory_pct_before_service"] = round(100.0 * g / cap, 2) if cap > 0 else np.nan
        assignments.append(out)

    assigned_df = pd.DataFrame(assignments)
    truck_stream_df = pd.DataFrame(truck_rows)

    return assigned_df, truck_stream_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate weekly Bigbelly planning heuristic.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--max-bins", type=int, default=None)
    parser.add_argument("--num-trucks", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=7)

    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)

    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--dump-turnaround-min", type=float, default=17.0)
    parser.add_argument("--max-extra-dumps", type=int, default=8)
    parser.add_argument("--max-overtime-min", type=float, default=180.0)

    parser.add_argument("--tiny-pickup-threshold-gal", type=float, default=5.0)
    parser.add_argument("--required-only", action="store_true")
    parser.add_argument("--require-routable", action="store_true")

    args = parser.parse_args()

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

    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)

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

    if "service_deadline" not in df.columns:
        df["service_deadline"] = np.nan
    if "deadline_interval" not in df.columns:
        df["deadline_interval"] = df["days_since_last_service"].apply(
            lambda x: interval_deadline(x, args.horizon_days)
        )

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["stream"] = df["stream"].apply(canonical_stream)
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)

    if args.require_routable:
        routable_mask = is_routable_for_routing(df)
        stop_lookup_serials = load_stop_lookup_serials(root)
        stop_lookup_mask = df["Serial"].astype(str).str.strip().isin(stop_lookup_serials)

        df = df[routable_mask & stop_lookup_mask].copy()

        if df.empty:
            raise ValueError(
                "No routable bins remain after applying --require-routable "
                "(coordinates + bin_stop_lookup membership)."
            )

    instance = choose_instance(
        df=df,
        max_bins=args.max_bins,
        required_only=args.required_only,
        horizon_days=args.horizon_days,
    )
    if instance.empty:
        raise ValueError("No bins available for planning after filtering.")

    state_df = build_bin_state_table(instance)
    state_df["services_in_horizon"] = 0
    state_df["served_in_horizon"] = 0

    caps = stream_caps(args)
    trucks = [f"T{k + 1}" for k in range(args.num_trucks)]
    streams = [s for s in sorted(state_df["stream"].dropna().unique().tolist()) if s != "Unknown"]

    all_schedule_rows = []
    all_truck_stream_rows = []
    all_load_rows = []
    all_inventory_rows = []

    total_pickups = 0
    total_extra_dumps = 0
    total_overtime_min = 0.0
    overflow_bin_days = 0

    # Day-by-day constructive plan
    for day in range(args.horizon_days):
        day_work = state_df.copy()
        day_work = day_work[day_work["served_in_horizon"] == 0].copy()

        due_records = []
        for idx, row in day_work.iterrows():
            due_info = earliest_due_day(
                fill_gal=safe_float(row["fill_gal"]),
                days_since_service=safe_float(row["days_since_last_service"]),
                growth_gal_per_day=safe_float(row["growth_gal_per_day"]),
                threshold_gal=None if pd.isna(row["threshold_gal"]) else safe_float(row["threshold_gal"]),
                bin_capacity_gal=safe_float(row["bin_capacity_gal"]),
                horizon_days=args.horizon_days - day,
            )

            due_day_local = due_info["due_day"]
            overflow_day_local = due_info["overflow_day"]

            due_day_abs = None if due_day_local is None else day + int(due_day_local)
            overflow_day_abs = None if overflow_day_local is None else day + int(overflow_day_local)

            projected_pickup_gal = safe_float(row["fill_gal"])

            priority = compute_priority(
                current_day=day,
                due_day=due_day_abs,
                overflow_day=overflow_day_abs,
                projected_pickup_gal=projected_pickup_gal,
                tiny_pickup_threshold_gal=args.tiny_pickup_threshold_gal,
                days_since_service=safe_float(row["days_since_last_service"]),
                services_in_horizon=safe_float(row.get("services_in_horizon", 0.0)),
            )

            due_records.append(
                {
                    "row_idx": idx,
                    "due_day": due_day_abs,
                    "overflow_day": overflow_day_abs,
                    "projected_pickup_gal": projected_pickup_gal,
                    "priority": priority,
                    "days_since_service_state": safe_float(row["days_since_last_service"]),
                }
            )

        due_df = pd.DataFrame(due_records)
        day_work = day_work.merge(due_df, left_index=True, right_on="row_idx", how="left").drop(columns=["row_idx"])

        # Inventory log before service on this day
        for _, row in day_work.iterrows():
            fill_gal = safe_float(row["fill_gal"])
            cap = safe_float(row["bin_capacity_gal"])
            if fill_gal > cap + 1e-9:
                overflow_bin_days += 1

            all_inventory_rows.append(
                {
                    "Serial": row["Serial"],
                    "stream": row["stream"],
                    "day": day,
                    "inventory_gal_start": round(fill_gal, 2),
                    "inventory_pct_start": round(100.0 * fill_gal / cap, 2) if cap > 0 else np.nan,
                    "pickup_flag": 0,
                    "pickup_gal": 0.0,
                    "inventory_gal_after_service": round(fill_gal, 2),
                }
            )

        # Stream ordering: first by count of due/overdue bins, then by priority mass
        stream_day_summary = []
        for s in streams:
            sdf = day_work[day_work["stream"] == s].copy()
            due_cnt = int(((sdf["due_day"].notna()) & (sdf["due_day"] <= day)).sum())
            pr_sum = float(sdf["priority"].sum()) if not sdf.empty else 0.0
            stream_day_summary.append((s, due_cnt, pr_sum))
        stream_day_summary.sort(key=lambda x: (x[1], x[2]), reverse=True)

        remaining_trucks = trucks.copy()
        assigned_today = []

        # First pass: allocate at least one truck to streams with due bins if possible
        prelim_alloc = {}
        streams_with_due = [s for s, due_cnt, _ in stream_day_summary if due_cnt > 0]
        for s in streams_with_due:
            if remaining_trucks:
                prelim_alloc[s] = 1
                remaining_trucks.pop(0)

        # Second pass: allocate leftover trucks to streams with strongest need
        extra_stream_order = []
        for s, _, _ in stream_day_summary:
            sdf = day_work[day_work["stream"] == s].copy()
            total_gal = float(sdf["projected_pickup_gal"].sum()) if not sdf.empty else 0.0
            extra_stream_order.append((s, total_gal))
        extra_stream_order.sort(key=lambda x: x[1], reverse=True)

        while remaining_trucks:
            allocated = False
            for s, _ in extra_stream_order:
                prelim_alloc[s] = prelim_alloc.get(s, 0) + 1
                remaining_trucks.pop(0)
                allocated = True
                break
            if not allocated:
                break

        truck_cursor = 0
        for s, _, _ in stream_day_summary:
            trucks_for_stream_n = prelim_alloc.get(s, 0)
            if trucks_for_stream_n <= 0:
                continue

            stream_trucks = trucks[truck_cursor: truck_cursor + trucks_for_stream_n]
            truck_cursor += trucks_for_stream_n

            sdf = day_work[day_work["stream"] == s].copy()
            selected_s, metrics_s = plan_stream_for_day(
                day=day,
                stream_df=sdf,
                num_available_trucks=len(stream_trucks),
                args=args,
                volume_cap_stream=caps[s],
            )

            assigned_s, truck_stream_s = assign_trucks_within_stream(
                selected_df=selected_s,
                stream=s,
                day=day,
                trucks_for_stream=stream_trucks,
                extra_dumps_total=metrics_s["extra_dumps_total"],
                args=args,
                volume_cap_stream=caps[s],
            )

            if not assigned_s.empty:
                assigned_today.append(assigned_s)

            if not truck_stream_s.empty:
                all_truck_stream_rows.extend(truck_stream_s.to_dict(orient="records"))

            # Build load check rows
            if not assigned_s.empty:
                for t in stream_trucks:
                    td = assigned_s[assigned_s["truck"] == t].copy()
                    extra_dumps_t = int(
                        truck_stream_s.loc[truck_stream_s["truck"] == t, "extra_dumps"].iloc[0]
                    ) if not truck_stream_s.empty and (truck_stream_s["truck"] == t).any() else 0

                    gal_used = float(td["pickup_gal"].sum()) if not td.empty else 0.0
                    lb_used = float(td["pickup_lb"].sum()) if not td.empty else 0.0
                    min_wo_dumps = float((td["avg_service_min"] + td["avg_travel_proxy_min"]).sum()) if not td.empty else 0.0
                    minutes_total = min_wo_dumps + args.dump_turnaround_min * extra_dumps_t
                    overtime = max(0.0, minutes_total - args.truck_work_min)

                    total_overtime_min += overtime
                    total_extra_dumps += extra_dumps_t

                    all_load_rows.append(
                        {
                            "day": day,
                            "truck": t,
                            "assigned_stream": s,
                            "pickup_gal_used": round(gal_used, 2),
                            "pickup_gal_capacity_effective": round(caps[s] * (1 + extra_dumps_t), 2),
                            "pickup_lb_used": round(lb_used, 2),
                            "pickup_lb_capacity_effective": round(args.truck_mass_lb * (1 + extra_dumps_t), 2),
                            "extra_dumps": extra_dumps_t,
                            "minutes_used_without_dumps": round(min_wo_dumps, 2),
                            "minutes_used_total": round(minutes_total, 2),
                            "overtime_min": round(overtime, 2),
                            "minutes_capacity_regular": round(args.truck_work_min, 2),
                            "minutes_capacity_with_overtime": round(args.truck_work_min + args.max_overtime_min, 2),
                        }
                    )

        if assigned_today:
            assigned_today_df = pd.concat(assigned_today, ignore_index=True)
        else:
            assigned_today_df = pd.DataFrame()
        
        if not assigned_today_df.empty:
            assigned_today_df = assigned_today_df[
                pd.to_numeric(assigned_today_df["pickup_gal"], errors="coerce").fillna(0.0) > 1e-9
            ].copy()

        # Update state
        served_serials = set()
        if not assigned_today_df.empty:
            for _, row in assigned_today_df.iterrows():
                serial = str(row["Serial"])
                served_serials.add(serial)
                total_pickups += 1

                all_schedule_rows.append(
                    {
                        "Serial": serial,
                        "stream": row["stream"],
                        "service_day": int(row["service_day"]),
                        "truck": row["truck"],
                        "inventory_gal_before_service": round(safe_float(row["inventory_gal_before_service"]), 2),
                        "inventory_pct_before_service": round(safe_float(row["inventory_pct_before_service"]), 2),
                        "pickup_gal": round(safe_float(row["pickup_gal"]), 2),
                        "pickup_lb": round(safe_float(row["pickup_lb"]), 2),
                        "interval_deadline": (
                            None
                            if pd.isna(row.get("deadline_interval", np.nan))
                            else int(safe_float(row["deadline_interval"]))
                        ),
                    }
                )

            # mark inventory rows
            inv_index = {(r["Serial"], r["day"]): idx for idx, r in enumerate(all_inventory_rows)}
            for _, row in assigned_today_df.iterrows():
                key = (str(row["Serial"]), int(row["service_day"]))
                idx = inv_index.get(key)
                if idx is not None:
                    all_inventory_rows[idx]["pickup_flag"] = 1
                    all_inventory_rows[idx]["pickup_gal"] = round(safe_float(row["pickup_gal"]), 2)
                    all_inventory_rows[idx]["inventory_gal_after_service"] = 0.0
        if served_serials:
            state_df.loc[
                state_df["Serial"].astype(str).isin(served_serials),
                "served_in_horizon"
            ] = 1
        # end-of-day state transition
        new_fill = []
        new_days_since = []
        for _, row in state_df.iterrows():
            serial = str(row["Serial"])
            fill = safe_float(row["fill_gal"])
            growth = safe_float(row["growth_gal_per_day"])
            max_inv = safe_float(row["max_inventory_gal"])
            dsl = safe_float(row["days_since_last_service"])

            if serial in served_serials:
                next_fill = 0.0
                next_dsl = 0.0
            else:
                next_fill = min(max_inv, fill + growth)
                next_dsl = dsl + 1.0

            new_fill.append(next_fill)
            new_days_since.append(next_dsl)

        if "services_in_horizon" not in state_df.columns:
            state_df["services_in_horizon"] = 0
            state_df["served_in_horizon"] = 0

        state_df["services_in_horizon"] = state_df.apply(
            lambda r: safe_float(r["services_in_horizon"]) + 1
            if str(r["Serial"]) in served_serials
            else safe_float(r["services_in_horizon"]),
            axis=1,
        )
        state_df["fill_gal"] = new_fill
        state_df["days_since_last_service"] = new_days_since

    # Build outputs
    if all_schedule_rows:
        schedule_df = pd.DataFrame(all_schedule_rows).sort_values(["service_day", "Serial"])
    else:
        schedule_df = pd.DataFrame(columns=[
            "Serial", "stream", "service_day", "truck",
            "inventory_gal_before_service", "inventory_pct_before_service",
            "pickup_gal", "pickup_lb", "interval_deadline"
        ])

    if all_truck_stream_rows:
        truck_stream_df = pd.DataFrame(all_truck_stream_rows).sort_values(["day", "truck"])
    else:
        truck_stream_df = pd.DataFrame(columns=["day", "truck", "assigned_stream", "extra_dumps"])

    if all_load_rows:
        load_df = pd.DataFrame(all_load_rows).sort_values(["day", "truck"])
    else:
        load_df = pd.DataFrame(columns=[
            "day", "truck", "assigned_stream",
            "pickup_gal_used", "pickup_gal_capacity_effective",
            "pickup_lb_used", "pickup_lb_capacity_effective",
            "extra_dumps", "minutes_used_without_dumps",
            "minutes_used_total", "overtime_min",
            "minutes_capacity_regular", "minutes_capacity_with_overtime"
        ])

    if all_inventory_rows:
        inventory_df = pd.DataFrame(all_inventory_rows).sort_values(["day", "Serial"])
    else:
        inventory_df = pd.DataFrame(columns=[
            "Serial", "stream", "day", "inventory_gal_start", "inventory_pct_start",
            "pickup_flag", "pickup_gal", "inventory_gal_after_service"
        ])

    status = "Heuristic Feasible"

    planning_summary_df = pd.DataFrame([
        {
            "status": status,
            "objective_value": np.nan,
            "horizon_days": args.horizon_days,
            "num_bins_in_instance": int(len(state_df)),
            "num_trucks": int(args.num_trucks),
            "truck_work_min": float(args.truck_work_min),
            "total_pickups": round(float(total_pickups), 2),
            "total_extra_dumps": round(float(total_extra_dumps), 2),
            "total_overtime_min": round(float(total_overtime_min), 2),
            "overflow_bin_days": int(overflow_bin_days),
            "cbc_time_limit_sec": np.nan,
            "cbc_gap_rel": np.nan,
            "require_routable": bool(args.require_routable),
            "planner_type": "rolling_horizon_greedy_heuristic",
        }
    ])

    schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
    truck_stream_fp = paths["processed"] / "small_instance_truck_streams.csv"
    load_fp = paths["processed"] / "small_instance_truck_load_check.csv"
    inventory_fp = paths["processed"] / "small_instance_inventory_trajectory.csv"
    planning_summary_fp = paths["processed"] / "small_instance_planning_summary.csv"

    schedule_df.to_csv(schedule_fp, index=False)
    truck_stream_df.to_csv(truck_stream_fp, index=False)
    load_df.to_csv(load_fp, index=False)
    inventory_df.to_csv(inventory_fp, index=False)
    planning_summary_df.to_csv(planning_summary_fp, index=False)

    print(f"[STATUS] {status}")
    print(f"[OK] Wrote: {schedule_fp}")
    print(f"[OK] Wrote: {truck_stream_fp}")
    print(f"[OK] Wrote: {load_fp}")
    print(f"[OK] Wrote: {inventory_fp}")
    print(f"[OK] Wrote: {planning_summary_fp}")

    if not planning_summary_df.empty:
        print("\nPlanning summary:")
        print(planning_summary_df.to_string(index=False))

    if not schedule_df.empty:
        print("\nSchedule preview:")
        print(schedule_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()