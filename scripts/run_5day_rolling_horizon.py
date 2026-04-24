from __future__ import annotations

"""
run_5day_rolling_horizon.py
-------------------------------------------------------------
Run a 5-day rolling-horizon experiment for the Bigbelly project.

For each weekday:
1. Start from the current bin state table.
2. Write a temporary projection input for that day.
3. Run the 7-day heuristic planner.
4. Keep only service_day == 0 assignments.
5. Run daily routing on those assignments.
6. Update fill state and days-since-last-service.
7. Repeat for the next day.

Outputs
-------
Writes combined outputs under data/processed/rolling_horizon_5day/:
- rolling_day_state_history.csv
- rolling_day_schedule.csv
- rolling_day_route_plan.csv
- rolling_day_route_stops.csv
- rolling_day_route_summary.csv
- rolling_day_metrics.csv
- rolling_5day_summary.csv
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def ensure_dirs(root: Path) -> dict[str, Path]:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    outdir = processed / "rolling_horizon_5day"
    outdir.mkdir(parents=True, exist_ok=True)

    return {
        "processed": processed,
        "outdir": outdir,
    }


def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    remaining = 7 - float(days_since_last_service)
    if remaining <= 0:
        return 0.0
    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))
    return np.nan


def load_base_projection(processed: Path, input_path: str | None) -> pd.DataFrame:
    if input_path:
        fp = Path(input_path)
    else:
        parquet_fp = processed / "bin_7day_projection_inputs.parquet"
        csv_fp = processed / "bin_7day_projection_inputs.csv"
        if parquet_fp.exists():
            fp = parquet_fp
        elif csv_fp.exists():
            fp = csv_fp
        else:
            raise FileNotFoundError(
                "Could not find bin_7day_projection_inputs.parquet or .csv"
            )

    if fp.suffix.lower() == ".csv":
        df = pd.read_csv(fp)
    else:
        df = pd.read_parquet(fp)

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

    df = df.copy()
    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["stream"] = df["stream"].apply(canonical_stream)
    df["days_since_last_service"] = pd.to_numeric(df["days_since_last_service"], errors="coerce").fillna(0.0)
    df["current_fill_pct_est"] = pd.to_numeric(df["current_fill_pct_est"], errors="coerce").fillna(0.0)
    df["daily_fill_growth_pct"] = pd.to_numeric(df["daily_fill_growth_pct"], errors="coerce").fillna(0.0)
    df["bin_capacity_gal"] = pd.to_numeric(df["bin_capacity_gal"], errors="coerce").fillna(0.0)
    df["threshold_pct"] = pd.to_numeric(df["threshold_pct"], errors="coerce")
    df["density_lb_per_gal"] = pd.to_numeric(df["density_lb_per_gal"], errors="coerce").fillna(1.0)
    df["avg_service_min"] = pd.to_numeric(df["avg_service_min"], errors="coerce").fillna(4.0)
    df["avg_travel_proxy_min"] = pd.to_numeric(df["avg_travel_proxy_min"], errors="coerce").fillna(3.0)
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)

    return df


def filter_to_routable_bins(base_df: pd.DataFrame, processed: Path) -> pd.DataFrame:
    lookup_fp = processed / "bin_stop_lookup.csv"

    if not lookup_fp.exists():
        raise FileNotFoundError(
            f"Missing {lookup_fp}. Run scripts/build_travel_matrix.py first."
        )

    lookup_df = pd.read_csv(lookup_fp)

    if "Serial" not in lookup_df.columns:
        raise KeyError("bin_stop_lookup.csv must contain a 'Serial' column.")

    lookup_df["Serial"] = lookup_df["Serial"].astype(str).str.strip()
    valid_serials = set(lookup_df["Serial"])

    before = len(base_df)

    base_df = base_df.copy()
    base_df["Serial"] = base_df["Serial"].astype(str).str.strip()
    base_df = base_df[base_df["Serial"].isin(valid_serials)].copy()

    after = len(base_df)

    print(f"[INFO] Routable-bin filter applied: kept {after} of {before} bins.")
    print(f"[INFO] Removed {before - after} bins missing from bin_stop_lookup.csv.")

    if base_df.empty:
        raise ValueError("After routable-bin filtering, no bins remain.")

    return base_df

def initialize_state(base_df: pd.DataFrame) -> pd.DataFrame:
    state = base_df.copy()
    state["fill_gal"] = state["bin_capacity_gal"] * state["current_fill_pct_est"] / 100.0
    state["growth_gal_per_day"] = state["bin_capacity_gal"] * state["daily_fill_growth_pct"] / 100.0
    return state


def rebuild_projection_from_state(state_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    out = state_df.copy()

    out["current_fill_pct_est"] = np.where(
        out["bin_capacity_gal"] > 0,
        100.0 * out["fill_gal"] / out["bin_capacity_gal"],
        0.0,
    )

    out["must_service_within_horizon"] = out["days_since_last_service"].apply(
        lambda x: (7 - safe_float(x)) <= horizon_days
    )

    out["service_deadline"] = out["days_since_last_service"].apply(
        lambda x: interval_deadline(safe_float(x), horizon_days)
    )
    out["deadline_interval"] = out["service_deadline"]

    keep_cols = list(dict.fromkeys(list(state_df.columns) + ["service_deadline", "deadline_interval"]))
    return out[keep_cols].copy()


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def overwrite_day0_schedule_and_trucks(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_fp = processed / "small_instance_service_schedule.csv"
    truck_stream_fp = processed / "small_instance_truck_streams.csv"

    if not schedule_fp.exists():
        raise FileNotFoundError(f"Missing {schedule_fp}")
    if not truck_stream_fp.exists():
        raise FileNotFoundError(f"Missing {truck_stream_fp}")

    schedule = pd.read_csv(schedule_fp)
    truck_streams = pd.read_csv(truck_stream_fp)

    if not schedule.empty:
        schedule = schedule.copy()
        schedule["service_day"] = pd.to_numeric(schedule["service_day"], errors="coerce")
        schedule = schedule[schedule["service_day"] == 0].copy()
        schedule["service_day"] = 0

    if not truck_streams.empty:
        truck_streams = truck_streams.copy()
        truck_streams["day"] = pd.to_numeric(truck_streams["day"], errors="coerce")
        truck_streams = truck_streams[truck_streams["day"] == 0].copy()
        truck_streams["day"] = 0

    schedule.to_csv(schedule_fp, index=False)
    truck_streams.to_csv(truck_stream_fp, index=False)

    return schedule, truck_streams


def read_optional_csv(fp: Path) -> pd.DataFrame:
    if fp.exists():
        return pd.read_csv(fp)
    return pd.DataFrame()


def update_state_after_day(
    state_df: pd.DataFrame,
    day0_schedule: pd.DataFrame,
) -> pd.DataFrame:
    served_serials = set()
    if not day0_schedule.empty:
        served_serials = set(day0_schedule["Serial"].astype(str).str.strip().tolist())

    new_state = state_df.copy()
    new_fill = []
    new_dsl = []

    for _, row in new_state.iterrows():
        serial = str(row["Serial"]).strip()
        fill = safe_float(row["fill_gal"])
        growth = safe_float(row["growth_gal_per_day"])
        cap = safe_float(row["bin_capacity_gal"])
        dsl = safe_float(row["days_since_last_service"])

        if serial in served_serials:
            next_fill = 0.0
            next_dsl = 0.0
        else:
            next_fill = min(cap * 1.50, fill + growth)
            next_dsl = dsl + 1.0

        new_fill.append(next_fill)
        new_dsl.append(next_dsl)

    new_state["fill_gal"] = new_fill
    new_state["days_since_last_service"] = new_dsl

    return new_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-day rolling-horizon experiment.")
    parser.add_argument("--projection-input", type=str, default=None)
    parser.add_argument("--num-days", type=int, default=5)
    parser.add_argument("--num-trucks", type=int, default=3)
    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--max-overtime-min", type=float, default=180.0)
    parser.add_argument("--tiny-pickup-threshold-gal", type=float, default=5.0)
    parser.add_argument("--require-routable", action="store_true")
    parser.add_argument("--max-bins", type=int, default=None)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
    "--use-observed-shift-span",
    action="store_true",
    help="Use expanded effective truck-day span based on observed staggered driver shifts",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)
    processed = paths["processed"]
    outdir = paths["outdir"]

    base_df = load_base_projection(processed, args.projection_input)

    # Critical fix: only plan over routable bins
    if args.require_routable:
        base_df = filter_to_routable_bins(base_df, processed)

    state_df = initialize_state(base_df)

    all_state_rows = []
    all_schedule_rows = []
    all_route_plan_rows = []
    all_route_stop_rows = []
    all_route_summary_rows = []
    day_metrics_rows = []

    py_exec = "python"

    for rolling_day in range(args.num_days):
        print(f"\n========== ROLLING DAY {rolling_day + 1} / {args.num_days} ==========")

        # Snapshot state before planning
        snap = state_df.copy()
        snap["rolling_day"] = rolling_day + 1
        snap["fill_pct_start"] = np.where(
            snap["bin_capacity_gal"] > 0,
            100.0 * snap["fill_gal"] / snap["bin_capacity_gal"],
            0.0,
        )
        all_state_rows.extend(snap.to_dict(orient="records"))

        # Build fresh projection input for this day
        projection_df = rebuild_projection_from_state(state_df, args.horizon_days)
        temp_projection_fp = processed / "rolling_temp_projection_input.csv"
        projection_df.to_csv(temp_projection_fp, index=False)

        # Run weekly heuristic
        cmd_plan = [
            py_exec,
            "scripts/solve_7day_schedule.py",
            "--input", str(temp_projection_fp),
            "--num-trucks", str(args.num_trucks),
            "--truck-work-min", str(args.truck_work_min),
            "--max-overtime-min", str(args.max_overtime_min),
            "--tiny-pickup-threshold-gal", str(args.tiny_pickup_threshold_gal),
            "--horizon-days", str(args.horizon_days),
             "--cbc-time-limit-sec", "600",
             "--cbc-gap-rel", "0.10",
        ]
        if args.use_observed_shift_span:
            cmd_plan.append("--use-observed-shift-span")
        if args.max_bins is not None:
            cmd_plan.extend(["--max-bins", str(args.max_bins)])
        if args.require_routable:
            cmd_plan.append("--require-routable")

        run_subprocess(cmd_plan, cwd=root)

        # Keep only day 0 decisions
        day0_schedule, day0_trucks = overwrite_day0_schedule_and_trucks(processed)

        # Run routing on day 0 schedule only
        route_work_min = 750.0 if args.use_observed_shift_span else args.truck_work_min
        cmd_route = [
            py_exec,
            "scripts/solve_daily_routing.py",
            "--projection-input", str(temp_projection_fp),
            "--truck-work-min", str(route_work_min),
        ]
        run_subprocess(cmd_route, cwd=root)

        # Read outputs
        route_plan = read_optional_csv(processed / "daily_route_plan.csv")
        route_stops = read_optional_csv(processed / "daily_route_stops.csv")
        route_summary = read_optional_csv(processed / "daily_route_summary.csv")
        planning_summary = read_optional_csv(processed / "small_instance_planning_summary.csv")
        load_check = read_optional_csv(processed / "small_instance_truck_load_check.csv")

        if not day0_schedule.empty:
            day0_schedule = day0_schedule.copy()
            day0_schedule["rolling_day"] = rolling_day + 1
            all_schedule_rows.extend(day0_schedule.to_dict(orient="records"))

        if not route_plan.empty:
            route_plan = route_plan.copy()
            route_plan["rolling_day"] = rolling_day + 1
            all_route_plan_rows.extend(route_plan.to_dict(orient="records"))

        if not route_stops.empty:
            route_stops = route_stops.copy()
            route_stops["rolling_day"] = rolling_day + 1
            all_route_stop_rows.extend(route_stops.to_dict(orient="records"))

        if not route_summary.empty:
            route_summary = route_summary.copy()
            route_summary["rolling_day"] = rolling_day + 1
            all_route_summary_rows.extend(route_summary.to_dict(orient="records"))

        pickups_today = len(day0_schedule) if not day0_schedule.empty else 0
        pickup_gal_today = float(day0_schedule["pickup_gal"].sum()) if not day0_schedule.empty else 0.0
        pickup_lb_today = float(day0_schedule["pickup_lb"].sum()) if not day0_schedule.empty else 0.0
        routes_today = int(len(route_plan)) if not route_plan.empty else 0
        route_minutes_today = float(route_plan["route_minutes"].sum()) if not route_plan.empty else 0.0
        overtime_today = float(load_check["overtime_min"].sum()) if not load_check.empty else 0.0

        overflow_today = int((state_df["fill_gal"] > state_df["bin_capacity_gal"]).sum())

        day_metrics_rows.append(
            {
                "rolling_day": rolling_day + 1,
                "pickups_today": pickups_today,
                "pickup_gal_today": round(pickup_gal_today, 2),
                "pickup_lb_today": round(pickup_lb_today, 2),
                "routes_today": routes_today,
                "route_minutes_today": round(route_minutes_today, 2),
                "overtime_today": round(overtime_today, 2),
                "overflow_bins_start_of_day": overflow_today,
                "planner_status": planning_summary["status"].iloc[0] if not planning_summary.empty and "status" in planning_summary.columns else None,
            }
        )

        # Update state for next rolling day
        state_df = update_state_after_day(state_df, day0_schedule)

    # Write combined outputs
    state_hist_df = pd.DataFrame(all_state_rows)
    rolling_sched_df = pd.DataFrame(all_schedule_rows)
    rolling_route_plan_df = pd.DataFrame(all_route_plan_rows)
    rolling_route_stops_df = pd.DataFrame(all_route_stop_rows)
    rolling_route_summary_df = pd.DataFrame(all_route_summary_rows)
    day_metrics_df = pd.DataFrame(day_metrics_rows)

    overall_summary = pd.DataFrame([{
        "num_days": args.num_days,
        "total_pickups": int(day_metrics_df["pickups_today"].sum()) if not day_metrics_df.empty else 0,
        "total_pickup_gal": round(float(day_metrics_df["pickup_gal_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
        "total_pickup_lb": round(float(day_metrics_df["pickup_lb_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
        "total_routes": int(day_metrics_df["routes_today"].sum()) if not day_metrics_df.empty else 0,
        "total_route_minutes": round(float(day_metrics_df["route_minutes_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
        "total_overtime": round(float(day_metrics_df["overtime_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
        "avg_overflow_bins_start_of_day": round(float(day_metrics_df["overflow_bins_start_of_day"].mean()), 2) if not day_metrics_df.empty else 0.0,
        "planner_type": "rolling_horizon_greedy_heuristic_outer_loop",
    }])

    state_hist_df.to_csv(outdir / "rolling_day_state_history.csv", index=False)
    rolling_sched_df.to_csv(outdir / "rolling_day_schedule.csv", index=False)
    rolling_route_plan_df.to_csv(outdir / "rolling_day_route_plan.csv", index=False)
    rolling_route_stops_df.to_csv(outdir / "rolling_day_route_stops.csv", index=False)
    rolling_route_summary_df.to_csv(outdir / "rolling_day_route_summary.csv", index=False)
    day_metrics_df.to_csv(outdir / "rolling_day_metrics.csv", index=False)
    overall_summary.to_csv(outdir / "rolling_5day_summary.csv", index=False)

    print("\n========== 5-DAY SUMMARY ==========")
    if not overall_summary.empty:
        print(overall_summary.to_string(index=False))
    print(f"\n[OK] Wrote combined outputs to: {outdir}")


if __name__ == "__main__":
    main()