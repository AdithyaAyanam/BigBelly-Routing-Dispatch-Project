from __future__ import annotations

"""
run_5day_rolling_horizon.py
-------------------------------------------------------------
Run a 5-day rolling-horizon experiment for the Bigbelly project.

This version is aligned with the final corrected seven-calendar-day / five-service-day planning model:
- 7-calendar-day lookahead with 5 weekday service days
- 4 trucks
- 1 driver per truck
- 480 minutes per truck-day
- 0 baseline overtime
- $40/hour driver wage
- 4 minutes service time per bin
- 8 minutes average planning travel time
- 17 minutes per dump cycle
- $20 overflow penalty per bin-day
- overflow penalty applied across all 7 calendar days, including non-service days
- compost weekday hygiene enforced
- no Phase-1 routable-bin filter
- Day 0 routing is treated as a downstream feasibility check

For each operating day:
1. Start from the current bin state table.
2. Write a temporary projection input for that day.
3. Run the 7-day planning model.
4. Keep only service_day == 0 assignments.
5. Rebuild the Day 0 travel matrix for the current Day 0 schedule.
6. Run daily routing on those Day 0 assignments.
7. Update fill state and days-since-last-service.
8. Repeat for the next operating day.

Outputs are written under:
data/processed/rolling_horizon_5day/
"""

import argparse
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
        "bottles and cans": "Bottles/Cans",
        "recycling": "Bottles/Cans",
        "single stream": "Bottles/Cans",
    }
    return lookup.get(s, str(x).strip() if str(x).strip() else "Unknown")


def clean_serial(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


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
    return {"processed": processed, "outdir": outdir}


def interval_deadline(days_since_last_service: float, horizon_days: int, stream: str | None = None) -> float:
    stream_name = canonical_stream(stream) if stream is not None else ""
    hygiene_window = 5 if stream_name == "Compostables" else 7
    remaining = hygiene_window - float(days_since_last_service)

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
            raise FileNotFoundError("Could not find bin_7day_projection_inputs.parquet or .csv. Run scripts/build_projected_fill.py first.")

    df = pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_parquet(fp)

    required_cols = [
        "Serial", "stream", "threshold_pct", "days_since_last_service",
        "current_fill_pct_est", "daily_fill_growth_pct", "bin_capacity_gal",
        "density_lb_per_gal", "avg_service_min", "avg_travel_proxy_min",
        "must_service_within_horizon",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input file missing columns: {missing}")

    df = df.copy()
    df["Serial"] = df["Serial"].apply(clean_serial)
    df["stream"] = df["stream"].apply(canonical_stream)

    numeric_defaults = {
        "days_since_last_service": 0.0,
        "current_fill_pct_est": 0.0,
        "daily_fill_growth_pct": 0.0,
        "bin_capacity_gal": 0.0,
        "density_lb_per_gal": 1.0,
        "avg_service_min": 4.0,
        "avg_travel_proxy_min": 8.0,
    }
    for col, default in numeric_defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["threshold_pct"] = pd.to_numeric(df["threshold_pct"], errors="coerce")
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)
    return df


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

    if "is_active_bin" in out.columns:
        out = out[out["is_active_bin"].fillna(False).astype(bool)].copy()

    out["service_deadline"] = out.apply(
        lambda row: interval_deadline(safe_float(row["days_since_last_service"]), horizon_days, row["stream"]),
        axis=1,
    )
    out["deadline_interval"] = out["service_deadline"]
    out["must_service_within_horizon"] = out["service_deadline"].notna()

    keep_cols = list(dict.fromkeys(list(state_df.columns) + ["service_deadline", "deadline_interval"]))
    return out[keep_cols].copy()


def run_subprocess(cmd: list[str], cwd: Path) -> None:
    print("\\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
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
        schedule["Serial"] = schedule["Serial"].apply(clean_serial)
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
    return pd.read_csv(fp) if fp.exists() else pd.DataFrame()


def update_state_after_day(state_df: pd.DataFrame, day0_schedule: pd.DataFrame, max_fill_factor: float = 1.50) -> pd.DataFrame:
    served_serials = set()
    if not day0_schedule.empty:
        served_serials = set(day0_schedule["Serial"].apply(clean_serial).tolist())

    new_state = state_df.copy()
    new_fill = []
    new_dsl = []

    for _, row in new_state.iterrows():
        serial = clean_serial(row["Serial"])
        fill = safe_float(row["fill_gal"])
        growth = safe_float(row["growth_gal_per_day"])
        cap = safe_float(row["bin_capacity_gal"])
        dsl = safe_float(row["days_since_last_service"])

        if serial in served_serials:
            next_fill = 0.0
            next_dsl = 0.0
        else:
            next_fill = min(cap * max_fill_factor, fill + growth)
            next_dsl = dsl + 1.0

        new_fill.append(next_fill)
        new_dsl.append(next_dsl)

    new_state["fill_gal"] = new_fill
    new_state["days_since_last_service"] = new_dsl
    return new_state


def validate_day_routes(processed: Path, day0_schedule: pd.DataFrame, truck_work_min: float) -> dict[str, Any]:
    route_plan = read_optional_csv(processed / "daily_route_plan.csv")
    route_stops = read_optional_csv(processed / "daily_route_stops.csv")
    route_summary = read_optional_csv(processed / "daily_route_summary.csv")

    if route_plan.empty or route_stops.empty:
        return {
            "routing_feasible": False,
            "route_plan_rows": 0,
            "route_summary_rows": 0,
            "routed_serials": 0,
            "missing_routed_serials": None,
            "extra_routed_serials": None,
            "dropped_stops": None,
            "volume_violations": None,
            "mass_violations": None,
            "time_violations": None,
        }

    dropped_stops = 0
    if not route_summary.empty and "dropped_stops" in route_summary.columns:
        dropped_stops = int(route_summary["dropped_stops"].fillna(0).sum())

    plan = route_plan.copy()
    plan["volume_ok"] = plan["route_gal"] <= plan["volume_capacity_effective"]
    plan["mass_ok"] = plan["route_lb"] <= plan["mass_capacity_effective"]
    plan["time_ok"] = plan["route_minutes"] <= float(truck_work_min)

    volume_violations = int((~plan["volume_ok"]).sum())
    mass_violations = int((~plan["mass_ok"]).sum())
    time_violations = int((~plan["time_ok"]).sum())

    scheduled_serials = set(day0_schedule["Serial"].apply(clean_serial).tolist()) if not day0_schedule.empty else set()

    stops = route_stops.copy()
    stops["Serials_at_stop"] = stops["Serials_at_stop"].fillna("").astype(str)

    routed_serials = set()
    for cell in stops.loc[stops["is_depot"] == 0, "Serials_at_stop"]:
        for serial in str(cell).split(";"):
            serial = clean_serial(serial)
            if serial:
                routed_serials.add(serial)

    missing = sorted(scheduled_serials - routed_serials)
    extra = sorted(routed_serials - scheduled_serials)

    feasible = (
        volume_violations == 0
        and mass_violations == 0
        and time_violations == 0
        and len(missing) == 0
        and len(extra) == 0
        and dropped_stops == 0
    )

    return {
        "routing_feasible": feasible,
        "route_plan_rows": int(len(route_plan)),
        "route_summary_rows": int(len(route_summary)),
        "routed_serials": int(len(routed_serials)),
        "missing_routed_serials": int(len(missing)),
        "extra_routed_serials": int(len(extra)),
        "dropped_stops": dropped_stops,
        "volume_violations": volume_violations,
        "mass_violations": mass_violations,
        "time_violations": time_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-day rolling-horizon experiment.")

    parser.add_argument("--projection-input", type=str, default=None)
    parser.add_argument("--num-days", type=int, default=5)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument(
        "--num-service-days",
        type=int,
        default=5,
        help=(
            "Number of weekday operating/service days inside the seven-calendar-day "
            "lookahead. Use 5 because drivers do not work weekends."
        ),
    )

    parser.add_argument("--num-trucks", type=int, default=4)
    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--sensitivity-truck-work-min", type=float, default=750.0)
    parser.add_argument("--max-overtime-min", type=float, default=0.0)
    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)
    parser.add_argument("--service-min-per-bin", type=float, default=4.0)
    parser.add_argument("--travel-min-between-stops", type=float, default=8.0)
    parser.add_argument("--dump-turnaround-min", type=float, default=17.0)
    parser.add_argument("--max-extra-dumps", type=int, default=3)
    parser.add_argument("--driver-wage-per-hour", type=float, default=40.0)
    parser.add_argument("--overflow-penalty", type=float, default=20.0)
    parser.add_argument("--tiny-pickup-threshold-gal", type=float, default=5.0)
    parser.add_argument("--tiny-pickup-penalty", type=float, default=0.0)

    parser.add_argument("--disable-compost-weekday-hygiene", action="store_true")
    parser.add_argument("--max-bins", type=int, default=None)

    parser.add_argument("--cbc-time-limit-sec", type=int, default=600)
    parser.add_argument("--cbc-gap-rel", type=float, default=0.10)

    parser.add_argument("--depot-lat", type=float, required=True)
    parser.add_argument("--depot-lng", type=float, required=True)
    parser.add_argument("--depot-label", type=str, default="Edwards Track")
    parser.add_argument("--network-type", type=str, default="drive_service")
    parser.add_argument("--routing-direction", type=str, default="undirected", choices=["directed", "undirected"])
    parser.add_argument("--query-margin-m", type=float, default=1000.0)
    parser.add_argument("--travel-speed-mph", type=float, default=6.0)
    parser.add_argument("--routing-time-limit-sec", type=int, default=300)
    parser.add_argument("--skip-routing", action="store_true")

    args = parser.parse_args()

    if args.num_service_days > args.horizon_days:
        raise ValueError("--num-service-days cannot exceed --horizon-days")
    if args.num_service_days < 1:
        raise ValueError("--num-service-days must be at least 1")

    root = repo_root()
    paths = ensure_dirs(root)
    processed = paths["processed"]
    outdir = paths["outdir"]

    base_df = load_base_projection(processed, args.projection_input)
    state_df = initialize_state(base_df)

    all_state_rows = []
    all_schedule_rows = []
    all_truck_stream_rows = []
    all_route_plan_rows = []
    all_route_stop_rows = []
    all_route_summary_rows = []
    day_metrics_rows = []

    py_exec = "python"

    for rolling_day in range(args.num_days):
        print(f"\\n========== ROLLING DAY {rolling_day + 1} / {args.num_days} ==========")

        snap = state_df.copy()
        snap["rolling_day"] = rolling_day + 1
        snap["fill_pct_start"] = np.where(
            snap["bin_capacity_gal"] > 0,
            100.0 * snap["fill_gal"] / snap["bin_capacity_gal"],
            0.0,
        )
        snap["true_overflow_start"] = snap["fill_pct_start"] > 100.0
        all_state_rows.extend(snap.to_dict(orient="records"))

        projection_df = rebuild_projection_from_state(state_df, args.horizon_days)
        temp_projection_fp = processed / "rolling_temp_projection_input.csv"
        projection_df.to_csv(temp_projection_fp, index=False)

        cmd_plan = [
            py_exec, "scripts/solve_7day_schedule.py",
            "--input", str(temp_projection_fp),
            "--num-trucks", str(args.num_trucks),
            "--horizon-days", str(args.horizon_days),
            "--num-service-days", str(args.num_service_days),
            "--truck-work-min", str(args.truck_work_min),
            "--sensitivity-truck-work-min", str(args.sensitivity_truck_work_min),
            "--truck-mass-lb", str(args.truck_mass_lb),
            "--waste-volume-gal", str(args.waste_volume_gal),
            "--compost-volume-gal", str(args.compost_volume_gal),
            "--recycling-volume-gal", str(args.recycling_volume_gal),
            "--service-min-per-bin", str(args.service_min_per_bin),
            "--travel-min-between-stops", str(args.travel_min_between_stops),
            "--dump-turnaround-min", str(args.dump_turnaround_min),
            "--max-extra-dumps", str(args.max_extra_dumps),
            "--max-overtime-min", str(args.max_overtime_min),
            "--driver-wage-per-hour", str(args.driver_wage_per_hour),
            "--overflow-penalty", str(args.overflow_penalty),
            "--tiny-pickup-threshold-gal", str(args.tiny_pickup_threshold_gal),
            "--tiny-pickup-penalty", str(args.tiny_pickup_penalty),
            "--cbc-time-limit-sec", str(args.cbc_time_limit_sec),
            "--cbc-gap-rel", str(args.cbc_gap_rel),
        ]

        if not args.disable_compost_weekday_hygiene:
            cmd_plan.append("--enforce-compost-weekday-hygiene")

        if args.max_bins is not None:
            cmd_plan.extend(["--max-bins", str(args.max_bins)])

        run_subprocess(cmd_plan, cwd=root)

        day0_schedule, day0_trucks = overwrite_day0_schedule_and_trucks(processed)

        if not day0_schedule.empty:
            day0_schedule = day0_schedule.copy()
            day0_schedule["rolling_day"] = rolling_day + 1
            all_schedule_rows.extend(day0_schedule.to_dict(orient="records"))

        if not day0_trucks.empty:
            day0_trucks = day0_trucks.copy()
            day0_trucks["rolling_day"] = rolling_day + 1
            all_truck_stream_rows.extend(day0_trucks.to_dict(orient="records"))

        if not args.skip_routing:
            cmd_matrix = [
                py_exec, "scripts/build_travel_matrix.py",
                "--input", str(temp_projection_fp),
                "--schedule", str(processed / "small_instance_service_schedule.csv"),
                "--route-day", "0",
                "--depot-lat", str(args.depot_lat),
                "--depot-lng", str(args.depot_lng),
                "--depot-label", str(args.depot_label),
                "--network-type", str(args.network_type),
                "--routing-direction", str(args.routing_direction),
                "--query-margin-m", str(args.query_margin_m),
                "--travel-speed-mph", str(args.travel_speed_mph),
            ]
            run_subprocess(cmd_matrix, cwd=root)

            cmd_route = [
                py_exec, "scripts/solve_daily_routing.py",
                "--projection-input", str(temp_projection_fp),
                "--truck-work-min", str(args.truck_work_min),
                "--routing-time-limit-sec", str(args.routing_time_limit_sec),
            ]
            run_subprocess(cmd_route, cwd=root)

        route_plan = read_optional_csv(processed / "daily_route_plan.csv")
        route_stops = read_optional_csv(processed / "daily_route_stops.csv")
        route_summary = read_optional_csv(processed / "daily_route_summary.csv")
        planning_summary = read_optional_csv(processed / "small_instance_planning_summary.csv")
        load_check = read_optional_csv(processed / "small_instance_truck_load_check.csv")

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
        route_minutes_today = float(route_plan["route_minutes"].sum()) if not route_plan.empty and "route_minutes" in route_plan.columns else 0.0
        overtime_today = float(load_check["overtime_min"].sum()) if not load_check.empty and "overtime_min" in load_check.columns else 0.0
        extra_dumps_today = float(load_check["extra_dumps"].sum()) if not load_check.empty and "extra_dumps" in load_check.columns else 0.0

        capacity_time_violations = 0
        if not load_check.empty:
            capacity_time_violations = int(
                (
                    (load_check["pickup_gal_used"] > load_check["pickup_gal_capacity_effective"])
                    | (load_check["pickup_lb_used"] > load_check["pickup_lb_capacity_effective"])
                    | (load_check["minutes_used_total"] > load_check["minutes_capacity_with_overtime"])
                ).sum()
            )

        overflow_today = int((state_df["fill_gal"] > state_df["bin_capacity_gal"]).sum())

        compost_bins = set(state_df.loc[state_df["stream"] == "Compostables", "Serial"].apply(clean_serial))
        served_compost = set()
        if not day0_schedule.empty:
            served_compost = set(day0_schedule.loc[day0_schedule["stream"] == "Compostables", "Serial"].apply(clean_serial))

        routing_validation = validate_day_routes(processed, day0_schedule, args.truck_work_min) if not args.skip_routing else {"routing_feasible": None}

        day_metrics_rows.append(
            {
                "rolling_day": rolling_day + 1,
                "horizon_days": args.horizon_days,
                "num_service_days": args.num_service_days,
                "service_days": ",".join(str(d) for d in range(args.num_service_days)),
                "nonservice_days": ",".join(str(d) for d in range(args.num_service_days, args.horizon_days)),
                "pickups_today": pickups_today,
                "pickup_gal_today": round(pickup_gal_today, 2),
                "pickup_lb_today": round(pickup_lb_today, 2),
                "routes_today": routes_today,
                "route_minutes_today": round(route_minutes_today, 2),
                "overtime_today": round(overtime_today, 2),
                "extra_dumps_today": round(extra_dumps_today, 2),
                "capacity_time_violations": capacity_time_violations,
                "overflow_bins_start_of_day": overflow_today,
                "compost_bins_total": len(compost_bins),
                "compost_bins_served_today": len(served_compost),
                "planner_status": planning_summary["status"].iloc[0] if not planning_summary.empty and "status" in planning_summary.columns else None,
                "planner_objective": float(planning_summary["objective_value"].iloc[0]) if not planning_summary.empty and "objective_value" in planning_summary.columns else None,
                "num_bins_in_instance": int(planning_summary["num_bins_in_instance"].iloc[0]) if not planning_summary.empty and "num_bins_in_instance" in planning_summary.columns else None,
                **routing_validation,
            }
        )

        state_df = update_state_after_day(state_df, day0_schedule)

    state_hist_df = pd.DataFrame(all_state_rows)
    rolling_sched_df = pd.DataFrame(all_schedule_rows)
    rolling_truck_stream_df = pd.DataFrame(all_truck_stream_rows)
    rolling_route_plan_df = pd.DataFrame(all_route_plan_rows)
    rolling_route_stops_df = pd.DataFrame(all_route_stop_rows)
    rolling_route_summary_df = pd.DataFrame(all_route_summary_rows)
    day_metrics_df = pd.DataFrame(day_metrics_rows)

    overall_summary = pd.DataFrame(
        [
            {
                "num_days": args.num_days,
                "horizon_days": args.horizon_days,
                "num_service_days": args.num_service_days,
                "service_days": ",".join(str(d) for d in range(args.num_service_days)),
                "nonservice_days": ",".join(str(d) for d in range(args.num_service_days, args.horizon_days)),
                "total_pickups": int(day_metrics_df["pickups_today"].sum()) if not day_metrics_df.empty else 0,
                "total_pickup_gal": round(float(day_metrics_df["pickup_gal_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
                "total_pickup_lb": round(float(day_metrics_df["pickup_lb_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
                "total_routes": int(day_metrics_df["routes_today"].sum()) if not day_metrics_df.empty else 0,
                "total_route_minutes": round(float(day_metrics_df["route_minutes_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
                "total_overtime": round(float(day_metrics_df["overtime_today"].sum()), 2) if not day_metrics_df.empty else 0.0,
                "total_capacity_time_violations": int(day_metrics_df["capacity_time_violations"].sum()) if not day_metrics_df.empty else 0,
                "avg_overflow_bins_start_of_day": round(float(day_metrics_df["overflow_bins_start_of_day"].mean()), 2) if not day_metrics_df.empty else 0.0,
                "routing_feasible_all_days": bool(day_metrics_df["routing_feasible"].fillna(False).all()) if not day_metrics_df.empty and "routing_feasible" in day_metrics_df.columns else None,
                "planner_type": "5day_rolling_horizon_reoptimization_with_7day_MIP_subproblems",
                "phase1_assumptions": (
                    "7-calendar-day lookahead, 5 weekday service days, 4 trucks, "
                    "480 min, no overtime, $20 overflow across all 7 days, "
                    "compost weekday hygiene"
                ),
            }
        ]
    )

    state_hist_df.to_csv(outdir / "rolling_day_state_history.csv", index=False)
    rolling_sched_df.to_csv(outdir / "rolling_day_schedule.csv", index=False)
    rolling_truck_stream_df.to_csv(outdir / "rolling_day_truck_streams.csv", index=False)
    rolling_route_plan_df.to_csv(outdir / "rolling_day_route_plan.csv", index=False)
    rolling_route_stops_df.to_csv(outdir / "rolling_day_route_stops.csv", index=False)
    rolling_route_summary_df.to_csv(outdir / "rolling_day_route_summary.csv", index=False)
    day_metrics_df.to_csv(outdir / "rolling_day_metrics.csv", index=False)
    overall_summary.to_csv(outdir / "rolling_5day_summary.csv", index=False)

    print("\\n========== 5-DAY DAILY METRICS ==========")
    if not day_metrics_df.empty:
        print(day_metrics_df.to_string(index=False))

    print("\\n========== 5-DAY SUMMARY ==========")
    if not overall_summary.empty:
        print(overall_summary.to_string(index=False))

    print(f"\\n[OK] Wrote combined outputs to: {outdir}")


if __name__ == "__main__":
    main()
