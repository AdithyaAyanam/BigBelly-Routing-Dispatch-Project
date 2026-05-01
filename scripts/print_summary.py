from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f"[WARN] Missing file: {path}")
    return None


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_planning_summary() -> None:
    path = PROCESSED_DIR / "small_instance_planning_summary.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return

    print_section("7-DAY PLANNING SUMMARY")
    row = df.iloc[0]

    fields = [
        "status",
        "objective_value",
        "horizon_days",
        "num_bins_in_instance",
        "num_trucks",
        "truck_work_min_input",
        "effective_truck_work_min",
        "resource_model",
        "normal_lift_trucks",
        "backup_nonlift_trucks",
        "total_pickups",
        "total_extra_dumps",
        "total_overtime_min",
        "overflow_bin_days",
        "overflow_flag_days",
        "total_overflow_slack_gal",
        "dump_penalty",
        "overflow_penalty",
        "cbc_time_limit_sec",
        "cbc_gap_rel",
        "require_routable",
    ]

    for field in fields:
        if field in df.columns:
            print(f"{field}: {row[field]}")


def print_daily_route_summary() -> None:
    path = PROCESSED_DIR / "daily_route_summary.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return

    print_section("DAILY ROUTE SUMMARY")
    print(df.to_string(index=False))


def print_daily_route_plan() -> None:
    path = PROCESSED_DIR / "daily_route_plan.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return

    print_section("DAILY ROUTE PLAN")
    preview_cols = [
        "day",
        "stream",
        "truck",
        "routing_mode",
        "num_stops",
        "route_gal",
        "route_lb",
        "route_minutes",
        "volume_capacity_effective",
        "mass_capacity_effective",
        "extra_dumps_from_phase1",
    ]

    cols = [c for c in preview_cols if c in df.columns]
    print(df[cols].to_string(index=False))


def print_rolling_summary() -> None:
    path = PROCESSED_DIR / "rolling_horizon_5day" / "rolling_day_metrics.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return

    print_section("5-DAY ROLLING-HORIZON SUMMARY")
    print(df.to_string(index=False))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print_section("5-DAY ROLLING-HORIZON TOTALS")
        totals = df[numeric_cols].sum(numeric_only=True)
        print(totals.to_string())


def print_key_output_files() -> None:
    print_section("KEY OUTPUT FILES")

    expected_files = [
        "bin_7day_projection_inputs.csv",
        "routing_nodes.csv",
        "bin_stop_lookup.csv",
        "travel_matrix_long.csv",
        "travel_matrix_wide.csv",
        "small_instance_service_schedule.csv",
        "small_instance_truck_streams.csv",
        "small_instance_truck_load_check.csv",
        "small_instance_inventory_trajectory.csv",
        "small_instance_planning_summary.csv",
        "daily_route_plan.csv",
        "daily_route_stops.csv",
        "daily_route_summary.csv",
        "rolling_day_metrics.csv",
    ]

    for filename in expected_files:
        path = PROCESSED_DIR / filename
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{status}: {filename}")
def print_planning_vs_routing_ratio() -> None:
    load_path = PROCESSED_DIR / "small_instance_truck_load_check.csv"
    route_path = PROCESSED_DIR / "daily_route_plan.csv"

    load = read_csv_if_exists(load_path)
    route = read_csv_if_exists(route_path)

    if load is None or route is None or load.empty or route.empty:
        return

    print_section("PLANNING VS ROUTING RATIO")

    if "minutes_used_total" not in load.columns:
        print("[WARN] small_instance_truck_load_check.csv missing minutes_used_total")
        return

    if "route_minutes" not in route.columns:
        print("[WARN] daily_route_plan.csv missing route_minutes")
        return

    # Day-0 comparison because daily_route_plan.csv represents the routed day-0 implementation.
    if "day" in load.columns:
        load_day0 = load[load["day"] == 0].copy()
    else:
        load_day0 = load.copy()

    if "day" in route.columns:
        route_day0 = route[route["day"] == 0].copy()
    else:
        route_day0 = route.copy()

    planning_minutes = pd.to_numeric(
        load_day0["minutes_used_total"], errors="coerce"
    ).fillna(0).sum()

    routing_minutes = pd.to_numeric(
        route_day0["route_minutes"], errors="coerce"
    ).fillna(0).sum()

    if routing_minutes <= 0:
        print("[WARN] Routing minutes are zero, so the ratio is undefined.")
        return

    ratio = planning_minutes / routing_minutes

    print(f"Day-0 planning minutes: {planning_minutes:.2f}")
    print(f"Day-0 routed minutes: {routing_minutes:.2f}")
    print(f"Planning-to-routing ratio: {ratio:.2f}")

    print(
        "\nInterpretation: The planning-to-routing ratio compares the workload "
        "minutes charged in the Phase 1 planning model against the OR-Tools "
        "route minutes generated in Phase 2 for the implemented day-0 routes. "
        "A ratio closer to 1.0 indicates stronger alignment between the planning "
        "approximation and routed implementation time."
    )

def main() -> None:
    print_key_output_files()
    print_planning_summary()
    print_daily_route_plan()
    print_daily_route_summary()
    print_planning_vs_routing_ratio()
    print_rolling_summary()


if __name__ == "__main__":
    main()