from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ROLLING = ROOT / "data" / "processed" / "rolling_horizon_5day"

metrics_fp = ROLLING / "rolling_day_metrics.csv"
schedule_fp = ROLLING / "rolling_day_schedule.csv"
route_plan_fp = ROLLING / "rolling_day_route_plan.csv"
route_stops_fp = ROLLING / "rolling_day_route_stops.csv"
route_summary_fp = ROLLING / "rolling_day_route_summary.csv"
overall_fp = ROLLING / "rolling_5day_summary.csv"

required_files = [
    metrics_fp,
    schedule_fp,
    route_plan_fp,
    route_stops_fp,
    route_summary_fp,
    overall_fp,
]

missing_files = [str(fp) for fp in required_files if not fp.exists()]
if missing_files:
    raise FileNotFoundError(
        "Missing required rolling-horizon output files:\n"
        + "\n".join(missing_files)
        + "\n\nRun scripts/run_5day_rolling_horizon.py first."
    )


def clean_serial(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


metrics = pd.read_csv(metrics_fp)
schedule = pd.read_csv(schedule_fp)
route_plan = pd.read_csv(route_plan_fp)
route_stops = pd.read_csv(route_stops_fp)
route_summary = pd.read_csv(route_summary_fp)
overall = pd.read_csv(overall_fp)

print("\n" + "=" * 90)
print("FINAL ROLLING-HORIZON VALIDATION")
print("=" * 90)

print("\n[1] OVERALL SUMMARY")
print("-" * 90)
print(overall.T.to_string())

print("\n[2] DAILY METRICS")
print("-" * 90)
print(metrics.to_string(index=False))

print("\n[3] CAPACITY / PAYLOAD / TIME FEASIBILITY")
print("-" * 90)

if "capacity_time_violations" not in metrics.columns:
    raise KeyError("rolling_day_metrics.csv missing capacity_time_violations column.")

total_capacity_time_viol = int(metrics["capacity_time_violations"].fillna(0).sum())

print(metrics[["rolling_day", "capacity_time_violations"]].to_string(index=False))
print("\nTotal capacity/time violations:", total_capacity_time_viol)

print("\n[4] ROUTING FEASIBILITY")
print("-" * 90)

routing_cols = [
    "rolling_day",
    "routing_feasible",
    "routed_serials",
    "missing_routed_serials",
    "extra_routed_serials",
    "dropped_stops",
    "volume_violations",
    "mass_violations",
    "time_violations",
]
routing_cols = [c for c in routing_cols if c in metrics.columns]

if routing_cols:
    print(metrics[routing_cols].to_string(index=False))
else:
    print("No routing validation columns found in rolling_day_metrics.csv.")

routing_feasible_all_days = (
    bool(metrics["routing_feasible"].fillna(False).all())
    if "routing_feasible" in metrics.columns
    else False
)

total_missing = (
    int(metrics["missing_routed_serials"].fillna(0).sum())
    if "missing_routed_serials" in metrics.columns
    else None
)

total_extra = (
    int(metrics["extra_routed_serials"].fillna(0).sum())
    if "extra_routed_serials" in metrics.columns
    else None
)

total_dropped = (
    int(metrics["dropped_stops"].fillna(0).sum())
    if "dropped_stops" in metrics.columns
    else None
)

total_volume_viol = (
    int(metrics["volume_violations"].fillna(0).sum())
    if "volume_violations" in metrics.columns
    else None
)

total_mass_viol = (
    int(metrics["mass_violations"].fillna(0).sum())
    if "mass_violations" in metrics.columns
    else None
)

total_time_viol = (
    int(metrics["time_violations"].fillna(0).sum())
    if "time_violations" in metrics.columns
    else None
)

print("\nRouting feasible all days:", routing_feasible_all_days)
print("Total missing routed serials:", total_missing)
print("Total extra routed serials:", total_extra)
print("Total dropped stops:", total_dropped)
print("Total route volume violations:", total_volume_viol)
print("Total route mass violations:", total_mass_viol)
print("Total route time violations:", total_time_viol)

print("\n[5] SCHEDULED-VS-ROUTED BIN VALIDATION BY ROLLING DAY")
print("-" * 90)

required_schedule_cols = ["rolling_day", "Serial"]
required_stop_cols = ["rolling_day", "is_depot", "Serials_at_stop"]

missing_schedule_cols = [c for c in required_schedule_cols if c not in schedule.columns]
missing_stop_cols = [c for c in required_stop_cols if c not in route_stops.columns]

if missing_schedule_cols:
    raise KeyError(f"rolling_day_schedule.csv missing columns: {missing_schedule_cols}")

if missing_stop_cols:
    raise KeyError(f"rolling_day_route_stops.csv missing columns: {missing_stop_cols}")

schedule["Serial"] = schedule["Serial"].apply(clean_serial)
route_stops["Serials_at_stop"] = route_stops["Serials_at_stop"].fillna("").astype(str)

scheduled_vs_routed_rows = []

for rolling_day in sorted(schedule["rolling_day"].dropna().unique()):
    day_sched = schedule[schedule["rolling_day"] == rolling_day].copy()
    day_stops = route_stops[
        (route_stops["rolling_day"] == rolling_day)
        & (route_stops["is_depot"] == 0)
    ].copy()

    scheduled_serials = set(day_sched["Serial"].apply(clean_serial))

    routed_serials = set()
    for cell in day_stops["Serials_at_stop"]:
        for serial in str(cell).split(";"):
            serial = clean_serial(serial)
            if serial:
                routed_serials.add(serial)

    missing = sorted(scheduled_serials - routed_serials)
    extra = sorted(routed_serials - scheduled_serials)

    scheduled_vs_routed_rows.append(
        {
            "rolling_day": int(rolling_day),
            "scheduled_bins": len(scheduled_serials),
            "routed_bins": len(routed_serials),
            "missing_bins": len(missing),
            "extra_bins": len(extra),
            "missing_serials": ";".join(missing),
            "extra_serials": ";".join(extra),
        }
    )

svr = pd.DataFrame(scheduled_vs_routed_rows)

if not svr.empty:
    print(svr.drop(columns=["missing_serials", "extra_serials"]).to_string(index=False))
else:
    print("No scheduled-vs-routed rows found.")

bad_svr = (
    svr[(svr["missing_bins"] > 0) | (svr["extra_bins"] > 0)]
    if not svr.empty
    else pd.DataFrame()
)

if len(bad_svr):
    print("\nScheduled-vs-routed problems:")
    print(bad_svr.to_string(index=False))
else:
    print("\nAll rolling-day scheduled bins match routed bins.")

print("\n[6] ROUTE SUMMARY BY ROLLING DAY / STREAM")
print("-" * 90)

route_summary_cols = [
    "rolling_day",
    "day",
    "stream",
    "scheduled_bins",
    "scheduled_stops",
    "routed_stops",
    "dropped_stops",
    "active_trucks",
    "selected_mode",
    "total_pickup_gal",
    "total_pickup_lb",
    "total_effective_volume_cap",
    "total_effective_mass_cap",
]
route_summary_cols = [c for c in route_summary_cols if c in route_summary.columns]

if route_summary_cols:
    print(route_summary[route_summary_cols].to_string(index=False))
else:
    print(route_summary.to_string(index=False))

print("\n[7] ROUTE PLAN DETAILS")
print("-" * 90)

route_plan_cols = [
    "rolling_day",
    "day",
    "stream",
    "truck",
    "routing_mode",
    "num_stops",
    "route_gal",
    "volume_capacity_effective",
    "route_lb",
    "mass_capacity_effective",
    "route_minutes",
    "extra_dumps_from_phase1",
]
route_plan_cols = [c for c in route_plan_cols if c in route_plan.columns]

if route_plan_cols:
    print(route_plan[route_plan_cols].to_string(index=False))
else:
    print(route_plan.to_string(index=False))

print("\n[8] PICKUP / OVERFLOW TREND")
print("-" * 90)

trend_cols = [
    "rolling_day",
    "horizon_days",
    "num_service_days",
    "pickups_today",
    "pickup_gal_today",
    "pickup_lb_today",
    "routes_today",
    "route_minutes_today",
    "overtime_today",
    "extra_dumps_today",
    "capacity_time_violations",
    "overflow_bins_start_of_day",
    "compost_bins_total",
    "compost_bins_served_today",
    "planner_status",
    "planner_objective",
    "target_avg_interstop_min",
]
trend_cols = [c for c in trend_cols if c in metrics.columns]

print(metrics[trend_cols].to_string(index=False))

print("\n[9] FINAL VALIDATION CONCLUSION")
print("-" * 90)

capacity_ok = total_capacity_time_viol == 0
routing_ok = routing_feasible_all_days
missing_ok = total_missing == 0 if total_missing is not None else False
extra_ok = total_extra == 0 if total_extra is not None else False
dropped_ok = total_dropped == 0 if total_dropped is not None else False
route_volume_ok = total_volume_viol == 0 if total_volume_viol is not None else False
route_mass_ok = total_mass_viol == 0 if total_mass_viol is not None else False
route_time_ok = total_time_viol == 0 if total_time_viol is not None else False
scheduled_vs_routed_ok = len(bad_svr) == 0

final_ok = (
    capacity_ok
    and routing_ok
    and missing_ok
    and extra_ok
    and dropped_ok
    and route_volume_ok
    and route_mass_ok
    and route_time_ok
    and scheduled_vs_routed_ok
)

print("Capacity/time feasible across rolling days:", capacity_ok)
print("Routing feasible across rolling days:", routing_ok)
print("No missing routed serials:", missing_ok)
print("No extra routed serials:", extra_ok)
print("No dropped stops:", dropped_ok)
print("No route volume violations:", route_volume_ok)
print("No route mass violations:", route_mass_ok)
print("No route time violations:", route_time_ok)
print("Scheduled-vs-routed validation passed:", scheduled_vs_routed_ok)
print("Rolling horizon valid:", final_ok)

if final_ok:
    print(
        "\nConclusion: The rolling-horizon experiment is valid. "
        "Each rolling day solves a seven-calendar-day planning problem, "
        "executes only the Day 0 schedule, rebuilds the scaled Day 0 travel matrix, "
        "and routes all scheduled bins without dropped stops, missing bins, "
        "extra bins, or capacity/time violations."
    )
else:
    print(
        "\nConclusion: The rolling-horizon experiment has validation issues. "
        "Review the failed checks above before reporting the rolling-horizon result."
    )

print("=" * 90)
