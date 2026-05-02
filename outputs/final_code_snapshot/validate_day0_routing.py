from pathlib import Path
import pandas as pd

TRUCK_WORK_MIN = 480


def clean_serial(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

schedule_fp = PROCESSED / "small_instance_service_schedule.csv"
plan_fp = PROCESSED / "daily_route_plan.csv"
stops_fp = PROCESSED / "daily_route_stops.csv"
summary_fp = PROCESSED / "daily_route_summary.csv"

for fp in [schedule_fp, plan_fp, stops_fp, summary_fp]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing required file: {fp}")

schedule = pd.read_csv(schedule_fp)
plan = pd.read_csv(plan_fp)
stops = pd.read_csv(stops_fp)
summary = pd.read_csv(summary_fp)

print("\n" + "=" * 80)
print("DAY 0 ROUTING VALIDATION")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Route summary
# ---------------------------------------------------------------------

print("\n[1] Route summary")
print("-" * 80)
print(summary.to_string(index=False))

dropped_total = 0
if "dropped_stops" in summary.columns:
    dropped_total = int(summary["dropped_stops"].fillna(0).sum())
else:
    print("[WARN] daily_route_summary.csv has no dropped_stops column.")

print("\nDropped stops:", dropped_total)

# ---------------------------------------------------------------------
# 2. Route plan
# ---------------------------------------------------------------------

print("\n[2] Route plan")
print("-" * 80)
print(plan.to_string(index=False))

# ---------------------------------------------------------------------
# 3. Capacity, mass, and time validation
# ---------------------------------------------------------------------

print("\n[3] Capacity, mass, and time validation")
print("-" * 80)

required_plan_cols = [
    "route_gal",
    "volume_capacity_effective",
    "route_lb",
    "mass_capacity_effective",
    "route_minutes",
]

missing_cols = [c for c in required_plan_cols if c not in plan.columns]
if missing_cols:
    raise KeyError(f"daily_route_plan.csv missing required columns: {missing_cols}")

plan["volume_ok"] = plan["route_gal"] <= plan["volume_capacity_effective"]
plan["mass_ok"] = plan["route_lb"] <= plan["mass_capacity_effective"]
plan["time_ok"] = plan["route_minutes"] <= TRUCK_WORK_MIN

cols = [
    "day",
    "stream",
    "truck",
    "routing_mode",
    "num_stops",
    "route_gal",
    "volume_capacity_effective",
    "volume_ok",
    "route_lb",
    "mass_capacity_effective",
    "mass_ok",
    "route_minutes",
    "time_ok",
    "extra_dumps_from_phase1",
]
cols = [c for c in cols if c in plan.columns]
print(plan[cols].to_string(index=False))

volume_viol = int((~plan["volume_ok"]).sum())
mass_viol = int((~plan["mass_ok"]).sum())
time_viol = int((~plan["time_ok"]).sum())

print("\nViolations:")
print("Volume violations:", volume_viol)
print("Mass violations:", mass_viol)
print("Time violations:", time_viol)

# ---------------------------------------------------------------------
# 4. Scheduled-vs-routed bin validation
# ---------------------------------------------------------------------

print("\n[4] Scheduled-vs-routed bin validation")
print("-" * 80)

required_schedule_cols = ["Serial", "service_day"]
required_stops_cols = ["is_depot", "Serials_at_stop"]

missing_schedule_cols = [c for c in required_schedule_cols if c not in schedule.columns]
missing_stops_cols = [c for c in required_stops_cols if c not in stops.columns]

if missing_schedule_cols:
    raise KeyError(f"small_instance_service_schedule.csv missing columns: {missing_schedule_cols}")

if missing_stops_cols:
    raise KeyError(f"daily_route_stops.csv missing columns: {missing_stops_cols}")

schedule["Serial"] = schedule["Serial"].apply(clean_serial)
schedule["service_day"] = pd.to_numeric(schedule["service_day"], errors="coerce")

stops["Serials_at_stop"] = stops["Serials_at_stop"].fillna("").astype(str)

day0_sched = schedule[schedule["service_day"] == 0].copy()
scheduled_serials = set(day0_sched["Serial"].apply(clean_serial))

routed_serials = set()
for cell in stops.loc[stops["is_depot"] == 0, "Serials_at_stop"]:
    for serial in str(cell).split(";"):
        serial = clean_serial(serial)
        if serial:
            routed_serials.add(serial)

missing = sorted(scheduled_serials - routed_serials)
extra = sorted(routed_serials - scheduled_serials)

print("Scheduled Day 0 bins:", len(scheduled_serials))
print("Routed Day 0 bins:", len(routed_serials))
print("Missing scheduled bins from routes:", len(missing))
print("Extra routed bins not in Day 0 schedule:", len(extra))

if missing:
    print("\nMissing scheduled bins:")
    for serial in missing:
        print(serial)

if extra:
    print("\nExtra routed bins:")
    for serial in extra:
        print(serial)

# ---------------------------------------------------------------------
# 5. Final conclusion
# ---------------------------------------------------------------------

print("\n[5] Final conclusion")
print("-" * 80)

all_ok = (
    volume_viol == 0
    and mass_viol == 0
    and time_viol == 0
    and len(missing) == 0
    and len(extra) == 0
    and dropped_total == 0
)

print("Day 0 routing feasible:", all_ok)
print("Dropped stops:", dropped_total)

if all_ok:
    print(
        f"Conclusion: All Day 0 scheduled bins were routed, no stops were dropped, "
        f"and all routes satisfy volume, mass, and {TRUCK_WORK_MIN}-minute truck-day limits."
    )
else:
    print(
        "Conclusion: Day 0 routing has validation issues. Review missing bins, "
        "extra bins, dropped stops, or capacity/time violations before reporting."
    )

print("=" * 80)