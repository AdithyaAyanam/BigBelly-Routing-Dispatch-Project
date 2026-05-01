python - <<'PY'
import pandas as pd
from pathlib import Path

p = Path("data/processed")

load = pd.read_csv(p / "small_instance_truck_load_check.csv")
route = pd.read_csv(p / "daily_route_plan.csv")

print("truck_load_check columns:")
print(load.columns.tolist())

print("\ndaily_route_plan columns:")
print(route.columns.tolist())

# Try likely planning-minute columns
planning_candidates = [
    "minutes_used_total",
    "planning_minutes",
    "total_minutes",
    "work_minutes",
    "route_minutes_proxy",
    "minutes_used",
]

routing_candidates = [
    "route_minutes",
    "total_route_minutes",
]

planning_col = next((c for c in planning_candidates if c in load.columns), None)
routing_col = next((c for c in routing_candidates if c in route.columns), None)

if planning_col is None:
    raise ValueError("Could not find a planning-minute column in small_instance_truck_load_check.csv")

if routing_col is None:
    raise ValueError("Could not find a routing-minute column in daily_route_plan.csv")

planning_min = pd.to_numeric(load[planning_col], errors="coerce").fillna(0).sum()
routing_min = pd.to_numeric(route[routing_col], errors="coerce").fillna(0).sum()

print("\nPlanning-vs-routing comparison")
print("--------------------------------")
print(f"Planning minute column: {planning_col}")
print(f"Routing minute column:  {routing_col}")
print(f"Planning minutes:       {planning_min:.2f}")
print(f"Routing minutes:        {routing_min:.2f}")

if routing_min > 0:
    print(f"Planning-to-routing ratio: {planning_min / routing_min:.2f}")
else:
    print("Planning-to-routing ratio: undefined because routing minutes = 0")
PY