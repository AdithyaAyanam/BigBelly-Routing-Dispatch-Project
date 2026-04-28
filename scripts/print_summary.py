from pathlib import Path
import pandas as pd

# -----------------------------
# Daily route summary
# -----------------------------
route_fp = Path("data/processed/daily_route_summary.csv")

if route_fp.exists():
    route = pd.read_csv(route_fp)

    print("\n=== DAILY ROUTE SUMMARY ===")
    print(route.to_string(index=False))

    print("\nColumns available:")
    print(route.columns.tolist())

    print("\nTotal scheduled stops:")
    if "scheduled_stops" in route.columns:
        print(route["scheduled_stops"].sum())
    elif "num_stops" in route.columns:
        print(route["num_stops"].sum())
    elif "scheduled_bins" in route.columns:
        print(route["scheduled_bins"].sum())
    else:
        print("No stop-count column found.")

    print("\nTotal route/service minutes:")
    if "route_minutes" in route.columns:
        print(route["route_minutes"].sum())
    elif "total_route_minutes" in route.columns:
        print(route["total_route_minutes"].sum())
    elif "total_service_min_only" in route.columns:
        print(route["total_service_min_only"].sum())
    else:
        print("No route-minutes column found.")
else:
    print("\n=== DAILY ROUTE SUMMARY ===")
    print("daily_route_summary.csv not found yet. Run:")
    print("python scripts/solve_daily_routing.py --truck-work-min 480")


# -----------------------------
# 5-day rolling horizon summary
# -----------------------------

summary = pd.read_csv("data/processed/rolling_horizon_5day/rolling_5day_summary.csv")
days = pd.read_csv("data/processed/rolling_horizon_5day/rolling_5day_day_summary.csv")

print("\n=== 5-DAY ROLLING SUMMARY ===")
print(summary.to_string(index=False))

print("\n=== DAY-BY-DAY ROLLING SUMMARY ===")
print(days.to_string(index=False))
 

 