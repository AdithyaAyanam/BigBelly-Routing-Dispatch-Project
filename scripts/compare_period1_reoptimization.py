from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Original frozen 7-day baseline
 
BASELINE_DIR = ROOT / "outputs" / "full_7day_plan_for_period_comparison"

# Rolling-horizon output
ROLLING_DIR = PROCESSED / "rolling_horizon_5day"

original_fp = BASELINE_DIR / "small_instance_service_schedule.csv"
rolling_fp = ROLLING_DIR / "rolling_day_schedule.csv"
route_plan_fp = ROLLING_DIR / "rolling_day_route_plan.csv"
outdir = ROOT / "outputs" / "period1_reoptimization_comparison"
outdir.mkdir(parents=True, exist_ok=True)


def clean_serial(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


for fp in [original_fp, rolling_fp]:
    if not fp.exists():
        raise FileNotFoundError(f"Missing required file: {fp}")


original = pd.read_csv(original_fp)
rolling = pd.read_csv(rolling_fp)

original["Serial"] = original["Serial"].apply(clean_serial)
rolling["Serial"] = rolling["Serial"].apply(clean_serial)

original["service_day"] = pd.to_numeric(original["service_day"], errors="coerce")
rolling["rolling_day"] = pd.to_numeric(rolling["rolling_day"], errors="coerce")

# Original 7-day plan period 1: plan over periods 0-6, service_day == 1
original_p1 = original[original["service_day"] == 1].copy()

# Rolling Day 2 executed Day 0: re-solved over periods 1-7
rolling_day2 = rolling[rolling["rolling_day"] == 2].copy()

original_set = set(original_p1["Serial"])
rolling_set = set(rolling_day2["Serial"])

unchanged = sorted(original_set & rolling_set)
removed = sorted(original_set - rolling_set)
added = sorted(rolling_set - original_set)

summary = pd.DataFrame(
    [
        {
            "original_period1_bins": len(original_set),
            "rolling_day2_bins": len(rolling_set),
            "unchanged_bins": len(unchanged),
            "removed_from_period1_after_resolve": len(removed),
            "added_to_period1_after_resolve": len(added),
            "jaccard_similarity": (
                len(unchanged) / len(original_set | rolling_set)
                if len(original_set | rolling_set) > 0
                else 1.0
            ),
        }
    ]
)

details_rows = []

for serial in unchanged:
    details_rows.append(
        {
            "Serial": serial,
            "comparison_status": "unchanged",
        }
    )

for serial in removed:
    details_rows.append(
        {
            "Serial": serial,
            "comparison_status": "in_original_period1_not_in_rolling_day2",
        }
    )

for serial in added:
    details_rows.append(
        {
            "Serial": serial,
            "comparison_status": "added_in_rolling_day2",
        }
    )

details = pd.DataFrame(details_rows)

# Add stream and pickup info from both sources when available
orig_cols = ["Serial", "stream", "truck", "pickup_gal", "pickup_lb", "inventory_pct_before_service"]
orig_cols = [c for c in orig_cols if c in original_p1.columns]
roll_cols = ["Serial", "stream", "truck", "pickup_gal", "pickup_lb", "inventory_pct_before_service"]
roll_cols = [c for c in roll_cols if c in rolling_day2.columns]

orig_info = original_p1[orig_cols].copy()
roll_info = rolling_day2[roll_cols].copy()

orig_info = orig_info.rename(
    columns={
        "stream": "original_stream",
        "truck": "original_truck",
        "pickup_gal": "original_pickup_gal",
        "pickup_lb": "original_pickup_lb",
        "inventory_pct_before_service": "original_inventory_pct",
    }
)

roll_info = roll_info.rename(
    columns={
        "stream": "rolling_stream",
        "truck": "rolling_truck",
        "pickup_gal": "rolling_pickup_gal",
        "pickup_lb": "rolling_pickup_lb",
        "inventory_pct_before_service": "rolling_inventory_pct",
    }
)

details = details.merge(orig_info, on="Serial", how="left")
details = details.merge(roll_info, on="Serial", how="left")

summary_fp = outdir / "period1_reoptimization_summary.csv"
details_fp = outdir / "period1_reoptimization_details.csv"

summary.to_csv(summary_fp, index=False)
details.to_csv(details_fp, index=False)

print("\n" + "=" * 90)
print("PERIOD 1 RE-OPTIMIZATION COMPARISON")
print("=" * 90)

print("\nOriginal plan: periods 0-6, service_day == 1")
print("Rolling Day 2: shifted plan periods 1-7, executed Day 0")

print("\n[1] Summary")
print(summary.to_string(index=False))

print("\n[2] Counts by status")
print(details["comparison_status"].value_counts().to_string())

if len(removed):
    print("\n[3] Bins originally planned for period 1 but not selected after re-solve")
    print(details[details["comparison_status"] == "in_original_period1_not_in_rolling_day2"].to_string(index=False))
else:
    print("\n[3] No bins were removed from the period-1 service set after re-solving.")

if len(added):
    print("\n[4] Bins added to period 1 after re-solve")
    print(details[details["comparison_status"] == "added_in_rolling_day2"].to_string(index=False))
else:
    print("\n[4] No bins were added to the period-1 service set after re-solving.")

print("\nSaved:")
print(summary_fp)
print(details_fp)
print("=" * 90)