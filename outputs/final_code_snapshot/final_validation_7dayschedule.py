from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

summary_fp = PROCESSED / "small_instance_planning_summary.csv"
schedule_fp = PROCESSED / "small_instance_service_schedule.csv"
load_fp = PROCESSED / "small_instance_truck_load_check.csv"
inventory_fp = PROCESSED / "small_instance_inventory_trajectory.csv"

required_files = [summary_fp, schedule_fp, load_fp, inventory_fp]
missing_files = [str(p) for p in required_files if not p.exists()]

if missing_files:
    raise FileNotFoundError("Missing required output files:\n" + "\n".join(missing_files))

summary = pd.read_csv(summary_fp)
sched = pd.read_csv(schedule_fp)
load = pd.read_csv(load_fp)
inv = pd.read_csv(inventory_fp)

print("\n" + "=" * 80)
print("FINAL 7-CALENDAR-DAY / 5-SERVICE-DAY PLANNING VALIDATION")
print("=" * 80)

# ---------------------------------------------------------------------
# 1. Planning summary
# ---------------------------------------------------------------------

print("\n[1] PLANNING SUMMARY")
print("-" * 80)
print(summary.T.to_string())

# Pull service-day definitions from planning summary when available.
if "service_days" in summary.columns:
    service_days = {
        int(x) for x in str(summary["service_days"].iloc[0]).split(",") if str(x).strip() != ""
    }
else:
    service_days = {0, 1, 2, 3, 4}

if "nonservice_days" in summary.columns:
    nonservice_days = {
        int(x) for x in str(summary["nonservice_days"].iloc[0]).split(",") if str(x).strip() != ""
    }
else:
    nonservice_days = {5, 6}

# ---------------------------------------------------------------------
# 2. Capacity and time feasibility
# ---------------------------------------------------------------------

print("\n[2] CAPACITY / PAYLOAD / TIME FEASIBILITY")
print("-" * 80)

required_load_cols = [
    "pickup_gal_used",
    "pickup_gal_capacity_effective",
    "pickup_lb_used",
    "pickup_lb_capacity_effective",
    "minutes_used_total",
    "minutes_capacity_with_overtime",
]

missing = [c for c in required_load_cols if c not in load.columns]
if missing:
    raise KeyError(f"Missing load-check columns: {missing}")

capacity_time_viol = load[
    (load["pickup_gal_used"] > load["pickup_gal_capacity_effective"]) |
    (load["pickup_lb_used"] > load["pickup_lb_capacity_effective"]) |
    (load["minutes_used_total"] > load["minutes_capacity_with_overtime"])
].copy()

print("Capacity/time violations:", len(capacity_time_viol))

if len(capacity_time_viol):
    cols = [
        "day",
        "truck",
        "assigned_stream",
        "pickup_gal_used",
        "pickup_gal_capacity_effective",
        "pickup_lb_used",
        "pickup_lb_capacity_effective",
        "extra_dumps",
        "minutes_used_total",
        "minutes_capacity_with_overtime",
        "overtime_min",
    ]
    cols = [c for c in cols if c in capacity_time_viol.columns]
    print(capacity_time_viol[cols].to_string(index=False))
else:
    print("None")

# ---------------------------------------------------------------------
# 3. Pickup count summary
# ---------------------------------------------------------------------

print("\n[3] PICKUP COUNT SUMMARY")
print("-" * 80)

print("Total scheduled pickup rows:", len(sched))

if "pickup_gal" in sched.columns:
    print("Total pickup gallons:", round(sched["pickup_gal"].fillna(0).sum(), 2))

if "pickup_lb" in sched.columns:
    print("Total pickup pounds:", round(sched["pickup_lb"].fillna(0).sum(), 2))

if "service_day" in sched.columns:
    sched["service_day"] = pd.to_numeric(sched["service_day"], errors="coerce")
    print("\nPickups by day:")
    print(sched.groupby("service_day").size())

if "stream" in sched.columns:
    print("\nPickups by stream:")
    print(sched.groupby("stream").size())

# ---------------------------------------------------------------------
# 4. Weekday service-only check
# ---------------------------------------------------------------------

print("\n[4] WEEKDAY SERVICE-ONLY CHECK")
print("-" * 80)

if "service_day" not in sched.columns:
    raise KeyError("Schedule file missing service_day column.")

if "day" in load.columns:
    load["day"] = pd.to_numeric(load["day"], errors="coerce")

weekend_sched = sched[sched["service_day"].isin(nonservice_days)].copy()

print("Service days:", sorted(service_days))
print("Non-service days:", sorted(nonservice_days))
print("Scheduled pickups on non-service days:", len(weekend_sched))

if len(weekend_sched):
    print("\nNon-service day scheduled pickups:")
    cols = ["Serial", "stream", "service_day", "truck", "pickup_gal", "pickup_lb"]
    cols = [c for c in cols if c in weekend_sched.columns]
    print(weekend_sched[cols].to_string(index=False))
else:
    print("None")

bad_load = pd.DataFrame()
if "day" in load.columns:
    bad_load = load[
        load["day"].isin(nonservice_days)
        & (
            (load["pickup_gal_used"].fillna(0) > 0)
            | (load["pickup_lb_used"].fillna(0) > 0)
            | (load["minutes_used_total"].fillna(0) > 0)
        )
    ].copy()

    print("Truck load/work rows on non-service days:", len(bad_load))

    if len(bad_load):
        cols = [
            "day",
            "truck",
            "assigned_stream",
            "pickup_gal_used",
            "pickup_lb_used",
            "minutes_used_total",
            "extra_dumps",
            "overtime_min",
        ]
        cols = [c for c in cols if c in bad_load.columns]
        print(bad_load[cols].to_string(index=False))
    else:
        print("None")

# ---------------------------------------------------------------------
# 5. Compost weekday hygiene check
# ---------------------------------------------------------------------

print("\n[5] COMPOST WEEKDAY HYGIENE CHECK")
print("-" * 80)

required_compost_cols = ["Serial", "stream", "day", "pickup_flag"]
missing = [c for c in required_compost_cols if c not in inv.columns]
if missing:
    raise KeyError(f"Missing inventory columns for compost hygiene check: {missing}")

inv["Serial"] = inv["Serial"].astype(str).str.replace(r"\.0$", "", regex=True)
inv["day"] = pd.to_numeric(inv["day"], errors="coerce")

compost = inv[inv["stream"] == "Compostables"].copy()

compost_bins = set(compost["Serial"].unique())

served_compost_bins = set(
    compost.loc[
        (compost["pickup_flag"] > 0) & (compost["day"].isin(service_days)),
        "Serial",
    ].unique()
)

missing_compost = sorted(compost_bins - served_compost_bins)

print("Total compost bins:", len(compost_bins))
print("Compost bins serviced at least once within service days:", len(served_compost_bins))
print("Compost bins not serviced within service days:", len(missing_compost))

if missing_compost:
    print("\nCompost bins not serviced:")
    for serial in missing_compost:
        print(serial)
else:
    print("\nAll compost bins were serviced at least once within the weekday service window.")

# ---------------------------------------------------------------------
# 6. True overflow summary
# ---------------------------------------------------------------------

print("\n[6] TRUE OVERFLOW SUMMARY")
print("-" * 80)

required_inv_cols = [
    "Serial",
    "stream",
    "day",
    "inventory_pct_start",
    "pickup_flag",
    "pickup_gal",
]

missing = [c for c in required_inv_cols if c not in inv.columns]
if missing:
    raise KeyError(f"Missing inventory columns: {missing}")

inv["true_overflow"] = (inv["inventory_pct_start"] > 100).astype(int)

inv["served_true_overflow"] = (
    (inv["true_overflow"] == 1) & (inv["pickup_flag"] > 0)
).astype(int)

inv["unserved_true_overflow"] = (
    (inv["true_overflow"] == 1) & (inv["pickup_flag"] == 0)
).astype(int)

overflow_by_day = inv.groupby("day")[
    ["true_overflow", "served_true_overflow", "unserved_true_overflow"]
].sum()

print("True overflow by day:")
print(overflow_by_day)

print("\nOverflow totals:")
print("True overflow bin-days:", int(inv["true_overflow"].sum()))
print("Served true overflow bin-days:", int(inv["served_true_overflow"].sum()))
print("Unserved true overflow bin-days:", int(inv["unserved_true_overflow"].sum()))

print("\nUnserved true overflow rows:")
unserved_cols = [
    "Serial",
    "stream",
    "day",
    "inventory_pct_start",
    "pickup_flag",
    "pickup_gal",
]

unserved = inv[inv["unserved_true_overflow"] == 1][unserved_cols].sort_values(
    ["day", "stream", "Serial"]
)

if len(unserved):
    print(unserved.to_string(index=False))
else:
    print("None")

# ---------------------------------------------------------------------
# 7. Overflow-cost consistency check
# ---------------------------------------------------------------------

print("\n[7] OVERFLOW-COST CONSISTENCY CHECK")
print("-" * 80)

required_summary_cols = ["overflow_penalty", "overflow_cost"]
missing = [c for c in required_summary_cols if c not in summary.columns]
if missing:
    raise KeyError(f"Missing planning summary columns for overflow-cost check: {missing}")

overflow_penalty = float(summary["overflow_penalty"].iloc[0])
reported_overflow_cost = float(summary["overflow_cost"].iloc[0])

if "overflow_slack_gal" not in inv.columns:
    raise KeyError("Inventory file missing overflow_slack_gal column.")

model_overflow_days = int((inv["overflow_slack_gal"] > 1e-6).sum())
true_overflow_days = int(inv["true_overflow"].sum())
expected_model_overflow_cost = model_overflow_days * overflow_penalty
expected_true_overflow_cost = true_overflow_days * overflow_penalty

overflow_cost_matches_model_flags = (
    abs(reported_overflow_cost - expected_model_overflow_cost) < 1e-6
)

print("Overflow penalty:", overflow_penalty)
print("Model overflow-slack bin-days:", model_overflow_days)
print("True overflow bin-days:", true_overflow_days)
print("Reported overflow cost:", reported_overflow_cost)
print("Expected cost using model overflow flags:", expected_model_overflow_cost)
print("Expected cost using true overflow days:", expected_true_overflow_cost)
print("Reported cost matches model overflow flags:", overflow_cost_matches_model_flags)

if "overflow_penalty_applies_to" in summary.columns:
    print("Overflow penalty applies to:", summary["overflow_penalty_applies_to"].iloc[0])

if "expected_overflow_cost_from_model_flags" in summary.columns:
    print(
        "Expected overflow cost stored in planning summary:",
        summary["expected_overflow_cost_from_model_flags"].iloc[0],
    )

print("\nModel overflow-slack days by day:")
print((inv.assign(model_overflow_flag=(inv["overflow_slack_gal"] > 1e-6).astype(int))
         .groupby("day")["model_overflow_flag"]
         .sum()))

# ---------------------------------------------------------------------
# 8. Tiny pickup summary
# ---------------------------------------------------------------------

print("\n[8] TINY PICKUP SUMMARY")
print("-" * 80)

if "pickup_gal" not in sched.columns:
    print("No pickup_gal column found.")
else:
    tiny = sched[(sched["pickup_gal"] > 0) & (sched["pickup_gal"] < 5)].copy()

    print("Tiny pickups under 5 gallons:", len(tiny))

    if len(tiny):
        tiny_cols = [
            "Serial",
            "stream",
            "service_day",
            "truck",
            "inventory_gal_before_service",
            "inventory_pct_before_service",
            "pickup_gal",
            "pickup_lb",
            "interval_deadline",
        ]
        tiny_cols = [c for c in tiny_cols if c in tiny.columns]
        print(tiny[tiny_cols].to_string(index=False))

        if "interval_deadline" in tiny.columns:
            deadline_tiny = tiny[
                tiny["interval_deadline"].notna()
                & (tiny["service_day"] == tiny["interval_deadline"])
            ].copy()

            late_tiny = tiny[
                tiny["interval_deadline"].notna()
                & (tiny["service_day"] > tiny["interval_deadline"])
            ].copy()

            no_deadline_tiny = tiny[tiny["interval_deadline"].isna()].copy()

            print("\nTiny pickup classification:")
            print("Deadline-day tiny pickups:", len(deadline_tiny))
            print("Tiny pickups after displayed deadline:", len(late_tiny))
            print("Tiny pickups with no displayed deadline:", len(no_deadline_tiny))

            if len(late_tiny):
                print("\nTiny pickups after displayed deadline:")
                print(late_tiny[tiny_cols].to_string(index=False))
    else:
        print("None")

# ---------------------------------------------------------------------
# 9. Repeat tiny pickup investigation
# ---------------------------------------------------------------------

print("\n[9] REPEAT TINY PICKUP CHECK")
print("-" * 80)

if "pickup_gal" in sched.columns and "Serial" in sched.columns:
    tiny = sched[(sched["pickup_gal"] > 0) & (sched["pickup_gal"] < 5)].copy()

    if len(tiny):
        repeat_rows = []

        for serial in tiny["Serial"].astype(str).unique():
            rows = sched[sched["Serial"].astype(str) == serial].sort_values("service_day")
            tiny_rows = tiny[tiny["Serial"].astype(str) == serial].sort_values("service_day")
            prior_pickups = rows[rows["pickup_gal"] >= 5]

            for _, tr in tiny_rows.iterrows():
                prior = prior_pickups[prior_pickups["service_day"] < tr["service_day"]]
                repeat_rows.append(
                    {
                        "Serial": serial,
                        "stream": tr.get("stream"),
                        "tiny_service_day": tr.get("service_day"),
                        "tiny_pickup_gal": tr.get("pickup_gal"),
                        "prior_non_tiny_pickups_before_tiny": len(prior),
                        "prior_service_days": ",".join(prior["service_day"].astype(str).tolist()),
                    }
                )

        repeat_df = pd.DataFrame(repeat_rows)
        print(repeat_df.to_string(index=False))
    else:
        print("No tiny pickups.")
else:
    print("Required columns missing for repeat tiny pickup check.")

# ---------------------------------------------------------------------
# 10. Final validation conclusion
# ---------------------------------------------------------------------

print("\n[10] FINAL VALIDATION CONCLUSION")
print("-" * 80)

cap_ok = len(capacity_time_viol) == 0
weekday_service_ok = len(weekend_sched) == 0
weekday_load_ok = len(bad_load) == 0 if "bad_load" in locals() else True
compost_ok = len(missing_compost) == 0
unserved_overflow_total = int(inv["unserved_true_overflow"].sum())
tiny_count = (
    int(len(sched[(sched["pickup_gal"] > 0) & (sched["pickup_gal"] < 5)]))
    if "pickup_gal" in sched.columns
    else 0
)

print("Capacity/time feasible:", cap_ok)
print("Weekday service-only satisfied:", weekday_service_ok)
print("No truck work on non-service days:", weekday_load_ok)
print("Compost weekday hygiene satisfied:", compost_ok)
print("Compost bins not serviced:", len(missing_compost))
print("Overflow cost charged over model overflow flags:", overflow_cost_matches_model_flags)
print("Unserved true overflow bin-days:", unserved_overflow_total)
print("Tiny pickups under 5 gal:", tiny_count)

if cap_ok and weekday_service_ok and weekday_load_ok and compost_ok and overflow_cost_matches_model_flags:
    print(
        "\nConclusion: The seven-calendar-day schedule satisfies truck volume, "
        "payload, and shift-time constraints; restricts service to the five "
        "weekday operating days; satisfies compost weekday hygiene; and charges "
        "overflow penalties consistently across the full seven-calendar-day lookahead."
    )
else:
    print(
        "\nConclusion: The schedule has validation issues. Check capacity/time, "
        "non-service day pickups, truck work on Days 5-6, compost hygiene, or "
        "overflow-cost accounting."
    )

print(
    "Overflow should be reported using true overflow, defined as start-of-day "
    "inventory above 100% of nominal bin capacity."
)

print(
    "Tiny pickups should be described separately. If most are deadline-driven, "
    "they are hygiene-rule exceptions rather than unnecessary objective-driven pickups."
)

print("=" * 80)
