from __future__ import annotations

"""
make_bigbelly_figures.py
-------------------------------------------------------------
Generate report- and slide-ready figures for the Bigbelly project.

Reads outputs from:
- small_instance_service_schedule.csv
- small_instance_truck_load_check.csv
- small_instance_inventory_trajectory.csv
- daily_route_plan.csv
- rolling_horizon_5day/rolling_day_metrics.csv
- rolling_horizon_5day/rolling_5day_summary.csv

Outputs PNG figures under:
    outputs/figures/

Usage
-----
From repo root:

python scripts/make_bigbelly_figures.py

Optional:
python scripts/make_bigbelly_figures.py --processed-dir data/processed --outdir outputs/figures
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# ----------------------------
# Helpers
# ----------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(fp: Path) -> pd.DataFrame:
    if fp.exists():
        return pd.read_csv(fp)
    return pd.DataFrame()


def save_figure(fig: plt.Figure, outdir: Path, name: str) -> None:
    fp = outdir / name
    fig.tight_layout()
    fig.savefig(fp, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote: {fp}")


def add_box(ax, xy, w, h, text, fontsize=11):
    rect = Rectangle(xy, w, h, fill=False)
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, p1, p2):
    arrow = FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=12, linewidth=1.2
    )
    ax.add_patch(arrow)


# ----------------------------
# Figure 1: Two-phase flowchart
# ----------------------------

def fig_two_phase_framework(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    add_box(ax, (0.4, 2.0), 1.8, 1.0, "Bin projection\ninputs")
    add_box(ax, (2.7, 2.0), 2.2, 1.0, "Phase 1:\n7-Day Planning Model")
    add_box(ax, (5.4, 2.0), 1.8, 1.0, "Day-0 service\nextraction")
    add_box(ax, (7.7, 2.0), 2.0, 1.0, "Phase 2:\nDaily Routing")
    add_box(ax, (10.2, 2.0), 1.2, 1.0, "Executed\nroutes")

    add_box(ax, (7.7, 0.5), 2.0, 0.9, "State update:\nfill + days since\nlast service")
    add_box(ax, (2.7, 0.5), 2.2, 0.9, "Next rolling day:\nrebuild input +\nre-optimize")

    add_arrow(ax, (2.2, 2.5), (2.7, 2.5))
    add_arrow(ax, (4.9, 2.5), (5.4, 2.5))
    add_arrow(ax, (7.2, 2.5), (7.7, 2.5))
    add_arrow(ax, (9.7, 2.5), (10.2, 2.5))
    add_arrow(ax, (8.7, 2.0), (8.7, 1.4))
    add_arrow(ax, (7.7, 0.95), (4.9, 0.95))
    add_arrow(ax, (3.8, 1.4), (3.8, 2.0))

    ax.set_title("Two-Phase Bigbelly Optimization Framework", fontsize=14)
    save_figure(fig, outdir, "01_two_phase_framework.png")


# ----------------------------
# Figure 2: Rolling horizon timeline
# ----------------------------

def fig_rolling_horizon_timeline(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    y = 3.5
    ax.plot([0.7, 10.3], [y, y], linewidth=1.2)

    days = [
        ("Day 1", "Solve days 1–7\nExecute day 1"),
        ("Day 2", "Solve days 2–8\nExecute day 2"),
        ("Day 3", "Solve days 3–9\nExecute day 3"),
        ("Day 4", "Solve days 4–10\nExecute day 4"),
        ("Day 5", "Solve days 5–11\nExecute day 5"),
    ]
    xs = [1.3, 3.2, 5.1, 7.0, 8.9]

    for x, (label, text) in zip(xs, days):
        ax.plot([x, x], [y - 0.12, y + 0.12], linewidth=1.2)
        ax.text(x, y + 0.35, label, ha="center", va="bottom", fontsize=11)
        ax.text(x, y - 0.45, text, ha="center", va="top", fontsize=10)

    ax.set_title("Five-Day Rolling-Horizon Evaluation Procedure", fontsize=14)
    save_figure(fig, outdir, "02_rolling_horizon_timeline.png")


# ----------------------------
# Figure 3: Example inventory evolution
# ----------------------------

def fig_sample_inventory(processed_dir: Path, outdir: Path) -> None:
    inv = read_csv_if_exists(processed_dir / "small_instance_inventory_trajectory.csv")
    if inv.empty:
        print("[WARN] small_instance_inventory_trajectory.csv not found; skipping sample inventory figure.")
        return

    inv = inv.copy()

    # Standardize columns
    if "Serial" not in inv.columns:
        print("[WARN] Serial column missing in inventory trajectory.")
        return

    inv["Serial"] = inv["Serial"].astype(str).str.replace(".0", "", regex=False).str.strip()

    for col in ["pickup_flag", "day", "inventory_pct_start", "inventory_gal_start", "threshold_pct"]:
        if col in inv.columns:
            inv[col] = pd.to_numeric(inv[col], errors="coerce")

    if "pickup_flag" not in inv.columns:
        inv["pickup_flag"] = 0.0

    if "day" not in inv.columns or "inventory_pct_start" not in inv.columns:
        print("[WARN] Required plotting columns missing in inventory trajectory.")
        return

    # Keep only rows that can actually be plotted
    inv_valid = inv.dropna(subset=["day", "inventory_pct_start"]).copy()
    if inv_valid.empty:
        print("[WARN] No valid rows with day and inventory_pct_start.")
        return

    # Build selection stats from bins that have enough valid points
    serial_stats = (
        inv_valid.groupby("Serial", as_index=False)
        .agg(
            num_points=("day", "count"),
            num_services=("pickup_flag", "sum"),
            fill_range=("inventory_pct_start", lambda s: s.max() - s.min()),
            max_fill=("inventory_pct_start", "max"),
        )
    )

    serial_stats = serial_stats[serial_stats["num_points"] >= 2].copy()
    if serial_stats.empty:
        print("[WARN] No bins with at least 2 valid plotted points.")
        return

    # Prefer bins with 1-3 services and good variation
    candidates = serial_stats[
        (serial_stats["num_services"] >= 1) & (serial_stats["num_services"] <= 3)
    ].sort_values(["fill_range", "max_fill"], ascending=[False, False])

    if not candidates.empty:
        sample_serial = str(candidates.iloc[0]["Serial"])
    else:
        sample_serial = str(
            serial_stats.sort_values(["fill_range", "max_fill"], ascending=[False, False])
            .iloc[0]["Serial"]
        )

    df = inv_valid[inv_valid["Serial"] == sample_serial].sort_values("day").copy()

    if df.empty or len(df) < 2:
        print(f"[WARN] Chosen sample bin {sample_serial} has insufficient valid points.")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(df["day"], df["inventory_pct_start"], marker="o")
    ax.set_xlabel("Day in Horizon")
    ax.set_ylabel("Fill at Start of Day (%)")
    ax.set_title(f"Example Bin Fill Evolution Over the Planning Horizon\nSerial {sample_serial}")

    for _, row in df.iterrows():
        if row["pickup_flag"] > 0.5:
            ax.annotate(
                "Service",
                (row["day"], row["inventory_pct_start"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    if "threshold_pct" in df.columns:
        thr = df["threshold_pct"].dropna()
        if not thr.empty:
            threshold_val = float(thr.iloc[0])
            ax.axhline(threshold_val, linestyle="--", linewidth=1.2)
            ax.text(
                df["day"].min(),
                threshold_val + 1,
                "Threshold",
                fontsize=9,
                va="bottom",
            )

    save_figure(fig, outdir, "03_sample_bin_inventory.png")


# ----------------------------
# Figure 4: Day-0 service totals by stream
# ----------------------------

def fig_day0_service_by_stream(processed_dir: Path, outdir: Path) -> None:
    sched = read_csv_if_exists(processed_dir / "small_instance_service_schedule.csv")
    if sched.empty:
        print("[WARN] small_instance_service_schedule.csv not found; skipping day-0 service figures.")
        return

    sched = sched.copy()
    sched["service_day"] = pd.to_numeric(sched["service_day"], errors="coerce")
    sched["pickup_gal"] = pd.to_numeric(sched["pickup_gal"], errors="coerce").fillna(0.0)
    sched["pickup_lb"] = pd.to_numeric(sched["pickup_lb"], errors="coerce").fillna(0.0)

    day0 = sched[sched["service_day"] == 0].copy()
    agg = day0.groupby("stream", as_index=False).agg(
        bins_serviced=("Serial", "count"),
        pickup_gal=("pickup_gal", "sum"),
        pickup_lb=("pickup_lb", "sum"),
    )

    if agg.empty:
        print("[WARN] No day-0 service rows found.")
        return

    for col, ylabel, fname in [
        ("bins_serviced", "Bins Serviced", "04a_day0_bins_serviced_by_stream.png"),
        ("pickup_gal", "Pickup Gallons", "04b_day0_pickup_gal_by_stream.png"),
        ("pickup_lb", "Pickup Pounds", "04c_day0_pickup_lb_by_stream.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(agg["stream"], agg[col])
        ax.set_xlabel("Stream")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Day-0 Service Totals by Stream: {ylabel}")
        save_figure(fig, outdir, fname)


# ----------------------------
# Figure 5: Day-0 route workload by stream
# ----------------------------

def fig_day0_route_by_stream(processed_dir: Path, outdir: Path) -> None:
    route = read_csv_if_exists(processed_dir / "daily_route_plan.csv")
    if route.empty:
        print("[WARN] daily_route_plan.csv not found; skipping routing workload figures.")
        return

    route = route.copy()
    for c in ["route_minutes", "num_stops", "route_gal", "route_lb"]:
        if c in route.columns:
            route[c] = pd.to_numeric(route[c], errors="coerce").fillna(0)

    for col, ylabel, fname in [
        ("route_minutes", "Route Minutes", "05a_day0_route_minutes_by_stream.png"),
        ("num_stops", "Number of Stops", "05b_day0_stops_by_stream.png"),
        ("route_gal", "Route Gallons", "05c_day0_route_gal_by_stream.png"),
    ]:
        if col not in route.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(route["stream"], route[col])
        ax.set_xlabel("Stream")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Day-0 Route Workload by Stream: {ylabel}")
        save_figure(fig, outdir, fname)


# ----------------------------
# Figure 6: Truck utilization vs capacity
# ----------------------------

def fig_truck_utilization(processed_dir: Path, outdir: Path) -> None:
    load = read_csv_if_exists(processed_dir / "small_instance_truck_load_check.csv")
    if load.empty:
        print("[WARN] small_instance_truck_load_check.csv not found; skipping truck utilization figures.")
        return

    load = load.copy()
    load["day"] = pd.to_numeric(load["day"], errors="coerce")
    load = load[load["day"] == 0].copy()

    pairs = [
        ("pickup_gal_used", "pickup_gal_capacity_effective", "Gallons", "06a_truck_util_gallons.png"),
        ("pickup_lb_used", "pickup_lb_capacity_effective", "Pounds", "06b_truck_util_pounds.png"),
        ("minutes_used_total", "minutes_capacity_with_overtime", "Minutes", "06c_truck_util_minutes.png"),
    ]

    for used_col, cap_col, ylabel, fname in pairs:
        if used_col not in load.columns or cap_col not in load.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = range(len(load))
        ax.bar(x, load[used_col], label="Used")
        ax.bar(x, load[cap_col], fill=False, linewidth=1.2, label="Capacity")
        ax.set_xticks(list(x))
        ax.set_xticklabels(load["truck"].astype(str))
        ax.set_xlabel("Truck")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Day-0 Truck Utilization vs Effective Capacity: {ylabel}")
        ax.legend()
        save_figure(fig, outdir, fname)


# ----------------------------
# Figure 7: Rolling 5-day daily metrics
# ----------------------------

def fig_rolling_daily_metrics(processed_dir: Path, outdir: Path) -> None:
    fp = processed_dir / "rolling_horizon_5day" / "rolling_day_metrics.csv"
    metrics = read_csv_if_exists(fp)
    if metrics.empty:
        print("[WARN] rolling_day_metrics.csv not found; skipping rolling-day metrics figures.")
        return

    metrics = metrics.copy()
    metrics["rolling_day"] = pd.to_numeric(metrics["rolling_day"], errors="coerce")

    cols = [
        ("pickups_today", "Pickups", "07a_rolling_pickups.png"),
        ("route_minutes_today", "Route Minutes", "07b_rolling_route_minutes.png"),
        ("overflow_bins_start_of_day", "Overflow Bins at Start of Day", "07c_rolling_overflow.png"),
        ("overtime_today", "Overtime Minutes", "07d_rolling_overtime.png"),
    ]

    for col, ylabel, fname in cols:
        if col not in metrics.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(metrics["rolling_day"], metrics[col], marker="o")
        ax.set_xlabel("Rolling Day")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Operational Metrics Across the 5-Day Rolling Horizon: {ylabel}")
        save_figure(fig, outdir, fname)


# ----------------------------
# Figure 8: 5-day summary dashboard
# ----------------------------

def fig_rolling_summary_dashboard(processed_dir: Path, outdir: Path) -> None:
    fp = processed_dir / "rolling_horizon_5day" / "rolling_5day_summary.csv"
    summary = read_csv_if_exists(fp)
    if summary.empty:
        print("[WARN] rolling_5day_summary.csv not found; skipping summary dashboard.")
        return

    row = summary.iloc[0].to_dict()

    display_rows = [
        ["Number of rolling days", row.get("num_days", "")],
        ["Total pickups", row.get("total_pickups", "")],
        ["Total pickup volume (gal)", row.get("total_pickup_gal", "")],
        ["Total pickup weight (lb)", row.get("total_pickup_lb", "")],
        ["Total routes", row.get("total_routes", "")],
        ["Total route minutes", row.get("total_route_minutes", "")],
        ["Total overtime", row.get("total_overtime", "")],
        ["Avg overflow bins at start of day", row.get("avg_overflow_bins_start_of_day", "")],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax.set_title("Aggregate Results from the 5-Day Rolling-Horizon Experiment", fontsize=14, pad=12)
    save_figure(fig, outdir, "08_rolling_summary_dashboard.png")


# ----------------------------
# Figure 9: Extra dumps by truck
# ----------------------------

def fig_extra_dumps_by_truck(processed_dir: Path, outdir: Path) -> None:
    load = read_csv_if_exists(processed_dir / "small_instance_truck_load_check.csv")
    if load.empty:
        print("[WARN] small_instance_truck_load_check.csv not found; skipping extra dump figure.")
        return

    load = load.copy()
    load["day"] = pd.to_numeric(load["day"], errors="coerce")
    load["extra_dumps"] = pd.to_numeric(load["extra_dumps"], errors="coerce").fillna(0)
    load = load[load["day"] == 0].copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(load["truck"], load["extra_dumps"])
    ax.set_xlabel("Truck")
    ax.set_ylabel("Extra Dump Cycles")
    ax.set_title("Extra Dump Cycles Used in the Day-0 Plan")
    save_figure(fig, outdir, "09_extra_dumps_by_truck.png")


# ----------------------------
# Figure 10: Planning vs routing consistency table
# ----------------------------

def fig_planning_routing_consistency(processed_dir: Path, outdir: Path) -> None:
    load = read_csv_if_exists(processed_dir / "small_instance_truck_load_check.csv")
    route = read_csv_if_exists(processed_dir / "daily_route_plan.csv")
    if load.empty or route.empty:
        print("[WARN] Missing planner/routing outputs; skipping planning vs routing table.")
        return

    load = load.copy()
    route = route.copy()

    load["day"] = pd.to_numeric(load["day"], errors="coerce")
    route["day"] = pd.to_numeric(route["day"], errors="coerce")

    load = load[load["day"] == 0].copy()
    route = route[route["day"] == 0].copy()

    if "assigned_stream" in load.columns and "stream" not in load.columns:
        load = load.rename(columns={"assigned_stream": "stream"})

    merged = load.merge(
        route[["stream", "truck", "route_minutes", "num_stops", "route_gal"]],
        on=["stream", "truck"],
        how="outer",
    )

    needed_cols = ["stream", "truck", "pickup_gal_used", "route_gal", "route_minutes", "extra_dumps"]
    for col in needed_cols:
        if col not in merged.columns:
            merged[col] = ""

    display = merged[needed_cols].fillna("")

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=display.values.tolist(),
        colLabels=[
            "Stream",
            "Truck",
            "Planner Pickup Gal",
            "Route Gal",
            "Route Minutes",
            "Extra Dumps",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Consistency Between Planning Assignments and Day-0 Routing", fontsize=14, pad=10)
    save_figure(fig, outdir, "10_planning_routing_consistency.png")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Bigbelly report figures.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--outdir", type=str, default="outputs/figures")
    args = parser.parse_args()

    root = repo_root()
    processed_dir = root / args.processed_dir
    outdir = root / args.outdir
    ensure_outdir(outdir)

    fig_two_phase_framework(outdir)
    fig_rolling_horizon_timeline(outdir)
    fig_sample_inventory(processed_dir, outdir)
    fig_day0_service_by_stream(processed_dir, outdir)
    fig_day0_route_by_stream(processed_dir, outdir)
    fig_truck_utilization(processed_dir, outdir)
    fig_rolling_daily_metrics(processed_dir, outdir)
    fig_rolling_summary_dashboard(processed_dir, outdir)
    fig_extra_dumps_by_truck(processed_dir, outdir)
    fig_planning_routing_consistency(processed_dir, outdir)

    print(f"\n[OK] Finished writing figures to: {outdir}")


if __name__ == "__main__":
    main()