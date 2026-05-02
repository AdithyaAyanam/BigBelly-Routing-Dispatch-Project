from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

COMPARISON_DIR = ROOT / "outputs" / "rolling_vs_historical_comparison"
ROLLING_DIR = ROOT / "data" / "processed" / "rolling_horizon_5day"
OUTPUT_DIR = ROOT / "outputs" / "final_charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_FP = COMPARISON_DIR / "daily_model_vs_historical.csv"
STREAM_FP = COMPARISON_DIR / "stream_model_vs_historical.csv"
ROUTE_FP = COMPARISON_DIR / "model_route_summary.csv"
OVERFLOW_FP = COMPARISON_DIR / "rolling_overflow_trend.csv"
ROLLING_METRICS_FP = ROLLING_DIR / "rolling_day_metrics.csv"


# ============================================================
# Helpers
# ============================================================

def require_file(fp: Path):
    if not fp.exists():
        raise FileNotFoundError(f"Missing required file: {fp}")


def save_fig(name: str):
    out = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote: {out}")


def clean_day_labels(df):
    out = df.copy()
    if "historical_date" in out.columns:
        out["day_label"] = out["historical_date"].astype(str)
    else:
        out["day_label"] = "Rolling Day " + out["rolling_day"].astype(str)
    return out


# ============================================================
# Load data
# ============================================================

for fp in [DAILY_FP, STREAM_FP, ROUTE_FP, OVERFLOW_FP, ROLLING_METRICS_FP]:
    require_file(fp)

daily = pd.read_csv(DAILY_FP)
stream = pd.read_csv(STREAM_FP)
route = pd.read_csv(ROUTE_FP)
overflow = pd.read_csv(OVERFLOW_FP)
metrics = pd.read_csv(ROLLING_METRICS_FP)

daily = clean_day_labels(daily)


# ============================================================
# Chart 1: Two-phase methodology diagram
# ============================================================

def chart_methodology_diagram():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")

    boxes = [
        ("Historical + Asset Data", 0.05, 0.55),
        ("Forecast Fill Levels\n7-Day Projection", 0.23, 0.55),
        ("Phase 1: 7-Day MIP\n5 Service Days", 0.42, 0.55),
        ("Execute Day 0\nSchedule Only", 0.61, 0.55),
        ("Phase 2: Scaled\nDay 0 Routing", 0.78, 0.55),
        ("Update State\nRoll Forward", 0.42, 0.12),
    ]

    for text, x, y in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", linewidth=1.2, facecolor="white"),
            transform=ax.transAxes,
        )

    arrows = [
        ((0.13, 0.55), (0.18, 0.55)),
        ((0.31, 0.55), (0.36, 0.55)),
        ((0.50, 0.55), (0.56, 0.55)),
        ((0.68, 0.55), (0.73, 0.55)),
        ((0.78, 0.43), (0.52, 0.20)),
        ((0.42, 0.24), (0.42, 0.43)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

    ax.set_title(
        "Rolling-Horizon Bigbelly Decision-Support Workflow",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    save_fig("01_methodology_diagram.png")


# ============================================================
# Chart 2: Model vs historical pickups by day
# ============================================================

def chart_pickups_by_day():
    plot_df = daily.copy()

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        plot_df["pickups_today"],
        width,
        label="Model pickups",
    )

    ax.bar(
        x + width / 2,
        plot_df["historical_pickups"],
        width,
        label="Historical pickups",
    )

    ax.set_title("Model vs Historical Pickups by Day", fontsize=14, fontweight="bold")
    ax.set_xlabel("Historical Comparison Date")
    ax.set_ylabel("Number of Pickups")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["day_label"], rotation=0)
    ax.legend()

    for i, v in enumerate(plot_df["pickups_today"]):
        ax.text(i - width / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    for i, v in enumerate(plot_df["historical_pickups"]):
        ax.text(i + width / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    save_fig("02_model_vs_historical_pickups_by_day.png")


# ============================================================
# Chart 3: Model vs historical route minutes
# ============================================================

def chart_route_minutes_by_day():
    plot_df = daily.copy()

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        plot_df["route_minutes_today"],
        width,
        label="Model route minutes",
    )

    ax.bar(
        x + width / 2,
        plot_df["historical_route_minutes_proxy"],
        width,
        label="Historical route-minutes proxy",
    )

    ax.set_title(
        "Model vs Historical Route Minutes by Day",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Historical Comparison Date")
    ax.set_ylabel("Route Minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["day_label"], rotation=0)
    ax.legend()

    note = "Historical proxy = historical pickups × (4 min service + 8 min travel)"
    ax.text(
        0.5,
        -0.18,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    for i, v in enumerate(plot_df["route_minutes_today"]):
        ax.text(i - width / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    for i, v in enumerate(plot_df["historical_route_minutes_proxy"]):
        ax.text(i + width / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    save_fig("03_model_vs_historical_route_minutes_by_day.png")


# ============================================================
# Chart 4: Overflow trend across rolling days
# ============================================================

def chart_overflow_trend():
    plot_df = overflow.copy()

    if "overflow_bins_start" not in plot_df.columns:
        if "overflow_bins_start_of_day" in plot_df.columns:
            plot_df = plot_df.rename(
                columns={"overflow_bins_start_of_day": "overflow_bins_start"}
            )
        else:
            raise KeyError("Overflow file missing overflow_bins_start column.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        plot_df["rolling_day"],
        plot_df["overflow_bins_start"],
        marker="o",
        linewidth=2,
    )

    ax.set_title(
        "Start-of-Day Overflow Trend Across Rolling Days",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rolling Day")
    ax.set_ylabel("Overflow Bins at Start of Day")
    ax.set_xticks(plot_df["rolling_day"])

    for _, row in plot_df.iterrows():
        ax.text(
            row["rolling_day"],
            row["overflow_bins_start"],
            f"{int(row['overflow_bins_start'])}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_fig("04_overflow_trend.png")


# ============================================================
# Chart 5: Pickups by stream, model vs historical
# ============================================================

def chart_pickups_by_stream():
    plot_df = stream.copy()

    # Build combined day-stream label
    plot_df["label"] = (
        "Day "
        + plot_df["rolling_day"].astype(int).astype(str)
        + "\n"
        + plot_df["stream"].astype(str)
    )

    x = np.arange(len(plot_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(
        x - width / 2,
        plot_df["model_pickups"],
        width,
        label="Model pickups",
    )

    ax.bar(
        x + width / 2,
        plot_df["historical_pickups"],
        width,
        label="Historical pickups",
    )

    ax.set_title(
        "Pickups by Stream: Model vs Historical",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rolling Day and Stream")
    ax.set_ylabel("Number of Pickups")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax.legend()

    save_fig("05_pickups_by_stream_model_vs_historical.png")


# ============================================================
# Chart 6: Route minutes by truck with 480-minute limit
# ============================================================

def chart_route_minutes_by_truck():
    # This uses rolling route-level model summary if it has truck.
    route_plan_fp = ROLLING_DIR / "rolling_day_route_plan.csv"
    require_file(route_plan_fp)

    route_plan = pd.read_csv(route_plan_fp)

    required = ["rolling_day", "stream", "truck", "route_minutes"]
    missing = [c for c in required if c not in route_plan.columns]
    if missing:
        raise KeyError(f"rolling_day_route_plan.csv missing columns: {missing}")

    route_plan["label"] = (
        "Day "
        + route_plan["rolling_day"].astype(int).astype(str)
        + "\n"
        + route_plan["stream"].astype(str)
        + "\n"
        + route_plan["truck"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(route_plan))

    ax.bar(x, route_plan["route_minutes"])
    ax.axhline(480, linestyle="--", linewidth=1.5, label="480-minute truck-day limit")

    ax.set_title(
        "Route Minutes by Truck with 480-Minute Limit",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Rolling Day / Stream / Truck")
    ax.set_ylabel("Route Minutes")
    ax.set_xticks(x)
    ax.set_xticklabels(route_plan["label"], rotation=45, ha="right")
    ax.legend()

    for i, v in enumerate(route_plan["route_minutes"]):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=9)

    save_fig("06_route_minutes_by_truck_with_limit.png")


# ============================================================
# Chart 7: Validation dashboard table
# ============================================================

def chart_validation_dashboard():
    total_capacity = int(metrics["capacity_time_violations"].fillna(0).sum())

    total_dropped = (
        int(metrics["dropped_stops"].fillna(0).sum())
        if "dropped_stops" in metrics.columns
        else 0
    )

    total_missing = (
        int(metrics["missing_routed_serials"].fillna(0).sum())
        if "missing_routed_serials" in metrics.columns
        else 0
    )

    total_extra = (
        int(metrics["extra_routed_serials"].fillna(0).sum())
        if "extra_routed_serials" in metrics.columns
        else 0
    )

    total_volume = (
        int(metrics["volume_violations"].fillna(0).sum())
        if "volume_violations" in metrics.columns
        else 0
    )

    total_mass = (
        int(metrics["mass_violations"].fillna(0).sum())
        if "mass_violations" in metrics.columns
        else 0
    )

    total_time = (
        int(metrics["time_violations"].fillna(0).sum())
        if "time_violations" in metrics.columns
        else 0
    )

    routing_feasible = (
        bool(metrics["routing_feasible"].fillna(False).all())
        if "routing_feasible" in metrics.columns
        else False
    )

    table_data = [
        ["Capacity/time violations", total_capacity],
        ["Dropped stops", total_dropped],
        ["Missing routed bins", total_missing],
        ["Extra routed bins", total_extra],
        ["Route volume violations", total_volume],
        ["Route mass violations", total_mass],
        ["Route time violations", total_time],
        ["Routing feasible all days", str(routing_feasible)],
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=["Validation Check", "Result"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax.set_title(
        "Rolling-Horizon Validation Dashboard",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    save_fig("07_validation_dashboard.png")


# ============================================================
# Main
# ============================================================

def main():
    print("\nGenerating final charts...")
    print(f"Output folder: {OUTPUT_DIR}")

    chart_methodology_diagram()
    chart_pickups_by_day()
    chart_route_minutes_by_day()
    chart_overflow_trend()
    chart_pickups_by_stream()
    chart_route_minutes_by_truck()
    chart_validation_dashboard()

    print("\nDone. Charts saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()