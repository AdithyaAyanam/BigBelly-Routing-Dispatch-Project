
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def canonical_stream(x):
    """
    Standardize stream labels so historical data and model outputs match.
    """
    s = str(x).strip().lower()

    lookup = {
        "compostables": "Compostables",
        "compost": "Compostables",
        "organics": "Compostables",
        "food waste": "Compostables",

        "waste": "Waste",
        "landfill": "Waste",
        "trash": "Waste",

        "bottles/cans": "Bottles/Cans",
        "bottles & cans": "Bottles/Cans",
        "bottles and cans": "Bottles/Cans",
        "recycling": "Bottles/Cans",
        "recycle": "Bottles/Cans",
        "single stream": "Bottles/Cans",
    }

    return lookup.get(s, str(x).strip() if str(x).strip() else "Unknown")


def clean_serial(x):
    """
    Normalize serial values so 1514001 and 1514001.0 match.
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def find_col(df, candidates):
    """
    Find the first matching column from a list of possible column names.
    Case-insensitive.
    """
    lower_map = {c.lower().strip(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------
# Historical data loader
# ---------------------------------------------------------------------

def load_historical(raw_dir):
    """
    Load historical Daily Collection Activity files from data/raw.

    These Bigbelly exports may contain report metadata rows before the real
    table header. This function auto-detects the real header row by scanning
    for a row containing likely column names such as Serial, Collection, Date,
    Stream, Fullness, etc.
    """
    files = sorted(raw_dir.glob("Daily Collection Activity - CLEAN*.csv"))

    if not files:
        raise FileNotFoundError(
            "No historical collection files found under data/raw matching "
            "'Daily Collection Activity - CLEAN*.csv'"
        )

    def detect_header_row(fp):
        preview = pd.read_csv(fp, header=None, nrows=80, dtype=str)

        keywords = [
            "serial",
            "collection",
            "date",
            "time",
            "stream",
            "fullness",
            "asset",
            "description",
        ]

        best_row = 0
        best_score = -1

        for idx, row in preview.iterrows():
            values = [str(x).strip().lower() for x in row.tolist() if pd.notna(x)]
            joined = " | ".join(values)

            score = sum(1 for kw in keywords if kw in joined)

            if score > best_score:
                best_score = score
                best_row = idx

        return int(best_row)

    frames = []

    for fp in files:
        header_row = detect_header_row(fp)
        print(f"[INFO] {fp.name}: detected header row = {header_row}")

        df = pd.read_csv(fp, header=header_row)
        df = df.dropna(how="all").copy()

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Remove repeated header rows inside the file, if any
        first_col = df.columns[0]
        df = df[df[first_col].astype(str).str.strip() != first_col].copy()

        df["source_file"] = fp.name
        frames.append(df)

    hist = pd.concat(frames, ignore_index=True)

    print("\n[INFO] Historical columns after header detection:")
    for i, c in enumerate(hist.columns):
        print(f"{i}: {c}")

    time_col = find_col(
        hist,
        [
            "Collection_Time",
            "Collection Time",
            "Collection_Date",
            "Collection Date",
            "Collection Date Time",
            "Collection Datetime",
            "Collection Timestamp",
            "Service Date",
            "Service_Date",
            "Service Date Time",
            "Service Datetime",
            "Service Timestamp",
            "Date",
            "Timestamp",
            "Activity Date",
            "Activity_Date",
            "Activity Date Time",
            "Activity Datetime",
            "Activity Timestamp",
            "Created Date",
            "Created_Date",
            "Created At",
            "Created_At",
            "Completed Date",
            "Completed_Date",
            "Completed At",
            "Completed_At",
            "Last Collection",
            "Last_Collection",
            "Last Collection Date",
            "Last_Collection_Date",
            "Collection",
            "Time",
            "Datetime",
        ],
    )

    serial_col = find_col(
        hist,
        [
            "Serial",
            "serial",
            "Bin Serial",
            "Bin_Serial",
            "Asset Serial",
            "Asset_Serial",
            "Asset",
            "Asset ID",
            "Asset_ID",
        ],
    )

    stream_col = find_col(
        hist,
        [
            "Stream_Type",
            "Stream Type",
            "Streams",
            "stream",
            "Stream",
            "Waste Stream",
            "Waste_Stream",
        ],
    )

    fullness_col = find_col(
        hist,
        [
            "Fullness_Pct",
            "Fullness %",
            "Fullness",
            "fullness_pct",
            "Fullness Percent",
            "Fullness_Percent",
            "% Full",
            "Percent Full",
            "Fill Level",
            "Fill_Level",
        ],
    )

    if time_col is None:
        raise KeyError(
            "Could not find a collection timestamp/date column after header detection. "
            "Check the printed historical columns above and add the correct name to time_col candidates."
        )

    if serial_col is None:
        raise KeyError(
            "Could not find a Serial column after header detection. "
            "Check the printed historical columns above and add the correct name to serial_col candidates."
        )

    hist = hist.copy()

    hist["collection_time"] = pd.to_datetime(hist[time_col], errors="coerce")
    hist = hist.dropna(subset=["collection_time"]).copy()

    hist["date"] = hist["collection_time"].dt.date
    hist["Serial"] = hist[serial_col].apply(clean_serial)

    if stream_col is not None:
        hist["stream"] = hist[stream_col].apply(canonical_stream)
    else:
        hist["stream"] = "Unknown"

    if fullness_col is not None:
        hist["fullness_pct"] = safe_numeric(hist[fullness_col])
    else:
        hist["fullness_pct"] = np.nan

    hist = hist[hist["Serial"] != ""].copy()

    print(f"[INFO] using time_col = {time_col}")
    print(f"[INFO] using serial_col = {serial_col}")
    print(f"[INFO] using stream_col = {stream_col}")
    print(f"[INFO] using fullness_col = {fullness_col}")
    print(f"[INFO] loaded historical rows = {len(hist)}")

    return hist

# ---------------------------------------------------------------------
# Rolling output loader
# ---------------------------------------------------------------------

def load_rolling_outputs(rolling_dir):
    """
    Load rolling-horizon outputs.
    """
    metrics_fp = rolling_dir / "rolling_day_metrics.csv"
    schedule_fp = rolling_dir / "rolling_day_schedule.csv"
    route_plan_fp = rolling_dir / "rolling_day_route_plan.csv"
    route_summary_fp = rolling_dir / "rolling_day_route_summary.csv"
    state_fp = rolling_dir / "rolling_day_state_history.csv"

    required = [metrics_fp, schedule_fp, route_plan_fp, route_summary_fp]

    missing = [str(fp) for fp in required if not fp.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing rolling-horizon output files:\n"
            + "\n".join(missing)
            + "\n\nRun scripts/run_5day_rolling_horizon.py first."
        )

    metrics = pd.read_csv(metrics_fp)
    schedule = pd.read_csv(schedule_fp)
    route_plan = pd.read_csv(route_plan_fp)
    route_summary = pd.read_csv(route_summary_fp)

    state = pd.read_csv(state_fp) if state_fp.exists() else pd.DataFrame()

    # Normalize serial and stream columns
    if "Serial" in schedule.columns:
        schedule["Serial"] = schedule["Serial"].apply(clean_serial)

    if "stream" in schedule.columns:
        schedule["stream"] = schedule["stream"].apply(canonical_stream)

    if "stream" in route_plan.columns:
        route_plan["stream"] = route_plan["stream"].apply(canonical_stream)

    if "stream" in route_summary.columns:
        route_summary["stream"] = route_summary["stream"].apply(canonical_stream)

    return metrics, schedule, route_plan, route_summary, state


# ---------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare rolling-horizon model outputs against historical collection activity."
    )

    parser.add_argument(
        "--actual-start-date",
        type=str,
        default=None,
        help=(
            "Historical start date in YYYY-MM-DD. "
            "If omitted, the script uses the first available historical date."
        ),
    )

    parser.add_argument(
        "--num-days",
        type=int,
        default=None,
        help=(
            "Number of historical operating days to compare. "
            "If omitted, uses the number of rolling days in rolling_day_metrics.csv."
        ),
    )

    parser.add_argument(
        "--service-min-per-bin",
        type=float,
        default=4.0,
        help="Historical route-minutes proxy service time per pickup.",
    )

    parser.add_argument(
        "--travel-min-between-stops",
        type=float,
        default=8.0,
        help="Historical route-minutes proxy travel time per pickup.",
    )

    parser.add_argument(
        "--rolling-dir",
        type=str,
        default="data/processed/rolling_horizon_5day",
        help="Directory containing rolling-horizon output CSVs.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rolling_vs_historical_comparison",
        help="Directory where comparison outputs should be saved.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    rolling_dir = root / args.rolling_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    hist = load_historical(raw_dir)
    rolling_metrics, rolling_schedule, rolling_routes, rolling_route_summary, rolling_state = (
        load_rolling_outputs(rolling_dir)
    )

    if "rolling_day" not in rolling_metrics.columns:
        raise KeyError("rolling_day_metrics.csv missing rolling_day column.")

    rolling_days = sorted(rolling_metrics["rolling_day"].dropna().astype(int).unique())
    num_rolling_days = len(rolling_days)

    if num_rolling_days == 0:
        raise ValueError("No rolling days found in rolling_day_metrics.csv.")

    num_days = args.num_days if args.num_days is not None else num_rolling_days

    # ------------------------------------------------------------------
    # Choose historical comparison dates
    # ------------------------------------------------------------------

    available_dates = sorted(hist["date"].dropna().unique())

    if not available_dates:
        raise ValueError("No valid historical dates found.")

    if args.actual_start_date:
        start_date = pd.to_datetime(args.actual_start_date).date()
        comparison_dates = [d for d in available_dates if d >= start_date][:num_days]
    else:
        comparison_dates = available_dates[:num_days]

    if len(comparison_dates) < num_days:
        raise ValueError(
            f"Only found {len(comparison_dates)} historical dates, but need {num_days}."
        )

    hist_cmp = hist[hist["date"].isin(comparison_dates)].copy()

    date_map = pd.DataFrame(
        {
            "rolling_day": list(range(1, len(comparison_dates) + 1)),
            "historical_date": comparison_dates,
        }
    )

    # ------------------------------------------------------------------
    # Rolling daily summary
    # ------------------------------------------------------------------

    rolling_daily_cols = [
        "rolling_day",
        "horizon_days",
        "num_service_days",
        "target_avg_interstop_min",
        "pickups_today",
        "pickup_gal_today",
        "pickup_lb_today",
        "routes_today",
        "route_minutes_today",
        "overtime_today",
        "extra_dumps_today",
        "capacity_time_violations",
        "overflow_bins_start_of_day",
        "routing_feasible",
        "dropped_stops",
        "missing_routed_serials",
        "extra_routed_serials",
        "planner_status",
        "planner_objective",
    ]

    rolling_daily_cols = [c for c in rolling_daily_cols if c in rolling_metrics.columns]

    rolling_daily = rolling_metrics[rolling_daily_cols].copy()
    rolling_daily["rolling_day"] = rolling_daily["rolling_day"].astype(int)
    rolling_daily = rolling_daily.merge(date_map, on="rolling_day", how="left")

    # ------------------------------------------------------------------
    # Historical daily summary
    # ------------------------------------------------------------------

    historical_daily = (
        hist_cmp.groupby("date")
        .agg(
            historical_pickups=("Serial", "count"),
            historical_unique_bins=("Serial", "nunique"),
            historical_avg_fullness_pct=("fullness_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"date": "historical_date"})
    )

    approx_min_per_pickup = args.service_min_per_bin + args.travel_min_between_stops

    historical_daily["historical_route_minutes_proxy"] = (
        historical_daily["historical_pickups"] * approx_min_per_pickup
    )

    comparison_daily = rolling_daily.merge(
        historical_daily,
        on="historical_date",
        how="left",
    )

    comparison_daily["pickup_count_difference_model_minus_actual"] = (
        comparison_daily["pickups_today"] - comparison_daily["historical_pickups"]
    )

    comparison_daily["pickup_count_ratio_model_to_actual"] = (
        comparison_daily["pickups_today"]
        / comparison_daily["historical_pickups"].replace(0, np.nan)
    )

    comparison_daily["route_minutes_difference_model_minus_historical_proxy"] = (
        comparison_daily["route_minutes_today"]
        - comparison_daily["historical_route_minutes_proxy"]
    )

    comparison_daily["route_minutes_ratio_model_to_historical_proxy"] = (
        comparison_daily["route_minutes_today"]
        / comparison_daily["historical_route_minutes_proxy"].replace(0, np.nan)
    )

    # ------------------------------------------------------------------
    # Stream-level comparison
    # ------------------------------------------------------------------

    required_schedule_cols = ["rolling_day", "stream", "Serial"]
    missing_sched = [c for c in required_schedule_cols if c not in rolling_schedule.columns]

    if missing_sched:
        raise KeyError(f"rolling_day_schedule.csv missing columns: {missing_sched}")

    rolling_stream = (
        rolling_schedule.groupby(["rolling_day", "stream"])
        .agg(
            model_pickups=("Serial", "count"),
            model_unique_bins=("Serial", "nunique"),
            model_pickup_gal=("pickup_gal", "sum") if "pickup_gal" in rolling_schedule.columns else ("Serial", "count"),
            model_pickup_lb=("pickup_lb", "sum") if "pickup_lb" in rolling_schedule.columns else ("Serial", "count"),
        )
        .reset_index()
    )

    rolling_stream["rolling_day"] = rolling_stream["rolling_day"].astype(int)
    rolling_stream = rolling_stream.merge(date_map, on="rolling_day", how="left")

    hist_stream = (
        hist_cmp.groupby(["date", "stream"])
        .agg(
            historical_pickups=("Serial", "count"),
            historical_unique_bins=("Serial", "nunique"),
            historical_avg_fullness_pct=("fullness_pct", "mean"),
        )
        .reset_index()
        .rename(columns={"date": "historical_date"})
    )

    comparison_stream = rolling_stream.merge(
        hist_stream,
        on=["historical_date", "stream"],
        how="outer",
    )

    fill_zero_cols = [
        "model_pickups",
        "model_unique_bins",
        "model_pickup_gal",
        "model_pickup_lb",
        "historical_pickups",
        "historical_unique_bins",
    ]

    for col in fill_zero_cols:
        if col in comparison_stream.columns:
            comparison_stream[col] = comparison_stream[col].fillna(0)

    comparison_stream["pickup_difference_model_minus_actual"] = (
        comparison_stream["model_pickups"] - comparison_stream["historical_pickups"]
    )

    comparison_stream["pickup_ratio_model_to_actual"] = (
        comparison_stream["model_pickups"]
        / comparison_stream["historical_pickups"].replace(0, np.nan)
    )

    # ------------------------------------------------------------------
    # Route-level model summary
    # ------------------------------------------------------------------

    route_summary = (
        rolling_routes.groupby(["rolling_day", "stream"])
        .agg(
            model_routes=("truck", "count") if "truck" in rolling_routes.columns else ("stream", "count"),
            model_route_minutes=("route_minutes", "sum") if "route_minutes" in rolling_routes.columns else ("stream", "count"),
            model_route_gal=("route_gal", "sum") if "route_gal" in rolling_routes.columns else ("stream", "count"),
            model_route_lb=("route_lb", "sum") if "route_lb" in rolling_routes.columns else ("stream", "count"),
            model_stops=("num_stops", "sum") if "num_stops" in rolling_routes.columns else ("stream", "count"),
        )
        .reset_index()
    )

    route_summary["rolling_day"] = route_summary["rolling_day"].astype(int)
    route_summary = route_summary.merge(date_map, on="rolling_day", how="left")

    # ------------------------------------------------------------------
    # Overflow state trend
    # ------------------------------------------------------------------

    if not rolling_state.empty and {"rolling_day", "true_overflow_start"}.issubset(rolling_state.columns):
        overflow_trend = (
            rolling_state.groupby("rolling_day")
            .agg(
                overflow_bins_start=("true_overflow_start", "sum"),
                bins_tracked=("Serial", "nunique"),
            )
            .reset_index()
        )
    elif "overflow_bins_start_of_day" in rolling_metrics.columns:
        overflow_trend = rolling_metrics[
            ["rolling_day", "overflow_bins_start_of_day"]
        ].copy()
    else:
        overflow_trend = pd.DataFrame()

    # ------------------------------------------------------------------
    # Overall comparison summary
    # ------------------------------------------------------------------

    total_model_pickups = float(comparison_daily["pickups_today"].sum())
    total_hist_pickups = float(comparison_daily["historical_pickups"].sum())
    total_model_minutes = float(comparison_daily["route_minutes_today"].sum())
    total_hist_proxy_minutes = float(comparison_daily["historical_route_minutes_proxy"].sum())

    total_model_gal = (
        float(comparison_daily["pickup_gal_today"].sum())
        if "pickup_gal_today" in comparison_daily.columns
        else np.nan
    )

    total_model_lb = (
        float(comparison_daily["pickup_lb_today"].sum())
        if "pickup_lb_today" in comparison_daily.columns
        else np.nan
    )

    total_capacity_violations = (
        int(comparison_daily["capacity_time_violations"].fillna(0).sum())
        if "capacity_time_violations" in comparison_daily.columns
        else np.nan
    )

    total_dropped = (
        int(comparison_daily["dropped_stops"].fillna(0).sum())
        if "dropped_stops" in comparison_daily.columns
        else np.nan
    )

    total_missing = (
        int(comparison_daily["missing_routed_serials"].fillna(0).sum())
        if "missing_routed_serials" in comparison_daily.columns
        else np.nan
    )

    total_extra = (
        int(comparison_daily["extra_routed_serials"].fillna(0).sum())
        if "extra_routed_serials" in comparison_daily.columns
        else np.nan
    )

    overall = pd.DataFrame(
        [
            {
                "rolling_days_compared": num_rolling_days,
                "historical_dates_used": ", ".join(str(d) for d in comparison_dates),
                "planning_horizon_days": int(rolling_metrics["horizon_days"].iloc[0]) if "horizon_days" in rolling_metrics.columns else 7,
                "service_days_per_solve": int(rolling_metrics["num_service_days"].iloc[0]) if "num_service_days" in rolling_metrics.columns else 5,
                "target_avg_interstop_min": float(rolling_metrics["target_avg_interstop_min"].iloc[0]) if "target_avg_interstop_min" in rolling_metrics.columns else 8.0,
                "total_model_pickups": total_model_pickups,
                "total_historical_pickups": total_hist_pickups,
                "model_minus_historical_pickups": total_model_pickups - total_hist_pickups,
                "model_to_historical_pickup_ratio": total_model_pickups / total_hist_pickups if total_hist_pickups else np.nan,
                "total_model_pickup_gal": total_model_gal,
                "total_model_pickup_lb": total_model_lb,
                "total_model_route_minutes": total_model_minutes,
                "total_historical_route_minutes_proxy": total_hist_proxy_minutes,
                "model_minus_historical_route_minutes_proxy": total_model_minutes - total_hist_proxy_minutes,
                "model_to_historical_route_minutes_ratio": total_model_minutes / total_hist_proxy_minutes if total_hist_proxy_minutes else np.nan,
                "total_capacity_time_violations": total_capacity_violations,
                "total_dropped_stops": total_dropped,
                "total_missing_routed_serials": total_missing,
                "total_extra_routed_serials": total_extra,
            }
        ]
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    daily_fp = output_dir / "daily_model_vs_historical.csv"
    stream_fp = output_dir / "stream_model_vs_historical.csv"
    route_fp = output_dir / "model_route_summary.csv"
    overflow_fp = output_dir / "rolling_overflow_trend.csv"
    overall_fp = output_dir / "overall_comparison_summary.csv"
    report_fp = output_dir / "comparison_report.md"

    comparison_daily.to_csv(daily_fp, index=False)
    comparison_stream.to_csv(stream_fp, index=False)
    route_summary.to_csv(route_fp, index=False)
    overall.to_csv(overall_fp, index=False)

    if not overflow_trend.empty:
        overflow_trend.to_csv(overflow_fp, index=False)

    report = f"""# Rolling Horizon vs Historical Collection Comparison

## Scope

This comparison evaluates the rolling-horizon model output against historical collection activity.

- Rolling days compared: {num_rolling_days}
- Historical dates used: {", ".join(str(d) for d in comparison_dates)}
- Planning horizon per rolling solve: 7 calendar days
- Service days per rolling solve: 5 weekday service days
- Travel-time calibration target: 8-minute average non-depot inter-stop travel time
- Historical route minutes are a proxy: historical pickups × {approx_min_per_pickup:.1f} minutes

## Overall Results

- Total model pickups: {total_model_pickups:.0f}
- Total historical pickups: {total_hist_pickups:.0f}
- Difference, model minus historical: {total_model_pickups - total_hist_pickups:.0f}
- Model / historical pickup ratio: {(total_model_pickups / total_hist_pickups if total_hist_pickups else np.nan):.2f}
- Total model route minutes: {total_model_minutes:.1f}
- Total historical route-minutes proxy: {total_hist_proxy_minutes:.1f}
- Difference, model minus historical route-minutes proxy: {total_model_minutes - total_hist_proxy_minutes:.1f}
- Total model pickup gallons: {total_model_gal:.2f}
- Total model pickup pounds: {total_model_lb:.2f}
- Capacity/time violations: {total_capacity_violations}
- Dropped stops: {total_dropped}
- Missing routed serials: {total_missing}
- Extra routed serials: {total_extra}

## Interpretation Notes

The rolling-horizon model is not expected to exactly match historical operations. The model optimizes collection based on projected fill levels, overflow penalties, compost hygiene, truck capacity, and routing feasibility. Historical operations may reflect additional constraints that are not observed in the dataset, such as driver judgment, real-time requests, campus traffic, access restrictions, staff availability, special events, or manual prioritization.

The historical route-minutes value is only a proxy because actual driver route times are not directly observed. It uses the same planning-stage assumption of 4 minutes service time plus 8 minutes travel time per pickup.

The most important validation question is whether the model produces a feasible and operationally reasonable dispatch pattern: all scheduled bins are routed, no stops are dropped, no truck exceeds volume or mass capacity, and route times remain within the 480-minute truck-day limit.
"""

    report_fp.write_text(report, encoding="utf-8")

    # ------------------------------------------------------------------
    # Print outputs
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("ROLLING HORIZON VS HISTORICAL COMPARISON")
    print("=" * 90)

    print("\nHistorical dates used:")
    for d in comparison_dates:
        print("-", d)

    print("\n[1] Overall comparison summary")
    print(overall.T.to_string())

    print("\n[2] Daily comparison")
    print(comparison_daily.to_string(index=False))

    print("\n[3] Stream comparison")
    print(comparison_stream.to_string(index=False))

    print("\n[4] Model route summary")
    print(route_summary.to_string(index=False))

    if not overflow_trend.empty:
        print("\n[5] Overflow trend")
        print(overflow_trend.to_string(index=False))

    print("\nSaved outputs:")
    print(daily_fp)
    print(stream_fp)
    print(route_fp)
    print(overall_fp)

    if not overflow_trend.empty:
        print(overflow_fp)

    print(report_fp)
    print("=" * 90)


if __name__ == "__main__":
    main()
