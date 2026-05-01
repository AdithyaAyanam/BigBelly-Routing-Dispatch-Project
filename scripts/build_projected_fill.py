from __future__ import annotations

"""
build_projected_fill.py
-------------------------------------------------------------
Purpose
-------
Build a 7-day projected-fill input table for the Bigbelly scheduling model.

This script prepares the forecasting / preprocessing inputs that feed the
7-day scheduling MILP. It does NOT assign trucks or build routes. Instead, it
estimates how quickly each bin fills, how full it is today, and what the
service-trigger picture looks like over the next 7 days under a no-service
projection.

This version is aligned with the current modeling direction:
1. No geographic zoning.
2. Stream-specific service policy:
   - Waste: 60% threshold
   - Compostables: 60% threshold
   - Bottles/Cans: no fill threshold, rely on the 7-day rule
3. Output fields support an inventory-based downstream model:
   - current_fill_pct_est
   - daily_fill_growth_pct
   - threshold_pct
   - deadline_threshold / deadline_interval / service_deadline
   - clearer aliases:
       initial_deadline_threshold_no_service
       initial_deadline_interval_no_service
       initial_service_deadline_no_service
4. Includes projected gallons and pounds by day so the scheduler can enforce
   realistic mass and volume constraints.
5. Adds per-bin handling-weight QA flags.
6. Carries routing-relevant access fields through the file if available:
   - Access_Lat
   - Access_Lng
   - Stop_ID
   - service_walk_min

Inputs
------
Expected upstream files from run_pipeline.py:
- data/interim/assets.parquet
- data/interim/collections_merged.parquet

Outputs
-------
- data/processed/bin_7day_projection_inputs.csv
- data/processed/bin_7day_projection_inputs.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------
# Default stream density assumptions (lb/gal)
# -------------------------------------------------------------
# These are intentionally parameterized in main() so they can be
# changed from the command line without editing code.
DEFAULT_STREAM_DENSITY_LB_PER_GAL = {
    "Waste": 1.0,
    "Compostables": 1.2,   # lighter food + paper mix placeholder
    "Bottles/Cans": 0.3,
}


# -------------------------------------------------------------
# Helper: project root
# -------------------------------------------------------------

def compute_dynamic_travel_proxy_min(processed_dir: Path, fallback: float) -> float:
    travel_fp = processed_dir / "travel_matrix_long.csv"

    if not travel_fp.exists():
        print(f"[WARN] travel_matrix_long.csv not found; using fallback default_travel_min={fallback}")
        return float(fallback)

    tm = pd.read_csv(travel_fp)
    print(f"[INFO] travel_matrix_long columns = {tm.columns.tolist()}")

    time_col = None
    for candidate in [
        "travel_time_min",
        "travel_minutes",
        "travel_min",
        "time_min",
        "duration_min",
        "minutes",
        "route_minutes",
    ]:
        if candidate in tm.columns:
            time_col = candidate
            break

    if time_col is None:
        print(f"[WARN] No travel-time column found in travel_matrix_long.csv; using fallback default_travel_min={fallback}")
        return float(fallback)

    print(f"[INFO] using travel-time column: {time_col}")

    filtered = tm.copy()

    # Exclude depot-related rows.
    if "from_label" in filtered.columns:
        filtered = filtered[~filtered["from_label"].astype(str).str.lower().eq("depot")]
    if "to_label" in filtered.columns:
        filtered = filtered[~filtered["to_label"].astype(str).str.lower().eq("depot")]

    # Exclude self-pairs.
    if "from_node" in filtered.columns and "to_node" in filtered.columns:
        filtered = filtered[filtered["from_node"] != filtered["to_node"]]

    filtered[time_col] = pd.to_numeric(filtered[time_col], errors="coerce")
    filtered = filtered[(filtered[time_col] > 0) & filtered[time_col].notna()].copy()

    if filtered.empty:
        print(f"[WARN] No valid non-depot travel times found; using fallback default_travel_min={fallback}")
        return float(fallback)

    # Use a local-travel proxy: the 25th percentile of non-depot, non-self trips.
    # This avoids the old over-conservative all-pairs median while also avoiding
    # unrealistically tiny nearest-neighbor values.
    dynamic_value = float(filtered[time_col].quantile(0.25))

    print(
    f"[INFO] dynamic avg_travel_proxy_min from OSMnx local-travel 25th percentile = {dynamic_value:.2f}"
)

    return dynamic_value

def repo_root() -> Path:
    """
    Return the project root directory.

    Assumes this file is stored in:
        project_root/scripts/build_projected_fill.py
    """
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------------------
# Helper: ensure expected folders exist
# -------------------------------------------------------------
def ensure_dirs(root: Path) -> dict[str, Path]:
    """
    Ensure the processed and interim folders exist.
    """
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    interim_dir = data_dir / "interim"

    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    return {"root": root, "processed": processed_dir, "interim": interim_dir}


# -------------------------------------------------------------
# Helper: standardize stream names
# -------------------------------------------------------------
def canonical_stream(x: object) -> str:
    """
    Standardize stream labels into a small consistent set.

    Examples
    --------
    'compost'       -> 'Compostables'
    'landfill'      -> 'Waste'
    'recycling'     -> 'Bottles/Cans'
    'single stream' -> 'Bottles/Cans'
    """
    s = str(x).strip().lower()
    lookup = {
        "compostables": "Compostables",
        "compost": "Compostables",
        "waste": "Waste",
        "landfill": "Waste",
        "bottles/cans": "Bottles/Cans",
        "bottles & cans": "Bottles/Cans",
        "recycling": "Bottles/Cans",
        "single stream": "Bottles/Cans",
    }
    return lookup.get(s, str(x).strip() if str(x).strip() else "Unknown")


# -------------------------------------------------------------
# Helper: current service-threshold policy by stream
# -------------------------------------------------------------
def policy_threshold_for_stream(stream: str, fallback_threshold_pct: float = 60.0) -> float:
    """
    Return the current Zero Waste threshold policy by stream.

    Policy
    ------
    Waste and Compostables:
        use a 60% fullness trigger
    Bottles/Cans:
        no fill trigger; rely on the 7-day rule

    Returns
    -------
    float
        threshold percentage, or NaN if the stream has no fill trigger
    """
    stream = canonical_stream(stream)

    if stream in {"Waste", "Compostables"}:
        return float(fallback_threshold_pct)

    if stream == "Bottles/Cans":
        return np.nan

    return float(fallback_threshold_pct)


# -------------------------------------------------------------
# Clean / enrich the assets table
# -------------------------------------------------------------
def clean_assets(
    df_assets: pd.DataFrame,
    default_threshold_pct: float,
    enforce_stream_policy: bool,
    stream_density_lb_per_gal: dict[str, float],
) -> pd.DataFrame:
    """
    Clean and standardize the assets/bin master table.

    Main tasks
    ----------
    - verify Serial exists
    - standardize stream labels
    - retain/parse any raw threshold fields from the asset export
    - optionally override those with the currently approved stream policy
    - attach density assumptions
    - coerce coordinate / access fields to numeric where present
    - carry stop-level routing fields when available
    """
    df = df_assets.copy()

    if "Serial" not in df.columns:
        raise KeyError("assets.parquet missing 'Serial'")

    df["Serial"] = df["Serial"].astype(str).str.strip()

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip()

    if "Streams" in df.columns:
        df["stream"] = df["Streams"].apply(canonical_stream)
    else:
        df["stream"] = "Unknown"

    # Try to recover any raw threshold field from the assets file
    extracted = np.nan
    if "Fullness_Threshold" in df.columns:
        extracted = pd.to_numeric(
            df["Fullness_Threshold"].astype(str).str.extract(r"(\d+(\.\d+)?)")[0],
            errors="coerce",
        )
    elif "Fullness_Threshold_Pct" in df.columns:
        extracted = pd.to_numeric(df["Fullness_Threshold_Pct"], errors="coerce")

    if isinstance(extracted, pd.Series):
        df["threshold_pct_raw"] = extracted
    else:
        df["threshold_pct_raw"] = np.nan

    # Either enforce the approved stream-specific policy, or keep raw
    # thresholds when available and only backfill missing values.
    if enforce_stream_policy:
        df["threshold_pct"] = df["stream"].apply(
            lambda s: policy_threshold_for_stream(s, fallback_threshold_pct=default_threshold_pct)
        )
    else:
        df["threshold_pct"] = df["threshold_pct_raw"]
        missing = df["threshold_pct"].isna()
        df.loc[missing, "threshold_pct"] = df.loc[missing, "stream"].apply(
            lambda s: policy_threshold_for_stream(s, fallback_threshold_pct=default_threshold_pct)
        )

    # Numeric fields that may be present
    for col in ["Lat", "Lng", "Access_Lat", "Access_Lng", "service_walk_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stop fields for routing
    if "Stop_ID" in df.columns:
        df["Stop_ID"] = df["Stop_ID"].astype(str).str.strip()
    else:
        df["Stop_ID"] = df["Serial"]

    if "service_walk_min" not in df.columns:
        df["service_walk_min"] = 0.0

    # If truck-access coordinates are missing, fall back to raw Lat/Lng
    if "Access_Lat" not in df.columns:
        df["Access_Lat"] = np.nan
    if "Access_Lng" not in df.columns:
        df["Access_Lng"] = np.nan

    if "Lat" in df.columns:
        df["Access_Lat"] = df["Access_Lat"].fillna(df["Lat"])
    if "Lng" in df.columns:
        df["Access_Lng"] = df["Access_Lng"].fillna(df["Lng"])

    # Attach density for later mass projections
    df["density_lb_per_gal"] = df["stream"].map(stream_density_lb_per_gal).fillna(1.0)

    return df


# -------------------------------------------------------------
# Estimate historical fill growth rates
# -------------------------------------------------------------
def compute_growth_rates(df_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Estimate fill growth rates from historical collections.

    Logic
    -----
    For each bin:
    - sort collection events in time
    - compute the gap in days between consecutive collections
    - estimate the average growth over that gap as:
        fullness_at_collection / gap_days

    We then summarize growth at three levels:
    1. bin-specific median
    2. stream-level median
    3. overall median fallback

    An IQR-based upper filter removes extreme spikes caused by bad
    sensor readings or unusual one-off collection gaps.
    """
    df = df_merged.copy()

    needed = ["Serial", "Collection_Time", "Fullness_Pct"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"collections_merged.parquet missing '{col}'")

    if "Stream_Type" not in df.columns:
        df["Stream_Type"] = "Unknown"

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")
    df["Stream_Type"] = df["Stream_Type"].apply(canonical_stream)

    # Keep only rows with valid timestamps and valid fullness values
    hist = df.dropna(subset=["Collection_Time", "Fullness_Pct"]).copy()
    hist = hist.sort_values(["Serial", "Collection_Time"]).reset_index(drop=True)

    # Previous collection time for the same bin
    hist["prev_collection_time"] = hist.groupby("Serial")["Collection_Time"].shift(1)

    # Gap in days between consecutive collections
    hist["gap_days"] = (
        (hist["Collection_Time"] - hist["prev_collection_time"]).dt.total_seconds() / 86400.0
    )

    # Ignore missing or tiny gaps
    hist = hist[(hist["gap_days"].notna()) & (hist["gap_days"] > 0.5)].copy()

    # Estimate growth as fullness observed divided by gap length
    hist["growth_pct_per_day"] = hist["Fullness_Pct"] / hist["gap_days"]

    # Remove impossible / bad values
    hist.loc[
        ~np.isfinite(hist["growth_pct_per_day"]) | (hist["growth_pct_per_day"] < 0),
        "growth_pct_per_day",
    ] = np.nan

    hist = hist[hist["growth_pct_per_day"].notna()].copy()

    # IQR filter to trim spurious very high growth estimates
    if not hist.empty:
        q1 = hist["growth_pct_per_day"].quantile(0.25)
        q3 = hist["growth_pct_per_day"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3.0 * iqr
        hist = hist[hist["growth_pct_per_day"] <= upper].copy()

    # Multi-level fallback structure
    bin_growth = hist.groupby("Serial")["growth_pct_per_day"].median()
    stream_growth = hist.groupby("Stream_Type")["growth_pct_per_day"].median()
    overall_growth = float(hist["growth_pct_per_day"].median()) if not hist.empty else 12.0

    return hist, bin_growth, stream_growth, overall_growth


# -------------------------------------------------------------
# Find the most recent service event for each bin
# -------------------------------------------------------------
def compute_last_service(df_merged: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute the most recent known service date for each bin and the
    days since last service relative to the anchor date.
    """
    df = df_merged.copy()
    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")
    df = df.dropna(subset=["Collection_Time"]).copy()
    df = df.sort_values(["Serial", "Collection_Time"])

    last = df.groupby("Serial").tail(1).copy()
    last["last_service_date"] = last["Collection_Time"].dt.floor("D")
    last["days_since_last_service"] = (
        anchor_date.floor("D") - last["last_service_date"]
    ).dt.days.clip(lower=0)

    keep_cols = ["Serial", "Collection_Time", "last_service_date", "days_since_last_service"]
    if "Description" in last.columns:
        keep_cols.append("Description")
    if "Stream_Type" in last.columns:
        keep_cols.append("Stream_Type")

    return last[keep_cols].rename(columns={"Collection_Time": "last_collection_time"})


# -------------------------------------------------------------
# Threshold-based deadline
# -------------------------------------------------------------
def threshold_deadline(row: pd.Series, horizon_days: int) -> float:
    """
    Return the first day in the horizon when projected fill reaches
    or exceeds the bin's threshold, under the no-service projection.

    If the stream has no fill threshold, return NaN.
    """
    threshold = row.get("threshold_pct", np.nan)

    if pd.isna(threshold):
        return np.nan

    for d in range(horizon_days):
        if row[f"fill_day_{d}"] >= threshold:
            return float(d)

    return np.nan


# -------------------------------------------------------------
# 7-day rule deadline
# -------------------------------------------------------------
def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    """
    Return the first day in the horizon when the 7-day rule forces service.

    Examples
    --------
    days_since_last_service = 6 -> deadline day 1
    days_since_last_service = 7 -> deadline day 0
    days_since_last_service = 1 -> deadline day 6
    """
    remaining = 7 - float(days_since_last_service)

    if remaining <= 0:
        return 0.0

    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))

    return np.nan


# -------------------------------------------------------------
# Build the final projected-fill table
# -------------------------------------------------------------
def build_projection_table(
    assets: pd.DataFrame,
    last_service: pd.DataFrame,
    bin_growth: pd.Series,
    stream_growth: pd.Series,
    overall_growth: float,
    horizon_days: int,
    default_bin_capacity_gal: float,
    default_service_min: float,
    default_travel_min: float,
    safe_bin_content_lb: float,
    hard_bin_content_lb: float,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Construct the final 7-day modeling table.

    For each bin, this function:
    - merges asset data and recent service history
    - assigns a daily fill growth estimate
    - estimates current fill at day 0
    - projects fill over the next 7 days under no service
    - computes threshold and interval deadlines
    - computes projected gallons and projected pounds by day
    - attaches simple operational parameters for downstream use
    """
    base = assets.copy()

    # Keep active/in-service bins only
    if "Status" in base.columns:
        base = base[
            base["Status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["in service", "in_service", "active", ""])
        ].copy()

    keep_cols = [
        c
        for c in [
            "Serial",
            "Description",
            "stream",
            "threshold_pct",
            "threshold_pct_raw",
            "density_lb_per_gal",
            "Lat",
            "Lng",
            "Access_Lat",
            "Access_Lng",
            "Stop_ID",
            "service_walk_min",
        ]
        if c in base.columns
    ]
    base = base[keep_cols].drop_duplicates(subset=["Serial"]).copy()

    # Add recent service data
    base = base.merge(
        last_service[["Serial", "last_service_date", "days_since_last_service"]],
        on="Serial",
        how="left",
    )

    base["stream"] = base["stream"].fillna("Unknown").apply(canonical_stream)

    # Preserve the raw age before capping/filtering and make sensor quality explicit.
    # Do not silently drop stale/no-history bins here; downstream solver filters
    # on is_active_bin so QA can still review what was excluded.
    base["days_since_last_service_raw"] = base["days_since_last_service"]
    base["sensor_age_days"] = base["days_since_last_service_raw"]

    base["sensor_status"] = np.select(
        [
            base["sensor_age_days"].isna(),
            base["sensor_age_days"] > 21,
        ],
        [
            "no_history",
            "stale_or_off",
        ],
        default="active",
    )

    # Cap active age so extremely stale records do not create explosive estimates.
    # Stale/no-history bins receive age 0 and are marked inactive below.
    base["days_since_last_service"] = np.where(
        base["sensor_status"].eq("active"),
        base["days_since_last_service_raw"].clip(lower=0, upper=10),
        0.0,
    )

    # Growth estimate fallback logic:
    # 1) bin-level historical median
    # 2) stream-level historical median
    # 3) overall historical median
    base["daily_fill_growth_pct"] = base["Serial"].map(bin_growth)
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(
        base["stream"].map(stream_growth)
    )
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(overall_growth)

    base["is_active_bin"] = (
        base["sensor_status"].eq("active")
        & base["daily_fill_growth_pct"].fillna(0).gt(0)
    )

    # Estimate current fill at day 0
    base["current_fill_pct_est"] = (
        base["daily_fill_growth_pct"] * base["days_since_last_service"]
    ).clip(lower=0, upper=150)

    # Attach simple default operating parameters
    base["bin_capacity_gal"] = float(default_bin_capacity_gal)
    base["avg_service_min"] = float(default_service_min)

    # Keep original name for compatibility with current scheduler,
    # and also provide a clearer alias.
    dynamic_travel_min = compute_dynamic_travel_proxy_min(processed_dir, default_travel_min)

    base["avg_travel_proxy_min"] = dynamic_travel_min
    base["avg_travel_proxy_min_phase1"] = dynamic_travel_min
    # Build projected fields day by day
    for d in range(horizon_days):
        fill_col = f"fill_day_{d}"
        gal_col = f"pickup_gal_day_{d}"
        lb_col = f"pickup_lb_day_{d}"
        safe_flag_col = f"bin_over_safe_lb_day_{d}"
        hard_flag_col = f"bin_over_hard_lb_day_{d}"

        base[fill_col] = (
            base["current_fill_pct_est"] + base["daily_fill_growth_pct"] * d
        ).clip(upper=150)

        base[gal_col] = base["bin_capacity_gal"] * base[fill_col] / 100.0
        base[lb_col] = base[gal_col] * base["density_lb_per_gal"]

        # Practical handling QA flags
        base[safe_flag_col] = base[lb_col] > float(safe_bin_content_lb)
        base[hard_flag_col] = base[lb_col] > float(hard_bin_content_lb)

    # Compute threshold-based and 7-day-rule-based deadlines
    base["deadline_threshold"] = base.apply(
        lambda row: threshold_deadline(row, horizon_days),
        axis=1,
    )

    base["deadline_interval"] = base["days_since_last_service"].apply(
        lambda x: interval_deadline(x, horizon_days)
    )

    # Effective first service deadline = whichever comes first
    base["service_deadline"] = base[["deadline_threshold", "deadline_interval"]].min(
        axis=1,
        skipna=True,
    )

    # If neither rule binds in the horizon, leave deadline missing
    base.loc[
        base[["deadline_threshold", "deadline_interval"]].isna().all(axis=1),
        "service_deadline",
    ] = np.nan

    base["must_service_within_horizon"] = base["service_deadline"].notna()

    # Clearer aliases so it is obvious these are no-service urgency estimates
    base["initial_deadline_threshold_no_service"] = base["deadline_threshold"]
    base["initial_deadline_interval_no_service"] = base["deadline_interval"]
    base["initial_service_deadline_no_service"] = base["service_deadline"]

    # Final output column order
    cols = (
        [
            "Serial",
            "Description",
            "stream",
            "threshold_pct",
            "threshold_pct_raw",
            "density_lb_per_gal",
            "days_since_last_service",
            "days_since_last_service_raw",
            "sensor_age_days",
            "sensor_status",
            "is_active_bin",
            "daily_fill_growth_pct",
            "current_fill_pct_est",
            "bin_capacity_gal",
            "avg_service_min",
            "avg_travel_proxy_min",
            "avg_travel_proxy_min_phase1",
            "Stop_ID",
            "service_walk_min",
            "Lat",
            "Lng",
            "Access_Lat",
            "Access_Lng",
        ]
        + [f"fill_day_{d}" for d in range(horizon_days)]
        + [f"pickup_gal_day_{d}" for d in range(horizon_days)]
        + [f"pickup_lb_day_{d}" for d in range(horizon_days)]
        + [f"bin_over_safe_lb_day_{d}" for d in range(horizon_days)]
        + [f"bin_over_hard_lb_day_{d}" for d in range(horizon_days)]
        + [
            "deadline_threshold",
            "deadline_interval",
            "service_deadline",
            "initial_deadline_threshold_no_service",
            "initial_deadline_interval_no_service",
            "initial_service_deadline_no_service",
            "must_service_within_horizon",
        ]
    )
    cols = [c for c in cols if c in base.columns]

    # Sort by urgency
    return (
        base[cols]
        .sort_values(["service_deadline", "Serial"], na_position="last")
        .reset_index(drop=True)
    )


# -------------------------------------------------------------
# Main script
# -------------------------------------------------------------
def main() -> None:
    """
    Workflow
    --------
    1. Parse runtime arguments.
    2. Load interim assets and merged collection history.
    3. Clean / standardize assets.
    4. Compute growth rates.
    5. Compute last service dates.
    6. Build the projected-fill table.
    7. Save CSV and Parquet outputs.
    """
    parser = argparse.ArgumentParser(
        description="Build 7-day projected-fill optimization inputs from interim parquet files."
    )
    parser.add_argument("--anchor-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--default-threshold-pct", type=float, default=60.0)
    parser.add_argument("--default-bin-capacity-gal", type=float, default=150.0)
    parser.add_argument("--default-service-min", type=float, default=4.0)
    parser.add_argument("--default-travel-min", type=float, default=3.1)

    # Stream densities
    parser.add_argument("--waste-density", type=float, default=DEFAULT_STREAM_DENSITY_LB_PER_GAL["Waste"])
    parser.add_argument("--compost-density", type=float, default=DEFAULT_STREAM_DENSITY_LB_PER_GAL["Compostables"])
    parser.add_argument("--recycling-density", type=float, default=DEFAULT_STREAM_DENSITY_LB_PER_GAL["Bottles/Cans"])

    # Practical handling thresholds
    parser.add_argument("--safe-bin-content-lb", type=float, default=250.0)
    parser.add_argument("--hard-bin-content-lb", type=float, default=400.0)

    parser.add_argument(
        "--use-stream-threshold-policy",
        action="store_true",
        default=True,
        help="Apply Waste/Compost=60%% and Bottles/Cans=no-threshold policy.",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    merged_fp = paths["interim"] / "collections_merged.parquet"
    assets_fp = paths["interim"] / "assets.parquet"

    if not merged_fp.exists():
        raise FileNotFoundError(f"Missing {merged_fp}. Run python scripts/run_pipeline.py first.")
    if not assets_fp.exists():
        raise FileNotFoundError(f"Missing {assets_fp}. Run python scripts/run_pipeline.py first.")

    # Load upstream files
    df_merged = pd.read_parquet(merged_fp)
    df_assets = pd.read_parquet(assets_fp)

    stream_density_lb_per_gal = {
        "Waste": float(args.waste_density),
        "Compostables": float(args.compost_density),
        "Bottles/Cans": float(args.recycling_density),
    }

    # Clean / standardize assets
    df_assets = clean_assets(
        df_assets=df_assets,
        default_threshold_pct=args.default_threshold_pct,
        enforce_stream_policy=args.use_stream_threshold_policy,
        stream_density_lb_per_gal=stream_density_lb_per_gal,
    )

    # Make sure time is parsed consistently
    df_merged["Collection_Time"] = pd.to_datetime(df_merged["Collection_Time"], errors="coerce")

    # Anchor date = day 0 of the planning horizon
    anchor_date = (
        pd.Timestamp(args.anchor_date)
        if args.anchor_date
        else df_merged["Collection_Time"].max().floor("D")
    )

    # Estimate fill-rate growth
    hist, bin_growth, stream_growth, overall_growth = compute_growth_rates(df_merged)

    # Find most recent service info
    last_service = compute_last_service(df_merged, anchor_date)

    # Build final table
    out = build_projection_table(
        assets=df_assets,
        last_service=last_service,
        bin_growth=bin_growth,
        stream_growth=stream_growth,
        overall_growth=overall_growth,
        horizon_days=args.horizon_days,
        default_bin_capacity_gal=args.default_bin_capacity_gal,
        default_service_min=args.default_service_min,
        default_travel_min=args.default_travel_min,
        safe_bin_content_lb=args.safe_bin_content_lb,
        hard_bin_content_lb=args.hard_bin_content_lb,
        processed_dir=paths["processed"],
    )

    out["anchor_date"] = anchor_date
    out["horizon_day_0"] = anchor_date

    # Save outputs
    out_csv = paths["processed"] / "bin_7day_projection_inputs.csv"
    out_parquet = paths["processed"] / "bin_7day_projection_inputs.parquet"

    out.to_csv(out_csv, index=False)
    out.to_parquet(out_parquet, index=False)

    # Print QA summary
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_parquet}")
    print(f"[INFO] anchor_date = {anchor_date.date()}")
    print(f"[INFO] assets in output = {len(out):,}")
    print(f"[INFO] required within horizon = {int(out['must_service_within_horizon'].sum()):,}")
    print(f"[INFO] growth rows retained after IQR filter = {len(hist):,}")
    print(f"[INFO] stream densities (lb/gal) = {stream_density_lb_per_gal}")
    print(f"[INFO] safe_bin_content_lb = {args.safe_bin_content_lb}")
    print(f"[INFO] hard_bin_content_lb = {args.hard_bin_content_lb}")

    preview_cols = [
        "Serial",
        "Description",
        "stream",
        "days_since_last_service",
        "daily_fill_growth_pct",
        "fill_day_0",
        "fill_day_6",
        "threshold_pct",
        "initial_deadline_threshold_no_service",
        "initial_deadline_interval_no_service",
        "initial_service_deadline_no_service",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    print("\nTop bins by urgency:")
    print(out[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
