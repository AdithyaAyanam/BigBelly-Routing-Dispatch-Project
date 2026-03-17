# scripts/build_projected_fill.py
# -------------------------------------------------------------
# Purpose:
# Build a 7-day projected fill input table for the optimization model.
#
# This script reads the cleaned/interim Bigbelly data produced by
# run_pipeline.py, estimates how quickly each bin fills over time,
# estimates the current fill level for each bin, projects fill levels
# for the next 7 days, and computes the day by which each bin must be
# serviced based on:
#   1) fullness threshold, and/or
#   2) 7-day service rule
#
# Output:
#   data/processed/bin_7day_projection_inputs.csv
#   data/processed/bin_7day_projection_inputs.parquet
# -------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


## Return the project root directory.
"""
    Assumes this file lives in:
    project_root/scripts/build_projected_fill.py
so the repo root is one level above /scripts.
"""
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


## Ensure the expected data folders exist and return them.
"""
    Creates:
      - data/interim
      - data/processed
"""
def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    interim_dir = data_dir / "interim"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir, "interim": interim_dir}

##Standardize stream names into a small clean set.
"""
    Example:
      'compostables' -> 'Compostables'
      'waste'        -> 'Waste'
      'bottles/cans' -> 'Bottles/Cans'
"""      
def canonical_stream(x: object) -> str:
    s = str(x).strip()
    lookup = {
        "compostables": "Compostables",
        "waste": "Waste",
        "bottles/cans": "Bottles/Cans",
        "single stream": "Single Stream",
    }
    return lookup.get(s.lower(), s if s else "Unknown")

## Clean the assets/interim bin master table.
"""

    Main tasks:
      - verify Serial exists
      - clean Serial and Status text
      - extract threshold percentage if present
      - fall back to a default threshold if missing
      - standardize stream labels
      - coerce Lat/Lng to numeric if present
"""
def clean_assets(df_assets: pd.DataFrame, default_threshold_pct: float) -> pd.DataFrame:
    df = df_assets.copy()

    if "Serial" not in df.columns:
        raise KeyError("assets.parquet missing 'Serial'")

    # Standardize the bin ID
    df["Serial"] = df["Serial"].astype(str).str.strip()

    # Clean Status if it exists
    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip()

    # Extract threshold from a text column if available
    if "Fullness_Threshold" in df.columns:
        df["threshold_pct"] = pd.to_numeric(
            df["Fullness_Threshold"].astype(str).str.extract(r"(\d+(\.\d+)?)")[0],
            errors="coerce",
        )
    elif "Fullness_Threshold_Pct" in df.columns:
        df["threshold_pct"] = pd.to_numeric(df["Fullness_Threshold_Pct"], errors="coerce")
    else:
        df["threshold_pct"] = np.nan
    
    # Replace missing thresholds with a prototype default, e.g. 60%
    df["threshold_pct"] = df["threshold_pct"].fillna(default_threshold_pct)

    # Standardize the stream label from the assets file
    if "Streams" in df.columns:
        df["stream"] = df["Streams"].apply(canonical_stream)
    else:
        df["stream"] = "Unknown"

    # Convert location columns to numeric if present
    for col in ["Lat", "Lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

## Estimate daily fill growth rates from historical collection data.
"""
    Logic:
      - for each bin, sort collections by time
      - compute the time gap between consecutive collections
      - estimate growth rate as:
            fullness_at_collection / days_since_previous_collection
      - summarize at three levels:
            a) bin-specific median growth
            b) stream-level median growth
            c) overall median growth

    Returns:
      hist          : cleaned history rows used in the calculation
      bin_growth    : Series indexed by Serial
      stream_growth : Series indexed by Stream_Type
      overall_growth: single fallback number
"""
def compute_growth_rates(df_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    df = df_merged.copy()
    needed = ["Serial", "Collection_Time", "Fullness_Pct"]

    # Make sure the merged file has the fields needed for historical rate estimation
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"collections_merged.parquet missing '{col}'")

    # If stream type is missing, create a fallback column
    if "Stream_Type" not in df.columns:
        df["Stream_Type"] = "Unknown"

    # Clean key fields
    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")
    df["Stream_Type"] = df["Stream_Type"].apply(canonical_stream)

    # Only use rows with valid time and valid fullness
    hist = df.dropna(subset=["Collection_Time", "Fullness_Pct"]).copy()
    hist = hist.sort_values(["Serial", "Collection_Time"]).reset_index(drop=True)

    # Previous collection time for the same bin
    hist["prev_collection_time"] = hist.groupby("Serial")["Collection_Time"].shift(1)

    # Gap in days between consecutive collections
    hist["gap_days"] = (
        (hist["Collection_Time"] - hist["prev_collection_time"]).dt.total_seconds() / 86400.0
    )
    # Ignore tiny or missing gaps
    hist = hist[(hist["gap_days"].notna()) & (hist["gap_days"] > 0.5)].copy()

    # Estimate daily growth rate as fullness divided by number of days since last collection
    hist["growth_pct_per_day"] = hist["Fullness_Pct"] / hist["gap_days"]

    # Replace bad values with NaN
    hist.loc[
        ~np.isfinite(hist["growth_pct_per_day"]) | (hist["growth_pct_per_day"] < 0),
        "growth_pct_per_day",
    ] = np.nan

    # Keep only valid growth values
    hist = hist[hist["growth_pct_per_day"].notna()].copy()

    # Median growth rate by exact bin
    bin_growth = hist.groupby("Serial")["growth_pct_per_day"].median()

    # Median growth rate by stream
    stream_growth = hist.groupby("Stream_Type")["growth_pct_per_day"].median()

    # Overall fallback rate if a bin and stream both lack enough history
    overall_growth = float(hist["growth_pct_per_day"].median()) if not hist.empty else 12.0

    return hist, bin_growth, stream_growth, overall_growth


## Find the most recent collection date for each bin and compute
## days since last service relative to the anchor date.
   """
    Example:
      if last service = Feb 15 and anchor date = Feb 20
      then days_since_last_service = 5
    """
def compute_last_service(df_merged: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.DataFrame:
    df = df_merged.copy()
    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")

    # Only use rows with valid timestamps
    df = df.dropna(subset=["Collection_Time"]).copy()
    df = df.sort_values(["Serial", "Collection_Time"])

    # Keep the latest record for each bin
    last = df.groupby("Serial").tail(1).copy()

    # Floor to day so we compare dates rather than timestamps
    last["last_service_date"] = last["Collection_Time"].dt.floor("D")

 3   # Compute how many days since the bin was last serviced
    last["days_since_last_service"] = (
        anchor_date.floor("D") - last["last_service_date"]
    ).dt.days.clip(lower=0)

    # Keep a compact output table
    keep_cols = ["Serial", "Collection_Time", "last_service_date", "days_since_last_service"]
    if "Description" in last.columns:
        keep_cols.append("Description")
    if "Stream_Type" in last.columns:
        keep_cols.append("Stream_Type")

    return last[keep_cols].rename(columns={"Collection_Time": "last_collection_time"})

 ## Return the first day in the horizon when projected fill reaches/exceeds
 ## the bin's threshold.

    """
    If the threshold is never reached in the horizon, return NaN.
    """

##  Return the day by which the bin must be serviced due to the 7-day rule.
"""
    Example:
      days_since_last_service = 6 -> deadline = day 1
      days_since_last_service = 7 or more -> deadline = day 0

    If the bin is not due within the horizon, return NaN.
"""    
def threshold_deadline(row: pd.Series, horizon_days: int) -> float:
    for d in range(horizon_days):
        if row[f"fill_day_{d}"] >= row["threshold_pct"]:
            return float(d)
    return np.nan


##  Return the day by which the bin must be serviced due to the 7-day rule.
    """
    Example:
      days_since_last_service = 6 -> deadline = day 1
      days_since_last_service = 7 or more -> deadline = day 0

    If the bin is not due within the horizon, return NaN.
    """

def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    remaining = 7 - float(days_since_last_service)
    if remaining <= 0:
        return 0.0
    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))
    return np.nan


## Build the final 7-day projected-fill input table.
    
    """
    Construct the final 7-day projected-fill modeling table.

    For each eligible bin, this function:
      - merges asset and service-history information,
      - assigns a representative daily fill-growth estimate,
      - estimates the current fill level at the anchor date,
      - projects fill levels across the planning horizon,
      - computes the earliest service deadline induced by either
        the fullness threshold or the 7-day service rule,
      - attaches prototype operational parameters needed by the
        downstream optimization model.

    Returns
    -------
    pd.DataFrame
        A modeling-ready table containing projected fill levels,
        service deadlines, and operational proxy parameters for
        each bin in the 7-day horizon.
    """  
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
) -> pd.DataFrame:
    base = assets.copy()
    
    # Keep only active/in-service bins for the prototype
    if "Status" in base.columns:
        base = base[
            base["Status"].fillna("").astype(str).str.lower().isin(
                ["in service", "in_service", "active", ""]
            )
        ].copy()

    # Keep only the fields we need for the projection table
    keep_cols = ["Serial", "Description", "stream", "threshold_pct", "Lat", "Lng"]
    keep_cols = [c for c in keep_cols if c in base.columns]
    base = base[keep_cols].drop_duplicates(subset=["Serial"]).copy()

    # Add last known service date and days since service
    base = base.merge(
        last_service[["Serial", "last_service_date", "days_since_last_service"]],
        on="Serial",
        how="left",
    )

    # Standardize stream names
    base["stream"] = base["stream"].fillna("Unknown").apply(canonical_stream)

    # Preserve raw days-since-last-service before capping/filtering
    base["days_since_last_service_raw"] = base["days_since_last_service"]

    # Keep only bins with reasonably recent service history for the prototype
    base = base[
    base["days_since_last_service_raw"].notna() &
    (base["days_since_last_service_raw"] <= 21)
    ].copy()

    # Cap days-since-last-service so stale records do not explode the fill estimate
    base["days_since_last_service"] = base["days_since_last_service_raw"].clip(lower=0, upper=10)

    # Assign a growth estimate using this priority:
    #   1) bin-specific historical median
    #   2) stream-level historical median
    #   3) overall historical median
    base["daily_fill_growth_pct"] = base["Serial"].map(bin_growth)
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(base["stream"].map(stream_growth))
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(overall_growth)
    
    # Estimate current fill at day 0 using:
    #   current_fill ≈ daily_growth * days_since_last_service
    # Cap at 125% for prototype realism
    base["current_fill_pct_est"] = (
    base["daily_fill_growth_pct"] * base["days_since_last_service"]
    ).clip(lower=0, upper=125)

    # Build projected fill values for day 0..day 6
    for d in range(horizon_days):
        base[f"fill_day_{d}"] = base["current_fill_pct_est"] + base["daily_fill_growth_pct"] * d

    # Compute the deadline induced by fullness threshold
    base["deadline_threshold"] = base.apply(lambda row: threshold_deadline(row, horizon_days), axis=1)
    
    # Compute the deadline induced by the 7-day service rule
    base["deadline_interval"] = base["days_since_last_service"].apply(lambda x: interval_deadline(x, horizon_days))

    # True deadline = earlier of threshold deadline and interval deadline
    base["service_deadline"] = base[["deadline_threshold", "deadline_interval"]].min(axis=1, skipna=True)
    
    # If neither rule requires service in the horizon, leave deadline as NaN
    base.loc[
        base[["deadline_threshold", "deadline_interval"]].isna().all(axis=1),
        "service_deadline",
    ] = np.nan

    # Flag bins that must be serviced sometime within day 0..day 6
    base["must_service_within_horizon"] = base["service_deadline"].notna()

    # Prototype defaults for the small optimization instance
    base["bin_capacity_gal"] = float(default_bin_capacity_gal)
    base["avg_service_min"] = float(default_service_min)
    base["avg_travel_proxy_min"] = float(default_travel_min)

    # Final ordered set of columns for export
    cols = (
        [
            "Serial",
            "Description",
            "stream",
            "threshold_pct",
            "days_since_last_service",
            "daily_fill_growth_pct",
            "current_fill_pct_est",
        ]
        + [f"fill_day_{d}" for d in range(horizon_days)]
        + [
            "deadline_threshold",
            "deadline_interval",
            "service_deadline",
            "must_service_within_horizon",
            "bin_capacity_gal",
            "avg_service_min",
            "avg_travel_proxy_min",
            "Lat",
            "Lng",
        ]
    )
    cols = [c for c in cols if c in base.columns]

    # Sort by urgency so the most urgent bins appear first in previews
    return base[cols].sort_values(["service_deadline", "Serial"], na_position="last").reset_index(drop=True)


##    Main entry point for the script.
"""
 Workflow
    --------
    1. Parse user-specified runtime arguments.
    2. Load interim asset and merged collection data.
    3. Select the anchor date defining day 0 of the planning horizon.
    4. Estimate historical fill-growth rates at the bin, stream,
       and overall levels.
    5. Compute the most recent service date and days since last
       service for each bin.
    6. Construct the projected-fill input table for the optimization
       model over the 7-day horizon.
    7. Save the resulting table in CSV and Parquet formats.
    8. Print summary statistics and a preview for quality checks.
"""
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 7-day projected-fill optimization inputs from interim parquet files."
    )
    parser.add_argument("--anchor-date", type=str, default=None, help="YYYY-MM-DD. Defaults to latest collection date in collections_merged.parquet.")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--default-threshold-pct", type=float, default=60.0)
    parser.add_argument("--default-bin-capacity-gal", type=float, default=60.0)
    parser.add_argument("--default-service-min", type=float, default=6.0)
    parser.add_argument("--default-travel-min", type=float, default=10.0)
    args = parser.parse_args()

    # Resolve project folders
    root = repo_root()
    paths = ensure_dirs(root)

    # Expected upstream inputs created by run_pipeline.py
    merged_fp = paths["interim"] / "collections_merged.parquet"
    assets_fp = paths["interim"] / "assets.parquet"

    if not merged_fp.exists():
        raise FileNotFoundError(f"Missing {merged_fp}. Run python scripts/run_pipeline.py first.")
    if not assets_fp.exists():
        raise FileNotFoundError(f"Missing {assets_fp}. Run python scripts/run_pipeline.py first.")

    # Load interim datasets
    df_merged = pd.read_parquet(merged_fp)
    df_assets = pd.read_parquet(assets_fp)

    # Clean assets and standardize thresholds/streams
    df_assets = clean_assets(df_assets, default_threshold_pct=args.default_threshold_pct)

    # Ensure timestamps are parsed
    df_merged["Collection_Time"] = pd.to_datetime(df_merged["Collection_Time"], errors="coerce")
   
    # Choose the anchor date for day 0
    if args.anchor_date:
        anchor_date = pd.Timestamp(args.anchor_date)
    else:
        anchor_date = df_merged["Collection_Time"].max().floor("D")

    # Estimate historical fill growth rates
    _, bin_growth, stream_growth, overall_growth = compute_growth_rates(df_merged)

    # Compute most recent service information for each bin
    last_service = compute_last_service(df_merged, anchor_date)

    # Build the final 7-day projected-fill input table
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
    )
    
    # Add the anchor date to the output for traceability
    out["anchor_date"] = anchor_date
    out["horizon_day_0"] = anchor_date

    # Save outputs in both CSV and Parquet format
    out_csv = paths["processed"] / "bin_7day_projection_inputs.csv"
    out_parquet = paths["processed"] / "bin_7day_projection_inputs.parquet"
    out.to_csv(out_csv, index=False)
    out.to_parquet(out_parquet, index=False)

    # Print summary information for quick QA
    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_parquet}")
    print(f"[INFO] anchor_date = {anchor_date.date()}")
    print(f"[INFO] assets in output = {len(out):,}")
    print(f"[INFO] required within horizon = {int(out['must_service_within_horizon'].sum()):,}")
    print(f"[INFO] median daily growth pct = {overall_growth:.2f}")

    preview_cols = [
        "Serial",
        "Description",
        "stream",
        "days_since_last_service",
        "daily_fill_growth_pct",
        "fill_day_0",
        "fill_day_6",
        "threshold_pct",
        "service_deadline",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print("\nTop bins by urgency:")
    print(out[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
