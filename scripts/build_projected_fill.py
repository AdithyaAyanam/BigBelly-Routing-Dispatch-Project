from __future__ import annotations

"""Build 7-day projected fill inputs for the Bigbelly scheduling model.

Revisions in this version:
- applies the current pickup-threshold policy directly in code:
    * Waste: 60%
    * Compostables: 60%
    * Bottles/Cans: no fill threshold (7-day rule only)
- uses 150 gallons as the default bin capacity;
- attaches stream densities for downstream mass calculations;
- computes projected pickup mass as well as projected pickup volume;
- adds an IQR filter to suppress implausible fill-rate spikes;
- remains completely zoning-free.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STREAM_DENSITY_LB_PER_GAL = {
    "Waste": 1.0,
    "Compostables": 2.9,
    "Bottles/Cans": 0.3,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    interim_dir = data_dir / "interim"
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir, "interim": interim_dir}


def canonical_stream(x: object) -> str:
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


def policy_threshold_for_stream(stream: str, fallback_threshold_pct: float = 60.0) -> float:
    """Return the current Zero Waste threshold policy by stream.

    Waste and Compostables use a 60% trigger.
    Bottles/Cans has no fill trigger and is governed only by the 7-day rule.
    """
    stream = canonical_stream(stream)
    if stream in {"Waste", "Compostables"}:
        return float(fallback_threshold_pct)
    if stream == "Bottles/Cans":
        return np.nan
    return float(fallback_threshold_pct)


def clean_assets(
    df_assets: pd.DataFrame,
    default_threshold_pct: float,
    enforce_stream_policy: bool,
) -> pd.DataFrame:
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

    for col in ["Lat", "Lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["density_lb_per_gal"] = df["stream"].map(STREAM_DENSITY_LB_PER_GAL).fillna(1.0)
    return df


def compute_growth_rates(df_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, float]:
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

    hist = df.dropna(subset=["Collection_Time", "Fullness_Pct"]).copy()
    hist = hist.sort_values(["Serial", "Collection_Time"]).reset_index(drop=True)
    hist["prev_collection_time"] = hist.groupby("Serial")["Collection_Time"].shift(1)
    hist["gap_days"] = (
        (hist["Collection_Time"] - hist["prev_collection_time"]).dt.total_seconds() / 86400.0
    )
    hist = hist[(hist["gap_days"].notna()) & (hist["gap_days"] > 0.5)].copy()
    hist["growth_pct_per_day"] = hist["Fullness_Pct"] / hist["gap_days"]
    hist.loc[
        ~np.isfinite(hist["growth_pct_per_day"]) | (hist["growth_pct_per_day"] < 0),
        "growth_pct_per_day",
    ] = np.nan
    hist = hist[hist["growth_pct_per_day"].notna()].copy()

    if not hist.empty:
        q1 = hist["growth_pct_per_day"].quantile(0.25)
        q3 = hist["growth_pct_per_day"].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 3.0 * iqr
        hist = hist[hist["growth_pct_per_day"] <= upper].copy()

    bin_growth = hist.groupby("Serial")["growth_pct_per_day"].median()
    stream_growth = hist.groupby("Stream_Type")["growth_pct_per_day"].median()
    overall_growth = float(hist["growth_pct_per_day"].median()) if not hist.empty else 12.0
    return hist, bin_growth, stream_growth, overall_growth


def compute_last_service(df_merged: pd.DataFrame, anchor_date: pd.Timestamp) -> pd.DataFrame:
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


def threshold_deadline(row: pd.Series, horizon_days: int) -> float:
    threshold = row.get("threshold_pct", np.nan)
    if pd.isna(threshold):
        return np.nan
    for d in range(horizon_days):
        if row[f"fill_day_{d}"] >= threshold:
            return float(d)
    return np.nan


def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    remaining = 7 - float(days_since_last_service)
    if remaining <= 0:
        return 0.0
    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))
    return np.nan


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

    if "Status" in base.columns:
        base = base[
            base["Status"].fillna("").astype(str).str.lower().isin(["in service", "in_service", "active", ""])
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
        ]
        if c in base.columns
    ]
    base = base[keep_cols].drop_duplicates(subset=["Serial"]).copy()

    base = base.merge(
        last_service[["Serial", "last_service_date", "days_since_last_service"]],
        on="Serial",
        how="left",
    )

    base["stream"] = base["stream"].fillna("Unknown").apply(canonical_stream)
    base["days_since_last_service_raw"] = base["days_since_last_service"]
    base = base[
        base["days_since_last_service_raw"].notna() & (base["days_since_last_service_raw"] <= 21)
    ].copy()
    base["days_since_last_service"] = base["days_since_last_service_raw"].clip(lower=0, upper=10)

    base["daily_fill_growth_pct"] = base["Serial"].map(bin_growth)
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(base["stream"].map(stream_growth))
    base["daily_fill_growth_pct"] = base["daily_fill_growth_pct"].fillna(overall_growth)

    base["current_fill_pct_est"] = (
        base["daily_fill_growth_pct"] * base["days_since_last_service"]
    ).clip(lower=0, upper=125)

    base["bin_capacity_gal"] = float(default_bin_capacity_gal)
    base["avg_service_min"] = float(default_service_min)
    base["avg_travel_proxy_min"] = float(default_travel_min)

    for d in range(horizon_days):
        fill_col = f"fill_day_{d}"
        gal_col = f"pickup_gal_day_{d}"
        lb_col = f"pickup_lb_day_{d}"
        base[fill_col] = (base["current_fill_pct_est"] + base["daily_fill_growth_pct"] * d).clip(upper=125)
        base[gal_col] = base["bin_capacity_gal"] * base[fill_col] / 100.0
        base[lb_col] = base[gal_col] * base["density_lb_per_gal"]

    base["deadline_threshold"] = base.apply(lambda row: threshold_deadline(row, horizon_days), axis=1)
    base["deadline_interval"] = base["days_since_last_service"].apply(
        lambda x: interval_deadline(x, horizon_days)
    )
    base["service_deadline"] = base[["deadline_threshold", "deadline_interval"]].min(axis=1, skipna=True)
    base.loc[
        base[["deadline_threshold", "deadline_interval"]].isna().all(axis=1),
        "service_deadline",
    ] = np.nan
    base["must_service_within_horizon"] = base["service_deadline"].notna()

    cols = (
        [
            "Serial",
            "Description",
            "stream",
            "threshold_pct",
            "threshold_pct_raw",
            "density_lb_per_gal",
            "days_since_last_service",
            "daily_fill_growth_pct",
            "current_fill_pct_est",
            "bin_capacity_gal",
            "avg_service_min",
            "avg_travel_proxy_min",
        ]
        + [f"fill_day_{d}" for d in range(horizon_days)]
        + [f"pickup_gal_day_{d}" for d in range(horizon_days)]
        + [f"pickup_lb_day_{d}" for d in range(horizon_days)]
        + [
            "deadline_threshold",
            "deadline_interval",
            "service_deadline",
            "must_service_within_horizon",
            "Lat",
            "Lng",
        ]
    )
    cols = [c for c in cols if c in base.columns]

    return base[cols].sort_values(["service_deadline", "Serial"], na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 7-day projected-fill optimization inputs from interim parquet files."
    )
    parser.add_argument("--anchor-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--default-threshold-pct", type=float, default=60.0)
    parser.add_argument("--default-bin-capacity-gal", type=float, default=150.0)
    parser.add_argument("--default-service-min", type=float, default=4.0)
    parser.add_argument("--default-travel-min", type=float, default=10.0)
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

    df_merged = pd.read_parquet(merged_fp)
    df_assets = pd.read_parquet(assets_fp)

    df_assets = clean_assets(
        df_assets,
        default_threshold_pct=args.default_threshold_pct,
        enforce_stream_policy=args.use_stream_threshold_policy,
    )

    df_merged["Collection_Time"] = pd.to_datetime(df_merged["Collection_Time"], errors="coerce")
    anchor_date = pd.Timestamp(args.anchor_date) if args.anchor_date else df_merged["Collection_Time"].max().floor("D")

    hist, bin_growth, stream_growth, overall_growth = compute_growth_rates(df_merged)
    last_service = compute_last_service(df_merged, anchor_date)
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

    out["anchor_date"] = anchor_date
    out["horizon_day_0"] = anchor_date

    out_csv = paths["processed"] / "bin_7day_projection_inputs.csv"
    out_parquet = paths["processed"] / "bin_7day_projection_inputs.parquet"
    out.to_csv(out_csv, index=False)
    out.to_parquet(out_parquet, index=False)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_parquet}")
    print(f"[INFO] anchor_date = {anchor_date.date()}")
    print(f"[INFO] assets in output = {len(out):,}")
    print(f"[INFO] required within horizon = {int(out['must_service_within_horizon'].sum()):,}")
    print(f"[INFO] growth rows retained after IQR filter = {len(hist):,}")

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
