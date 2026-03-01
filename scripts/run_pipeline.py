# scripts/run_pipeline.py
"""
DATA PIPELINE (no EDA, no forecasting):
1) Load raw collections + assets from data/raw/
2) Clean types (timestamps, fullness %, stream labels) + remove duplicates
3) Merge collections ↔ assets on Serial to attach bin metadata + Lat/Lng
4) Save:
   - data/interim/{assets, collections_raw, collections_merged}.parquet
   - data/processed/daily_counts_by_stream.parquet (and optional alert-rate table)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# Repo paths
# -----------------------------
def repo_root() -> Path:
    # scripts/run_pipeline.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "data": data_dir,
        "raw": raw_dir,
        "interim": interim_dir,
        "processed": processed_dir,
    }


# -----------------------------
# Bigbelly export reading
# (handles metadata lines before the real CSV header)
# -----------------------------
def find_header_row(csv_path: Path, required_token: str) -> int:
    with csv_path.open("r", errors="ignore") as f:
        for i, line in enumerate(f):
            if required_token in line:
                return i
    raise ValueError(f"Could not find header token '{required_token}' in {csv_path.name}")


def read_bigbelly_assets(csv_path: Path) -> pd.DataFrame:
    # Assets header often begins with: "Description","Serial",...
    skip = find_header_row(csv_path, '"Description","Serial"')
    return pd.read_csv(csv_path, skiprows=skip)


def read_bigbelly_collections(csv_path: Path) -> pd.DataFrame:
    # Collections header often begins with: Serial,Description,Capacity,...
    skip = find_header_row(csv_path, "Serial,Description,Capacity")
    return pd.read_csv(csv_path, skiprows=skip)


# -----------------------------
# Cleaning helpers
# -----------------------------
def parse_fullness_to_pct(x: object) -> float | None:
    """
    Converts '60%' -> 60.0.
    Returns None for blanks, '-', 'Alert - Unknown Fullness', etc.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s in {"", "-", "nan", "None"}:
        return None
    m = re.search(r"(\d+(\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    return None


def clean_assets(df_assets: pd.DataFrame) -> pd.DataFrame:
    df = df_assets.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Serial" not in df.columns:
        raise KeyError("Assets file missing 'Serial' column.")

    df["Serial"] = df["Serial"].astype(str).str.strip()

    if "Lat" in df.columns:
        df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    if "Lng" in df.columns:
        df["Lng"] = pd.to_numeric(df["Lng"], errors="coerce")

    # Optional: extract threshold %
    if "Fullness_Threshold" in df.columns:
        df["Fullness_Threshold_Pct"] = (
            df["Fullness_Threshold"].astype(str).str.extract(r"(\d+(\.\d+)?)")[0]
        )
        df["Fullness_Threshold_Pct"] = pd.to_numeric(df["Fullness_Threshold_Pct"], errors="coerce")

    return df


def clean_collections(df_col: pd.DataFrame) -> pd.DataFrame:
    df = df_col.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required = {"Serial", "Collection_Time", "Stream_Type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Collections file missing columns: {sorted(missing)}")

    df["Serial"] = df["Serial"].astype(str).str.strip()

    # Parse timestamp
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")
    df = df[df["Collection_Time"].notna()].copy()

    # Fullness
    if "Fullness_Level_at_Collection" in df.columns:
        df["Fullness_Pct"] = df["Fullness_Level_at_Collection"].apply(parse_fullness_to_pct)
    else:
        df["Fullness_Pct"] = np.nan

    # Standardize stream label
    df["Stream_Type"] = df["Stream_Type"].astype(str).str.strip()

    # Reason flags
    if "Reason" in df.columns:
        rs = df["Reason"].astype(str).str.strip().str.lower()
        df["Is_Alert"] = rs.eq("alert")
        df["Is_Fullness_Reason"] = rs.eq("fullness")
    else:
        df["Is_Alert"] = False
        df["Is_Fullness_Reason"] = False

    # Derived time columns
    df["date"] = df["Collection_Time"].dt.floor("D")
    df["day_of_week"] = df["Collection_Time"].dt.day_name()
    df["is_weekend"] = df["Collection_Time"].dt.weekday >= 5

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    return df


# -----------------------------
# File discovery
# -----------------------------
def discover_collection_files(raw_dir: Path) -> list[Path]:
    # Preferred pattern: Daily Collection Activity - CLEAN 2024.csv, etc.
    files = sorted(raw_dir.glob("Daily Collection Activity - CLEAN *.csv"))
    if not files:
        # fallback
        files = sorted(raw_dir.glob("*Collection*Activity*.csv"))
    return files


def pick_assets_file(raw_dir: Path) -> Path:
    preferred = raw_dir / "Account Assets - CLEAN.csv"
    if preferred.exists():
        return preferred
    cands = sorted(raw_dir.glob("*Asset*.csv")) + sorted(raw_dir.glob("*Assets*.csv"))
    if not cands:
        raise FileNotFoundError(
            "No assets CSV found in data/raw. Expected 'Account Assets - CLEAN.csv' or similar."
        )
    return cands[0]


# -----------------------------
# Pipeline steps
# -----------------------------
def build_interim(raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    # Load + clean assets
    assets_fp = pick_assets_file(raw_dir)
    df_assets_raw = read_bigbelly_assets(assets_fp)
    df_assets = clean_assets(df_assets_raw)

    # Load + clean collections (concat years)
    col_files = discover_collection_files(raw_dir)
    if not col_files:
        raise FileNotFoundError(
            f"No collection CSVs found in {raw_dir}. "
            "Expected: Daily Collection Activity - CLEAN YYYY.csv"
        )

    dfs: list[pd.DataFrame] = []
    for fp in col_files:
        tmp_raw = read_bigbelly_collections(fp)
        tmp = clean_collections(tmp_raw)
        tmp["Source_File"] = fp.name
        dfs.append(tmp)

    df_col = pd.concat(dfs, ignore_index=True)

    # Merge: collections ↔ assets
    keep_asset_cols = [c for c in ["Serial", "Description", "Streams", "Lat", "Lng", "Status"] if c in df_assets.columns]
    df_merged = df_col.merge(df_assets[keep_asset_cols], on="Serial", how="left")

    if "Lat" in df_merged.columns and "Lng" in df_merged.columns:
        df_merged["Has_Location"] = df_merged["Lat"].notna() & df_merged["Lng"].notna()

    # Write interim datasets
    df_assets.to_parquet(interim_dir / "assets.parquet", index=False)
    df_col.to_parquet(interim_dir / "collections_raw.parquet", index=False)
    df_merged.to_parquet(interim_dir / "collections_merged.parquet", index=False)

    return df_merged


def build_processed(df_merged: pd.DataFrame, processed_dir: Path) -> None:
    """
    Build small, modeling-ready datasets that other scripts can reuse.
    Keep these stable and aggregated.
    """
    df = df_merged.copy()
    df = df[df["Collection_Time"].notna()].copy()

    # Daily counts by stream (forecast-friendly)
    daily_by_stream = (
        df.groupby(["date", "Stream_Type"])
        .size()
        .rename("collections")
        .reset_index()
        .sort_values(["Stream_Type", "date"])
    )
    daily_by_stream.to_parquet(processed_dir / "daily_counts_by_stream.parquet", index=False)

    # Optional: alert rate by day + stream
    if "Is_Alert" in df.columns:
        daily_alert = (
            df.groupby(["date", "Stream_Type"])["Is_Alert"]
            .mean()
            .rename("alert_rate")
            .reset_index()
            .sort_values(["Stream_Type", "date"])
        )
        daily_alert.to_parquet(processed_dir / "daily_alert_rate_by_stream.parquet", index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bigbelly pipeline (CLEAN/MERGE ONLY): raw CSVs -> interim + processed parquet"
    )
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning; use existing interim merged parquet")
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    merged_fp = paths["interim"] / "collections_merged.parquet"

    # Step 1: Build interim
    if args.skip_clean:
        if not merged_fp.exists():
            raise FileNotFoundError(f"--skip-clean set but missing: {merged_fp}")
        df_merged = pd.read_parquet(merged_fp)
        print(f"[OK] Loaded existing: {merged_fp}")
    else:
        df_merged = build_interim(paths["raw"], paths["interim"])
        print(f"[OK] Wrote interim to: {paths['interim']}")
        print(f"     merged rows: {len(df_merged):,}")

    # Step 2: Build processed
    build_processed(df_merged, paths["processed"])
    print(f"[OK] Wrote processed to: {paths['processed']}")

    # Quick QA prints
    pct_loc = None
    if "Has_Location" in df_merged.columns:
        pct_loc = float(df_merged["Has_Location"].mean())
    pct_full = float(df_merged["Fullness_Pct"].notna().mean()) if "Fullness_Pct" in df_merged.columns else None

    print("[QA]")
    print(f"  rows: {len(df_merged):,}")
    if pct_loc is not None:
        print(f"  % with location: {pct_loc:.1%}")
    if pct_full is not None:
        print(f"  % fullness known: {pct_full:.1%}")

    print("[DONE] run_pipeline complete.")
    print("Next:")
    print("  - python scripts/run_dashboard_metrics.py   (KPI tables)")
    print("  - python scripts/make_viz.py               (EDA plots)")
    print("  - forecasting script (separate)            (forecasts)")


if __name__ == "__main__":
    main()