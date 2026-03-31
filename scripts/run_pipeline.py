from __future__ import annotations

"""Bigbelly raw-data pipeline.

Key revisions in this version:
- converts collection timestamps from UTC to America/Los_Angeles before
  deriving date/day-of-week features;
- standardizes stream names more aggressively;
- preserves asset threshold metadata when available;
- keeps the pipeline zoning-free.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "America/Los_Angeles"


def repo_root() -> Path:
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


def parse_collection_time(series: pd.Series) -> pd.Series:
    """Parse Bigbelly timestamps as UTC, then convert to Berkeley local time."""
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)


def parse_fullness_to_pct(x: object) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s in {"", "-", "nan", "None", "Alert - Unknown Fullness"}:
        return None
    m = re.search(r"(\d+(\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1))
    return None


def find_header_row(csv_path: Path, required_token: str) -> int:
    with csv_path.open("r", errors="ignore") as f:
        for i, line in enumerate(f):
            if required_token in line:
                return i
    raise ValueError(f"Could not find header token '{required_token}' in {csv_path.name}")


def read_bigbelly_assets(csv_path: Path) -> pd.DataFrame:
    skip = find_header_row(csv_path, '"Description","Serial"')
    return pd.read_csv(csv_path, skiprows=skip)


def read_bigbelly_collections(csv_path: Path) -> pd.DataFrame:
    skip = find_header_row(csv_path, "Serial,Description,Capacity")
    return pd.read_csv(csv_path, skiprows=skip)


def clean_assets(df_assets: pd.DataFrame) -> pd.DataFrame:
    df = df_assets.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    if "Serial" not in df.columns:
        raise KeyError("Assets file missing 'Serial' column.")

    df["Serial"] = df["Serial"].astype(str).str.strip()

    if "Streams" in df.columns:
        df["Streams"] = df["Streams"].apply(canonical_stream)

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype(str).str.strip()

    for col in ["Lat", "Lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Fullness_Threshold" in df.columns:
        df["Fullness_Threshold_Pct"] = pd.to_numeric(
            df["Fullness_Threshold"].astype(str).str.extract(r"(\d+(\.\d+)?)")[0],
            errors="coerce",
        )

    return df


def clean_collections(df_col: pd.DataFrame) -> pd.DataFrame:
    df = df_col.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required = {"Serial", "Collection_Time", "Stream_Type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Collections file missing columns: {sorted(missing)}")

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Collection_Time"] = parse_collection_time(df["Collection_Time"])
    df = df[df["Collection_Time"].notna()].copy()

    if "Fullness_Level_at_Collection" in df.columns:
        df["Fullness_Pct"] = df["Fullness_Level_at_Collection"].apply(parse_fullness_to_pct)
    else:
        df["Fullness_Pct"] = np.nan

    df["Stream_Type"] = df["Stream_Type"].apply(canonical_stream)

    if "Reason" in df.columns:
        rs = df["Reason"].astype(str).str.strip().str.lower()
        df["Is_Alert"] = rs.eq("alert")
        df["Is_Fullness_Reason"] = rs.eq("fullness")
        df["Is_Age_Rule"] = rs.str.contains("age") | rs.str.contains("7-day")
        df["Is_Not_Ready"] = rs.eq("not ready")
    else:
        df["Is_Alert"] = False
        df["Is_Fullness_Reason"] = False
        df["Is_Age_Rule"] = False
        df["Is_Not_Ready"] = False

    df["date"] = df["Collection_Time"].dt.floor("D")
    df["day_of_week"] = df["Collection_Time"].dt.day_name()
    df["is_weekend"] = df["Collection_Time"].dt.weekday >= 5
    df = df.drop_duplicates()
    return df


def discover_collection_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("Daily Collection Activity - CLEAN *.csv"))
    if not files:
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


def build_interim(raw_dir: Path, interim_dir: Path) -> pd.DataFrame:
    assets_fp = pick_assets_file(raw_dir)
    df_assets_raw = read_bigbelly_assets(assets_fp)
    df_assets = clean_assets(df_assets_raw)

    col_files = discover_collection_files(raw_dir)
    if not col_files:
        raise FileNotFoundError(
            f"No collection CSVs found in {raw_dir}. Expected: Daily Collection Activity - CLEAN YYYY.csv"
        )

    dfs: list[pd.DataFrame] = []
    for fp in col_files:
        tmp = clean_collections(read_bigbelly_collections(fp))
        tmp["Source_File"] = fp.name
        dfs.append(tmp)

    df_col = pd.concat(dfs, ignore_index=True)

    keep_asset_cols = [
        c
        for c in ["Serial", "Description", "Streams", "Lat", "Lng", "Status", "Fullness_Threshold_Pct"]
        if c in df_assets.columns
    ]
    df_merged = df_col.merge(df_assets[keep_asset_cols], on="Serial", how="left")

    if "Lat" in df_merged.columns and "Lng" in df_merged.columns:
        df_merged["Has_Location"] = df_merged["Lat"].notna() & df_merged["Lng"].notna()

    df_assets.to_parquet(interim_dir / "assets.parquet", index=False)
    df_col.to_parquet(interim_dir / "collections_raw.parquet", index=False)
    df_merged.to_parquet(interim_dir / "collections_merged.parquet", index=False)

    return df_merged


def build_processed(df_merged: pd.DataFrame, processed_dir: Path) -> None:
    df = df_merged.copy()
    df = df[df["Collection_Time"].notna()].copy()

    daily_by_stream = (
        df.groupby(["date", "Stream_Type"]).size().rename("collections").reset_index().sort_values(["Stream_Type", "date"])
    )
    daily_by_stream.to_parquet(processed_dir / "daily_counts_by_stream.parquet", index=False)

    if "Is_Alert" in df.columns:
        daily_alert = (
            df.groupby(["date", "Stream_Type"])["Is_Alert"]
            .mean()
            .rename("alert_rate")
            .reset_index()
            .sort_values(["Stream_Type", "date"])
        )
        daily_alert.to_parquet(processed_dir / "daily_alert_rate_by_stream.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bigbelly pipeline: raw CSVs -> interim parquet -> processed parquet"
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip raw cleaning and use an existing interim merged parquet file.",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)
    merged_fp = paths["interim"] / "collections_merged.parquet"

    if args.skip_clean:
        if not merged_fp.exists():
            raise FileNotFoundError(f"--skip-clean set but missing: {merged_fp}")
        df_merged = pd.read_parquet(merged_fp)
        print(f"[OK] Loaded existing interim file: {merged_fp}")
    else:
        df_merged = build_interim(paths["raw"], paths["interim"])
        print(f"[OK] Wrote interim outputs to: {paths['interim']}")
        print(f"[INFO] merged rows: {len(df_merged):,}")

    build_processed(df_merged, paths["processed"])
    print(f"[OK] Wrote processed outputs to: {paths['processed']}")

    if "Has_Location" in df_merged.columns:
        print(f"[QA] % with location: {float(df_merged['Has_Location'].mean()):.1%}")
    if "Fullness_Pct" in df_merged.columns:
        print(f"[QA] % fullness known: {float(df_merged['Fullness_Pct'].notna().mean()):.1%}")
    print(f"[QA] timestamps localized to {LOCAL_TIMEZONE}")
    print("[DONE] run_pipeline complete.")


if __name__ == "__main__":
    main()
