# scripts/make_viz.py
"""
EDA PLOTS 

Reads:
  - data/interim/collections_merged.parquet

Writes:
  - outputs/eda/*.png (EDA charts)
  - outputs/eda/*.csv (small helper summaries)
  - outputs/eda/qa_summary.json (quick quality checks)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def repo_root() -> Path:
    # scripts/make_viz.py -> repo root is parent of scripts/
    return Path(__file__).resolve().parents[1]

def safe_name(x: object) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_")

def ensure_outdir(root: Path) -> Path:
    out_dir = root / "outputs" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_interim(root: Path) -> pd.DataFrame:
    fp = root / "data" / "interim" / "collections_merged.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing {fp}. Run: python scripts/run_pipeline.py first."
        )
    return pd.read_parquet(fp)


def save_line_plot(x, y, title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_bar_plot(categories, values, title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure()
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    root = repo_root()
    out_dir = ensure_outdir(root)
    df = load_interim(root)

    # --- Basic checks + standard fields
    if "Collection_Time" not in df.columns:
        raise KeyError("collections_merged.parquet missing 'Collection_Time'")

    df = df[df["Collection_Time"].notna()].copy()
    df["Collection_Time"] = pd.to_datetime(df["Collection_Time"], errors="coerce")
    df = df[df["Collection_Time"].notna()].copy()

    df["date"] = pd.to_datetime(df["Collection_Time"].dt.date)

    # Ensure some flags exist even if missing upstream
    if "Is_Alert" not in df.columns:
        df["Is_Alert"] = False
    if "Stream_Type" not in df.columns:
        df["Stream_Type"] = "Unknown"

    # -------------------------
    # 1) Collections per day (overall) 
    # -------------------------
    daily_all = (
        df.groupby("date")
        .size()
        .rename("collections")
        .reset_index()
        .sort_values("date")
    )
    daily_all.to_csv(out_dir / "daily_collections_all.csv", index=False)

    daily_all["roll7"] = daily_all["collections"].rolling(7).mean()
    daily_all["roll30"] = daily_all["collections"].rolling(30).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_all["date"], daily_all["collections"], alpha=0.3, label="Daily")
    plt.plot(daily_all["date"], daily_all["roll7"], label="7-day avg")
    plt.plot(daily_all["date"], daily_all["roll30"], label="30-day avg")

    plt.title("Collections per day")
    plt.xlabel("Date")
    plt.ylabel("Collections")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "collections_per_day.png", dpi=200)
    plt.close()
    # -------------------------
    # 2) Collections per day by Stream
    # -------------------------
    daily_stream = (
        df.groupby(["date", "Stream_Type"])
        .size()
        .rename("collections")
        .reset_index()
        .sort_values(["Stream_Type", "date"])
    )
    daily_stream.to_csv(out_dir / "daily_collections_by_stream.csv", index=False)

    for stream, sub in daily_stream.groupby("Stream_Type"):
        sub = sub.sort_values("date").reset_index(drop=True)

        # Rolling averages
        sub["roll7"] = sub["collections"].rolling(7).mean()
        sub["roll30"] = sub["collections"].rolling(30).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(sub["date"], sub["collections"], alpha=0.3, label="Daily")
        plt.plot(sub["date"], sub["roll7"], label="7-day avg")
        plt.plot(sub["date"], sub["roll30"], label="30-day avg")

        plt.title(f"Collections per day — {stream}")
        plt.xlabel("Date")
        plt.ylabel("Collections")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"collections_per_day_{safe_name(stream)}.png", dpi=200)
        plt.close()

    # -------------------------
    # 3) Alert rate patterns
    # -------------------------
    # Alert rate by day-of-week
    df["day_of_week"] = df["Collection_Time"].dt.day_name()
    alert_by_dow = (
        df.groupby("day_of_week")["Is_Alert"]
        .mean()
        .rename("alert_rate")
        .reset_index()
    )

    # Sort days in calendar order
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    alert_by_dow["day_of_week"] = pd.Categorical(alert_by_dow["day_of_week"], categories=dow_order, ordered=True)
    alert_by_dow = alert_by_dow.sort_values("day_of_week")
    alert_by_dow.to_csv(out_dir / "alert_rate_by_day_of_week.csv", index=False)

    save_bar_plot(
        alert_by_dow["day_of_week"].astype(str),
        alert_by_dow["alert_rate"],
        title="Alert rate by day of week",
        xlabel="Day of week",
        ylabel="Alert rate",
        outpath=out_dir / "alert_rate_by_day_of_week.png",
    )

    # Alert rate over time (daily)
    daily_alert = (
        df.groupby("date")["Is_Alert"]
        .mean()
        .rename("alert_rate")
        .reset_index()
        .sort_values("date")
    )
    daily_alert.to_csv(out_dir / "daily_alert_rate.csv", index=False)

    save_line_plot(
        daily_alert["date"],
        daily_alert["alert_rate"],
        title="Daily alert rate over time",
        xlabel="Date",
        ylabel="Alert rate",
        outpath=out_dir / "daily_alert_rate.png",
    )

    # -------------------------
    # 4) Fullness distribution (if available)
    # -------------------------
    if "Fullness_Pct" in df.columns:
        fullness = df["Fullness_Pct"].dropna()
        if len(fullness) > 0:
            plt.figure()
            plt.hist(fullness, bins=30)
            plt.title("Fullness % at collection (known values only)")
            plt.xlabel("Fullness %")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / "fullness_hist.png", dpi=200)
            plt.close()

    # =========================================================
    # 5) Top alert bins (Serial) bar chart (report-ready)
    # =========================================================
    # Only meaningful if we actually have alerts
    if df["Is_Alert"].any():
        top_n = 20
        top_alert_bins = (
            df.groupby("Serial")["Is_Alert"]
            .sum()
            .rename("alert_count")
            .reset_index()
            .sort_values("alert_count", ascending=False)
            .head(top_n)
        )
        top_alert_bins.to_csv(out_dir / "top_alert_bins.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.bar(top_alert_bins["Serial"].astype(str), top_alert_bins["alert_count"])
        plt.title(f"Top {top_n} bins by alert count (Serial)")
        plt.xlabel("Serial")
        plt.ylabel("Alert count")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "top_alert_bins.png", dpi=200)
        plt.close()

    # =========================================================
    # 6 a) Monthly timeline (YYYY-MM) — best for forecasting
    # =========================================================
    
    df["month"] = df["Collection_Time"].dt.to_period("M").astype(str)

    monthly_all = (
        df.groupby("month")
        .size()
        .rename("collections")
        .reset_index()
        .sort_values("month")
    )
    monthly_all.to_csv(out_dir / "monthly_collections_all.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.bar(monthly_all["month"], monthly_all["collections"])
    plt.title("Monthly collections (all bins) — Timeline (YYYY-MM)")
    plt.xlabel("Month")
    plt.ylabel("Collections")

    # show every 3rd label to avoid overlap
    step = 3
    ticks = list(range(0, len(monthly_all), step))
    plt.xticks(
        ticks=ticks,
        labels=monthly_all["month"].iloc[ticks],
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.savefig(out_dir / "monthly_collections_all_timeline.png", dpi=200)
    plt.close()

    # -------------------------
    # 6 b) Seasonality by calendar month (Jan..Dec) 
    # -------------------------
    df["month_num"] = df["Collection_Time"].dt.month
    df["month_name"] = df["Collection_Time"].dt.month_name()

    seasonality_all = (
        df.groupby(["month_num", "month_name"])
        .size()
        .rename("collections")
        .reset_index()
        .sort_values("month_num")
    )
    seasonality_all.to_csv(out_dir / "seasonality_by_month_all.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(seasonality_all["month_name"], seasonality_all["collections"])
    plt.title("Seasonality: Collections by Calendar Month (All Years)")
    plt.xlabel("Month")
    plt.ylabel("Collections")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "seasonality_by_month_all.png", dpi=200)
    plt.close()

    # -------------------------
    # 6c) Monthly timeline by Stream (YYYY-MM) — readable x-axis
    # -------------------------
    monthly_stream = (
        df.groupby(["month", "Stream_Type"])
        .size()
        .rename("collections")
        .reset_index()
        .sort_values(["Stream_Type", "month"])
    )
    monthly_stream.to_csv(out_dir / "monthly_collections_by_stream.csv", index=False)

    for stream, sub in monthly_stream.groupby("Stream_Type"):
        sub = sub.sort_values("month").reset_index(drop=True)

        plt.figure(figsize=(12, 5))
        plt.bar(sub["month"], sub["collections"])
        plt.title(f"Monthly collections — {stream} (Timeline YYYY-MM)")
        plt.xlabel("Month")
        plt.ylabel("Collections")

        step = 3
        ticks = list(range(0, len(sub), step))
        plt.xticks(
            ticks=ticks,
            labels=sub["month"].iloc[ticks],
            rotation=45,
            ha="right"
        )

        plt.tight_layout()
        plt.savefig(out_dir / f"monthly_collections_{safe_name(stream)}_timeline.png", dpi=200)
        plt.close()

    # -------------------------
    # 7) QA summary
    # -------------------------
    qa = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "streams": sorted(df["Stream_Type"].dropna().unique().tolist()),
        "pct_alert": float(df["Is_Alert"].mean()) if "Is_Alert" in df.columns else None,
        "pct_fullness_known": float(df["Fullness_Pct"].notna().mean()) if "Fullness_Pct" in df.columns else None,
    }

    if "Lat" in df.columns and "Lng" in df.columns:
        qa["pct_has_location"] = float((df["Lat"].notna() & df["Lng"].notna()).mean())
    else:
        qa["pct_has_location"] = None

    (out_dir / "qa_summary.json").write_text(json.dumps(qa, indent=2))

    print(f"[OK] EDA outputs saved to: {out_dir}")
    print("Generated:")
    print("  - collections_per_day.png")
    print("  - collections_per_day_<stream>.png")
    print("  - alert_rate_by_day_of_week.png")
    print("  - daily_alert_rate.png")
    print("  - fullness_hist.png (if Fullness_Pct exists)")
    print("  - top_alert_bins.png (if any alerts exist)")
    print("  - monthly_collections_all_timeline.png")
    print("  - seasonality_by_month_all.png")
    print("  - monthly_collections_<stream>_timeline.png")


if __name__ == "__main__":
    main()