import argparse
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd


# -----------------------------
# Simple helper: haversine distance
# -----------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(
        radians,
        [float(lat1), float(lon1), float(lat2), float(lon2)],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def nearest_neighbor_route_minutes(
    stops,
    depot_lat,
    depot_lon,
    speed_mph=6.0,
    service_min=4.0,
):
    """
    Fallback route-time estimator.

    This is not full OR-Tools routing.
    It gives a defensible historical proxy using nearest-neighbor routing:
    depot -> stops -> depot.

    Historical route minutes should be described as a proxy, not observed driver time.
    """
    if len(stops) == 0:
        return 0.0

    unvisited = stops.copy()
    current_lat, current_lon = depot_lat, depot_lon
    total_miles = 0.0

    while unvisited:
        distances = [
            haversine_miles(current_lat, current_lon, stop["lat"], stop["lon"])
            for stop in unvisited
        ]

        idx = distances.index(min(distances))
        next_stop = unvisited.pop(idx)

        total_miles += haversine_miles(
            current_lat,
            current_lon,
            next_stop["lat"],
            next_stop["lon"],
        )

        current_lat, current_lon = next_stop["lat"], next_stop["lon"]

    total_miles += haversine_miles(
        current_lat,
        current_lon,
        depot_lat,
        depot_lon,
    )

    travel_minutes = (total_miles / speed_mph) * 60.0
    service_minutes = len(stops) * service_min

    return travel_minutes + service_minutes


def normalize_stream(x):
    x = str(x).strip().lower()

    if "compost" in x:
        return "Compostables"

    if "bottle" in x or "can" in x or "recycl" in x:
        return "Bottles/Cans"

    if "waste" in x or "landfill" in x or "trash" in x:
        return "Waste"

    return "Unknown"


def density_for_stream(stream):
    if stream == "Waste":
        return 1.0

    if stream == "Compostables":
        return 1.2

    if stream == "Bottles/Cans":
        return 0.3

    return 1.0


def find_column(df, exact_candidates=None, contains_candidates=None):
    """
    Flexible column finder for Bigbelly exports.
    """
    exact_candidates = exact_candidates or []
    contains_candidates = contains_candidates or []

    lower_to_col = {c.lower().strip(): c for c in df.columns}

    for cand in exact_candidates:
        key = cand.lower().strip()
        if key in lower_to_col:
            return lower_to_col[key]

    for c in df.columns:
        lc = c.lower().strip()
        for token in contains_candidates:
            if token.lower().strip() in lc:
                return c

    return None


def read_bigbelly_csv(path):
    """
    Bigbelly exports often have metadata rows before the actual header.
    Try skiprows=10 first, then fallback to normal CSV.
    """
    path = Path(path)

    try:
        df = pd.read_csv(path, skiprows=10)
        if len(df.columns) > 1:
            return df
    except Exception:
        pass

    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        description="Build historical baseline comparison for Bigbelly collection records."
    )

    parser.add_argument("--collections", required=True)
    parser.add_argument("--assets", required=True)

    parser.add_argument(
        "--output",
        default="data/processed/historical_routed_baseline.csv",
    )

    parser.add_argument(
        "--dates",
        nargs="*",
        default=[
            "2026-02-04",
            "2026-02-05",
            "2026-02-06",
            "2026-02-09",
            "2026-02-10",
        ],
        help="Historical dates to compare against the model.",
    )

    parser.add_argument("--depot-lat", type=float, default=37.868998)
    parser.add_argument("--depot-lon", type=float, default=-122.264800)

    parser.add_argument(
        "--speed-mph",
        type=float,
        default=6.0,
        help="Assumed vehicle speed for historical route proxy.",
    )

    parser.add_argument(
        "--service-min",
        type=float,
        default=4.0,
        help="Service time per historical pickup.",
    )

    parser.add_argument(
        "--dump-min",
        type=float,
        default=17.0,
        help="Dump turnaround minutes added per active stream route.",
    )

    parser.add_argument(
        "--truck-work-min",
        type=float,
        default=480.0,
        help="Truck work minutes for overtime proxy. Use 480 to match final model.",
    )

    args = parser.parse_args()

    collections_path = Path(args.collections)
    assets_path = Path(args.assets)

    if not collections_path.exists():
        raise FileNotFoundError(f"Missing collections file: {collections_path}")

    if not assets_path.exists():
        raise FileNotFoundError(f"Missing assets file: {assets_path}")

    df = read_bigbelly_csv(collections_path)
    assets = read_bigbelly_csv(assets_path)

    print("[INFO] Collections columns:", list(df.columns))
    print("[INFO] Assets columns:", list(assets.columns))

    # -----------------------------
    # Identify likely collection columns
    # -----------------------------
    serial_col = find_column(
        df,
        exact_candidates=[
            "Serial",
            "Serial Number",
            "Asset Serial",
            "Bin Serial",
        ],
        contains_candidates=[
            "serial",
        ],
    )

    stream_col = find_column(
        df,
        contains_candidates=[
            "stream",
            "waste type",
            "material",
        ],
    )

    reason_col = find_column(
        df,
        contains_candidates=[
            "reason",
        ],
    )

    fill_col = find_column(
        df,
        contains_candidates=[
            "fill",
            "fullness",
        ],
    )

    time_col = find_column(
        df,
        contains_candidates=[
            "collection date",
            "collection time",
            "timestamp",
            "date",
            "time",
        ],
    )

    if serial_col is None:
        raise ValueError("Could not identify serial/bin column in collections file.")

    if stream_col is None:
        raise ValueError("Could not identify stream column in collections file.")

    if reason_col is None:
        raise ValueError("Could not identify reason column in collections file.")

    if fill_col is None:
        raise ValueError("Could not identify fill/fullness column in collections file.")

    if time_col is None:
        raise ValueError("Could not identify date/time column in collections file.")

    print("[INFO] Using collection columns:")
    print("  serial:", serial_col)
    print("  stream:", stream_col)
    print("  reason:", reason_col)
    print("  fill:", fill_col)
    print("  time:", time_col)

    # -----------------------------
    # Parse timestamps and filter dates
    # -----------------------------
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()

    df["local_date"] = df[time_col].dt.date.astype(str)

    df5 = df[df["local_date"].isin(args.dates)].copy()

    if df5.empty:
        raise ValueError(
            "No collection rows found for selected dates. "
            "Check --dates and the timestamp column."
        )

    # -----------------------------
    # Normalize stream and estimate quantity
    # -----------------------------
    df5["stream_clean"] = df5[stream_col].apply(normalize_stream)

    df5["fill_pct_num"] = (
        df5[fill_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    # Assumption: 150-gallon standard Bigbelly capacity
    df5["pickup_gal_est"] = 150.0 * df5["fill_pct_num"].fillna(0) / 100.0
    df5["density"] = df5["stream_clean"].apply(density_for_stream)
    df5["pickup_lb_est"] = df5["pickup_gal_est"] * df5["density"]

    # Historical low-value proxies
    df5["is_not_ready"] = (
        df5[reason_col]
        .astype(str)
        .str.lower()
        .str.contains("not ready", na=False)
    )

    df5["is_age"] = (
        df5[reason_col]
        .astype(str)
        .str.lower()
        .str.contains("age", na=False)
    )

    # Low fullness, excluding age-based service
    df5["low_value_proxy"] = (
        (df5["fill_pct_num"].fillna(0) < 60)
        & (~df5["is_age"])
    )

    # -----------------------------
    # Join with assets to get lat/lon
    # -----------------------------
    asset_serial_col = find_column(
        assets,
        exact_candidates=[
            "Serial",
            "Serial Number",
            "Asset Serial",
            "Bin Serial",
        ],
        contains_candidates=[
            "serial",
        ],
    )

    lat_col = find_column(
        assets,
        contains_candidates=[
            "lat",
            "latitude",
        ],
    )

    lon_col = find_column(
        assets,
        contains_candidates=[
            "lon",
            "lng",
            "long",
            "longitude",
        ],
    )

    if asset_serial_col is None or lat_col is None or lon_col is None:
        print("[WARN] Could not identify asset serial/lat/lon columns.")
        print("[WARN] Route minutes will not be calculated from locations.")

        merged = df5.copy()
        merged["lat"] = pd.NA
        merged["lon"] = pd.NA

    else:
        print("[INFO] Using asset columns:")
        print("  asset serial:", asset_serial_col)
        print("  latitude:", lat_col)
        print("  longitude:", lon_col)

        assets_small = assets[[asset_serial_col, lat_col, lon_col]].copy()
        assets_small[asset_serial_col] = (
            assets_small[asset_serial_col]
            .astype(str)
            .str.strip()
        )

        df5[serial_col] = df5[serial_col].astype(str).str.strip()

        merged = df5.merge(
            assets_small,
            left_on=serial_col,
            right_on=asset_serial_col,
            how="left",
        )

        merged = merged.rename(columns={lat_col: "lat", lon_col: "lon"})

    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")

    routed = merged.dropna(subset=["lat", "lon"]).copy()

    daily_rows = []

    for date, gday in merged.groupby("local_date"):
        day_pickups = len(gday)
        day_gal = gday["pickup_gal_est"].sum()
        day_lb = gday["pickup_lb_est"].sum()
        day_not_ready = int(gday["is_not_ready"].sum())
        day_low_value = int(gday["low_value_proxy"].sum())

        route_minutes_total = 0.0
        route_rows = 0

        gday_routed = routed[routed["local_date"] == date].copy()

        for stream, gs in gday_routed.groupby("stream_clean"):
            if stream == "Unknown":
                continue

            stops = [
                {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                }
                for _, row in gs.iterrows()
            ]

            if len(stops) == 0:
                continue

            route_min = nearest_neighbor_route_minutes(
                stops,
                args.depot_lat,
                args.depot_lon,
                speed_mph=args.speed_mph,
                service_min=args.service_min,
            )

            # Add at least one dump turnaround per active stream route.
            route_min += args.dump_min

            route_minutes_total += route_min
            route_rows += 1

        overtime_proxy = max(0.0, route_minutes_total - 3 * args.truck_work_min)

        daily_rows.append(
            {
                "date": date,
                "historical_pickups": int(day_pickups),
                "pickup_gal_est": round(day_gal, 2),
                "pickup_lb_est": round(day_lb, 2),
                "not_ready_pickups": int(day_not_ready),
                "low_value_proxy_pickups": int(day_low_value),
                "historical_route_rows_proxy": int(route_rows),
                "historical_route_minutes_proxy": round(route_minutes_total, 2),
                "historical_overtime_proxy": round(overtime_proxy, 2),
            }
        )

    summary = pd.DataFrame(daily_rows).sort_values("date")

    total = {
        "date": "TOTAL",
        "historical_pickups": int(summary["historical_pickups"].sum()),
        "pickup_gal_est": round(summary["pickup_gal_est"].sum(), 2),
        "pickup_lb_est": round(summary["pickup_lb_est"].sum(), 2),
        "not_ready_pickups": int(summary["not_ready_pickups"].sum()),
        "low_value_proxy_pickups": int(summary["low_value_proxy_pickups"].sum()),
        "historical_route_rows_proxy": int(summary["historical_route_rows_proxy"].sum()),
        "historical_route_minutes_proxy": round(
            summary["historical_route_minutes_proxy"].sum(),
            2,
        ),
        "historical_overtime_proxy": round(
            summary["historical_overtime_proxy"].sum(),
            2,
        ),
    }

    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("[OK] Wrote:", output_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()