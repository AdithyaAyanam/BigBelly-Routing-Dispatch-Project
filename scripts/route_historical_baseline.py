import argparse
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# -----------------------------
# Simple helper: haversine distance
# -----------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def nearest_neighbor_route_minutes(stops, depot_lat, depot_lon, speed_mph=6.0, service_min=4.0):
    """
    Fallback route-time estimator.
    This is not full OR-Tools, but gives a defensible baseline if OR-Tools integration is difficult.
    It routes stops using nearest neighbor and returns depot-to-stops-to-depot minutes.
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
            current_lat, current_lon, next_stop["lat"], next_stop["lon"]
        )
        current_lat, current_lon = next_stop["lat"], next_stop["lon"]

    total_miles += haversine_miles(current_lat, current_lon, depot_lat, depot_lon)

    travel_minutes = (total_miles / speed_mph) * 60
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", default="data/processed/historical_routed_baseline.csv")
    parser.add_argument("--start-date", default="2026-02-04")
    parser.add_argument("--end-date", default="2026-02-11")
    parser.add_argument("--dates", nargs="*", default=["2026-02-04", "2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10"])
    parser.add_argument("--depot-lat", type=float, default=37.8699)
    parser.add_argument("--depot-lon", type=float, default=-122.2678)
    parser.add_argument("--speed-mph", type=float, default=6.0)
    parser.add_argument("--service-min", type=float, default=4.0)
    parser.add_argument("--dump-min", type=float, default=17.0)
    parser.add_argument("--truck-work-min", type=float, default=750.0)
    args = parser.parse_args()

    collections_path = Path(args.collections)
    assets_path = Path(args.assets)

    df = pd.read_csv(collections_path, skiprows=10)
    assets = pd.read_csv(assets_path, skiprows=10)

    print("[INFO] Collections columns:", list(df.columns))
    print("[INFO] Assets columns:", list(assets.columns))

    # -----------------------------
    # Identify likely columns
    # Adjust here if your column names differ
    # -----------------------------
    serial_col = None
    for c in df.columns:
        if c.lower() in ["serial", "serial number", "asset serial", "bin serial"]:
            serial_col = c
            break

    stream_col = None
    for c in df.columns:
        if "stream" in c.lower() or "waste type" in c.lower() or "material" in c.lower():
            stream_col = c
            break

    reason_col = None
    for c in df.columns:
        if "reason" in c.lower():
            reason_col = c
            break

    fill_col = None
    for c in df.columns:
        if "fill" in c.lower() or "fullness" in c.lower():
            fill_col = c
            break

    time_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower() or "timestamp" in c.lower():
            time_col = c
            break

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

    print("[INFO] Using columns:")
    print("  serial:", serial_col)
    print("  stream:", stream_col)
    print("  reason:", reason_col)
    print("  fill:", fill_col)
    print("  time:", time_col)

    # Parse timestamps
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["local_date"] = df[time_col].dt.date.astype(str)
    c
    # Filter to the five report dates
    df5 = df[df["local_date"].isin(args.dates)].copy()

    # Normalize stream and fill
    df5["stream_clean"] = df5[stream_col].apply(normalize_stream)
    df5["fill_pct_num"] = (
        df5[fill_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    # Estimate pickup gallons and pounds
    # Assumption: 150-gallon standard Bigbelly capacity
    df5["pickup_gal_est"] = 150.0 * df5["fill_pct_num"].fillna(0) / 100.0
    df5["density"] = df5["stream_clean"].apply(density_for_stream)
    df5["pickup_lb_est"] = df5["pickup_gal_est"] * df5["density"]

    # Low-value proxy
    df5["is_not_ready"] = df5[reason_col].astype(str).str.lower().str.contains("not ready")
    df5["is_age"] = df5[reason_col].astype(str).str.lower().str.contains("age")
    df5["low_value_proxy"] = (df5["fill_pct_num"] < 60) & (~df5["is_age"])

    # -----------------------------
    # Join with assets to get lat/lon
    # -----------------------------
    asset_serial_col = None
    for c in assets.columns:
        if c.lower() in ["serial", "serial number", "asset serial", "bin serial"]:
            asset_serial_col = c
            break

    lat_col = None
    lon_col = None
    for c in assets.columns:
        lc = c.lower()
        if "lat" in lc:
            lat_col = c
        if "lon" in lc or "lng" in lc or "long" in lc:
            lon_col = c

    if asset_serial_col is None or lat_col is None or lon_col is None:
        print("[WARN] Could not identify asset serial/lat/lon columns.")
        print("[WARN] Route minutes will not be calculated from locations.")
        merged = df5.copy()
        merged["lat"] = None
        merged["lon"] = None
    else:
        merged = df5.merge(
            assets[[asset_serial_col, lat_col, lon_col]],
            left_on=serial_col,
            right_on=asset_serial_col,
            how="left"
        )
        merged = merged.rename(columns={lat_col: "lat", lon_col: "lon"})

    # Drop rows without coordinates for routing estimate
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

        # Route by stream using nearest neighbor fallback
        gday_routed = routed[routed["local_date"] == date]

        for stream, gs in gday_routed.groupby("stream_clean"):
            stops = [
                {"lat": float(row["lat"]), "lon": float(row["lon"])}
                for _, row in gs.iterrows()
            ]

            if len(stops) == 0:
                continue

            route_min = nearest_neighbor_route_minutes(
                stops,
                args.depot_lat,
                args.depot_lon,
                speed_mph=args.speed_mph,
                service_min=args.service_min
            )

            # Add at least one dump turnaround per active stream route
            route_min += args.dump_min

            route_minutes_total += route_min
            route_rows += 1

        overtime_proxy = max(0.0, route_minutes_total - 3 * args.truck_work_min)

        daily_rows.append({
            "date": date,
            "historical_pickups": day_pickups,
            "pickup_gal_est": round(day_gal, 2),
            "pickup_lb_est": round(day_lb, 2),
            "not_ready_pickups": day_not_ready,
            "low_value_proxy_pickups": day_low_value,
            "historical_route_rows_proxy": route_rows,
            "historical_route_minutes_proxy": round(route_minutes_total, 2),
            "historical_overtime_proxy": round(overtime_proxy, 2),
        })

    summary = pd.DataFrame(daily_rows).sort_values("date")

    total = {
        "date": "TOTAL",
        "historical_pickups": summary["historical_pickups"].sum(),
        "pickup_gal_est": round(summary["pickup_gal_est"].sum(), 2),
        "pickup_lb_est": round(summary["pickup_lb_est"].sum(), 2),
        "not_ready_pickups": summary["not_ready_pickups"].sum(),
        "low_value_proxy_pickups": summary["low_value_proxy_pickups"].sum(),
        "historical_route_rows_proxy": summary["historical_route_rows_proxy"].sum(),
        "historical_route_minutes_proxy": round(summary["historical_route_minutes_proxy"].sum(), 2),
        "historical_overtime_proxy": round(summary["historical_overtime_proxy"].sum(), 2),
    }

    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print("[OK] Wrote:", output_path)
    print(summary)


if __name__ == "__main__":
    main()