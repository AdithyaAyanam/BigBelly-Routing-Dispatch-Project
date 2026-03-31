
from __future__ import annotations

"""
build_travel_matrix.py
-------------------------------------------------------------
Purpose
-------
Build a depot-to-bin and bin-to-bin travel matrix for the Bigbelly routing layer.

This script is designed to plug into the existing Phase 1 pipeline:
- build_projected_fill.py creates projected-fill inputs
- solve_7day_schedule.py creates a service schedule
- this file creates the travel matrix needed for same-day routing

Current design choice
---------------------
This version uses a simple haversine-distance matrix and converts distance to
travel time using an average cart speed. It is intentionally lightweight and is
meant to be a clean first routing input builder.

Later, if you want higher fidelity, you can swap the haversine logic for an
OSMnx/network shortest-path matrix without changing the routing file format.

Inputs
------
Expected input file with coordinates:
- data/processed/bin_7day_projection_inputs.parquet
or
- data/processed/bin_7day_projection_inputs.csv

Optional filter file:
- data/processed/small_instance_service_schedule.csv
  If provided, the matrix will only include bins that actually appear in the
  current Phase 1 schedule. Otherwise, it includes all bins in the projection
  file that have valid coordinates.

Outputs
-------
- data/processed/travel_matrix_long.csv
- data/processed/travel_matrix_wide.csv
- data/processed/routing_nodes.csv

Output schema
-------------
routing_nodes.csv
    node_id, node_type, Serial, label, Lat, Lng

travel_matrix_long.csv
    from_node, to_node, from_label, to_label, distance_m, travel_min

travel_matrix_wide.csv
    square matrix of travel minutes indexed by node_id
"""

import argparse
from pathlib import Path
from math import radians, sin, cos, sqrt, asin

import pandas as pd


# -------------------------------------------------------------
# Helper: repo root
# -------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------------------
# Helper: directories
# -------------------------------------------------------------
def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


# -------------------------------------------------------------
# Distance helper: haversine in meters
# -------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = radians(lat1)
    p2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(p1) * cos(p2) * sin(dlambda / 2.0) ** 2
    c = 2.0 * asin(sqrt(a))
    return r * c


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build travel matrix for Bigbelly routing.")
    parser.add_argument("--input", type=str, default=None, help="Projection input parquet/csv with Lat/Lng")
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="Optional service schedule CSV to restrict the matrix to currently selected bins",
    )
    parser.add_argument(
        "--depot-lat",
        type=float,
        required=True,
        help="Depot latitude (e.g., Edwards Track staging area)",
    )
    parser.add_argument(
        "--depot-lng",
        type=float,
        required=True,
        help="Depot longitude (e.g., Edwards Track staging area)",
    )
    parser.add_argument(
        "--depot-label",
        type=str,
        default="DEPOT",
        help="Label for the depot / dump location",
    )
    parser.add_argument(
        "--cart-speed-mph",
        type=float,
        default=7.0,
        help="Average travel speed in miles per hour used to convert distance to time",
    )
    parser.add_argument(
        "--distance-inflation",
        type=float,
        default=1.20,
        help="Multiplier to make straight-line distance less optimistic on campus",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    if args.input:
        input_fp = Path(args.input)
    else:
        input_fp = paths["processed"] / "bin_7day_projection_inputs.parquet"
        if not input_fp.exists():
            input_fp = paths["processed"] / "bin_7day_projection_inputs.csv"

    if not input_fp.exists():
        raise FileNotFoundError(
            "Could not find bin_7day_projection_inputs. Run build_projected_fill.py first."
        )

    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)

    needed = ["Serial", "Lat", "Lng"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required coordinate columns: {missing}")

    df = df.copy()
    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lng"] = pd.to_numeric(df["Lng"], errors="coerce")
    if "Description" not in df.columns:
        df["Description"] = df["Serial"]

    # Optional: restrict to bins appearing in a current schedule.
    if args.schedule:
        sched_fp = Path(args.schedule)
    else:
        sched_fp = paths["processed"] / "small_instance_service_schedule.csv"

    if sched_fp.exists():
        sched = pd.read_csv(sched_fp)
        if "Serial" in sched.columns:
            active_serials = (
                sched["Serial"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            if active_serials:
                df = df[df["Serial"].isin(active_serials)].copy()

    df = df.dropna(subset=["Lat", "Lng"]).drop_duplicates(subset=["Serial"]).copy()
    if df.empty:
        raise ValueError("No bins with valid coordinates were found.")

    # Build node table. Node 0 is the depot.
    nodes = [
        {
            "node_id": 0,
            "node_type": "depot",
            "Serial": None,
            "label": args.depot_label,
            "Lat": float(args.depot_lat),
            "Lng": float(args.depot_lng),
        }
    ]

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        nodes.append(
            {
                "node_id": idx,
                "node_type": "bin",
                "Serial": row.Serial,
                "label": str(row.Description),
                "Lat": float(row.Lat),
                "Lng": float(row.Lng),
            }
        )

    nodes_df = pd.DataFrame(nodes)

    # Convert speed to meters per minute.
    meters_per_min = args.cart_speed_mph * 1609.344 / 60.0
    if meters_per_min <= 0:
        raise ValueError("cart-speed-mph must be positive")

    # Long-format matrix.
    rows = []
    for a in nodes_df.itertuples(index=False):
        for b in nodes_df.itertuples(index=False):
            if a.node_id == b.node_id:
                dist_m = 0.0
            else:
                dist_m = haversine_m(a.Lat, a.Lng, b.Lat, b.Lng) * float(args.distance_inflation)
            travel_min = dist_m / meters_per_min
            rows.append(
                {
                    "from_node": int(a.node_id),
                    "to_node": int(b.node_id),
                    "from_label": a.label,
                    "to_label": b.label,
                    "distance_m": round(dist_m, 2),
                    "travel_min": round(travel_min, 2),
                }
            )

    long_df = pd.DataFrame(rows).sort_values(["from_node", "to_node"]).reset_index(drop=True)
    wide_df = long_df.pivot(index="from_node", columns="to_node", values="travel_min").sort_index()

    nodes_fp = paths["processed"] / "routing_nodes.csv"
    long_fp = paths["processed"] / "travel_matrix_long.csv"
    wide_fp = paths["processed"] / "travel_matrix_wide.csv"

    nodes_df.to_csv(nodes_fp, index=False)
    long_df.to_csv(long_fp, index=False)
    wide_df.to_csv(wide_fp)

    print(f"[OK] Wrote: {nodes_fp}")
    print(f"[OK] Wrote: {long_fp}")
    print(f"[OK] Wrote: {wide_fp}")
    print(f"[INFO] nodes in matrix (including depot) = {len(nodes_df)}")


if __name__ == "__main__":
    main()
