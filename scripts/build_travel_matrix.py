from __future__ import annotations

"""
build_travel_matrix.py
-------------------------------------------------------------
Purpose
-------
Build a depot-to-stop and stop-to-stop travel matrix for the Bigbelly routing
layer using an OSMnx network instead of straight-line distance.

This version is designed for the campus Bigbelly project where the collection
vehicle has the footprint of a small car, so routing should occur on a
vehicle-feasible network rather than on arbitrary pedestrian cut-throughs.

Default network choice
----------------------
By default, this script requests a "drive_service" network from OSMnx.
That is usually a better starting point than "walk" for a small service vehicle.

If needed later, you can:
- switch to a different network type, or
- provide a custom Overpass filter.

Pipeline fit
------------
1. build_projected_fill.py
2. solve_7day_schedule.py
3. build_travel_matrix.py   <-- this file
4. solve_daily_routing.py

Inputs
------
Expected input file:
- data/processed/bin_7day_projection_inputs.parquet
or
- data/processed/bin_7day_projection_inputs.csv

Optional schedule file:
- data/processed/small_instance_service_schedule.csv
If available, the matrix is restricted to bins currently appearing in the
Phase 1 schedule. Otherwise, it includes all bins with valid access coordinates.

Preferred columns in the input
------------------------------
Required:
- Serial
- Lat / Lng   OR   Access_Lat / Access_Lng

Optional:
- Description
- Stop_ID
- service_walk_min

Outputs
-------
- data/processed/routing_nodes.csv
- data/processed/bin_stop_lookup.csv
- data/processed/routing_snap_lookup.csv
- data/processed/travel_matrix_long.csv
- data/processed/travel_matrix_wide.csv
- data/processed/osmnx_graph.graphml
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


# -------------------------------------------------------------
# Helper: repo root
# -------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# -------------------------------------------------------------
# Helper: ensure output folder exists
# -------------------------------------------------------------
def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


# -------------------------------------------------------------
# Haversine distance in meters
# -------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return r * c


# -------------------------------------------------------------
# Build query center and radius
# -------------------------------------------------------------
def query_center_and_radius(points_df: pd.DataFrame, margin_m: float) -> tuple[tuple[float, float], float]:
    """
    Compute a center point and search radius for graph download.

    The center is the mean latitude/longitude of all points.
    The radius is the maximum haversine distance from center to any point,
    plus a user-specified margin.
    """
    center_lat = float(points_df["Lat"].mean())
    center_lng = float(points_df["Lng"].mean())

    max_dist = 0.0
    for row in points_df.itertuples(index=False):
        d = haversine_m(center_lat, center_lng, float(row.Lat), float(row.Lng))
        if d > max_dist:
            max_dist = d

    return (center_lat, center_lng), float(max_dist + margin_m)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build stop-level travel matrix with OSMnx.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Projection input parquet/csv with Lat/Lng or Access_Lat/Access_Lng",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="Optional Phase 1 service schedule CSV used to restrict the matrix to active bins",
    )
    parser.add_argument(
        "--depot-lat",
        type=float,
        required=True,
        help="Depot latitude",
    )
    parser.add_argument(
        "--depot-lng",
        type=float,
        required=True,
        help="Depot longitude",
    )
    parser.add_argument(
        "--depot-label",
        type=str,
        default="DEPOT",
        help="Human-readable depot label",
    )
    parser.add_argument(
        "--network-type",
        type=str,
        default="drive_service",
        help="OSMnx network type, e.g. drive, drive_service, walk, bike, all",
    )
    parser.add_argument(
        "--custom-filter",
        type=str,
        default=None,
        help="Optional custom Overpass filter passed to OSMnx",
    )
    parser.add_argument(
        "--routing-direction",
        type=str,
        default="undirected",
        choices=["directed", "undirected"],
        help="Whether shortest paths should use the directed graph or an undirected approximation",
    )
    parser.add_argument(
        "--travel-speed-mph",
        type=float,
        default=7.0,
        help="Average vehicle speed used to convert network length to travel time",
    )
    parser.add_argument(
        "--query-margin-m",
        type=float,
        default=500.0,
        help="Extra buffer around the points when downloading the graph",
    )
    parser.add_argument(
        "--retain-all",
        action="store_true",
        help="If set, keep all graph components returned by OSMnx",
    )
    parser.add_argument(
        "--save-graphml",
        action="store_true",
        help="If set, save the downloaded graph as GraphML",
    )
    args = parser.parse_args()

    if args.travel_speed_mph <= 0:
        raise ValueError("--travel-speed-mph must be positive")

    root = repo_root()
    paths = ensure_dirs(root)

    # ---------------------------------------------------------
    # Resolve input file
    # ---------------------------------------------------------
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

    df = pd.read_csv(input_fp) if input_fp.suffix.lower() == ".csv" else pd.read_parquet(input_fp)
    df = df.copy()

    # ---------------------------------------------------------
    # Choose coordinate columns
    # ---------------------------------------------------------
    lat_col = "Access_Lat" if "Access_Lat" in df.columns else "Lat"
    lng_col = "Access_Lng" if "Access_Lng" in df.columns else "Lng"

    needed = ["Serial", lat_col, lng_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required coordinate columns: {missing}")

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lng_col] = pd.to_numeric(df[lng_col], errors="coerce")

    if "Description" not in df.columns:
        df["Description"] = df["Serial"]

    if "Stop_ID" not in df.columns:
        # Fallback: each bin is its own stop
        df["Stop_ID"] = df["Serial"]

    if "service_walk_min" not in df.columns:
        df["service_walk_min"] = 0.0

    df["Stop_ID"] = df["Stop_ID"].astype(str).str.strip()
    df["service_walk_min"] = pd.to_numeric(df["service_walk_min"], errors="coerce").fillna(0.0)

    # ---------------------------------------------------------
    # Optional filter: keep only bins in current schedule
    # ---------------------------------------------------------
    if args.schedule:
        sched_fp = Path(args.schedule)
    else:
        sched_fp = paths["processed"] / "small_instance_service_schedule.csv"

    if args.schedule and sched_fp.exists():
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

    df = df.dropna(subset=[lat_col, lng_col]).drop_duplicates(subset=["Serial"]).copy()
    if df.empty:
        raise ValueError("No bins with valid coordinates were found.")

    # ---------------------------------------------------------
    # Save bin-to-stop lookup
    # ---------------------------------------------------------
    bin_stop_lookup = df[
        ["Serial", "Stop_ID", "Description", "service_walk_min", lat_col, lng_col]
    ].copy()
    bin_stop_lookup = bin_stop_lookup.rename(
        columns={
            lat_col: "Access_Lat",
            lng_col: "Access_Lng",
        }
    )

    # ---------------------------------------------------------
    # Build stop-level routing nodes
    # ---------------------------------------------------------
    stop_df = (
        df.groupby("Stop_ID", as_index=False)
        .agg(
            Lat=(lat_col, "first"),
            Lng=(lng_col, "first"),
            label=("Description", "first"),
            service_walk_min=("service_walk_min", "max"),
        )
        .copy()
    )

    nodes = [
        {
            "node_id": 0,
            "node_type": "depot",
            "Stop_ID": None,
            "Serial": None,
            "label": args.depot_label,
            "Lat": float(args.depot_lat),
            "Lng": float(args.depot_lng),
            "service_walk_min": 0.0,
        }
    ]

    for idx, row in enumerate(stop_df.itertuples(index=False), start=1):
        nodes.append(
            {
                "node_id": idx,
                "node_type": "stop",
                "Stop_ID": str(row.Stop_ID),
                "Serial": None,
                "label": str(row.label),
                "Lat": float(row.Lat),
                "Lng": float(row.Lng),
                "service_walk_min": float(row.service_walk_min),
            }
        )

    nodes_df = pd.DataFrame(nodes)

    # ---------------------------------------------------------
    # Build graph query footprint
    # ---------------------------------------------------------
    query_points = nodes_df[["Lat", "Lng"]].copy()
    center_point, query_dist_m = query_center_and_radius(query_points, margin_m=args.query_margin_m)

    # ---------------------------------------------------------
    # Download OSMnx graph
    # ---------------------------------------------------------
    graph_kwargs = {
        "center_point": center_point,
        "dist": query_dist_m,
        "dist_type": "bbox",
        "network_type": args.network_type,
        "simplify": True,
        "retain_all": args.retain_all,
        "truncate_by_edge": True,
    }
    if args.custom_filter:
        graph_kwargs["custom_filter"] = args.custom_filter

    G = ox.graph_from_point(**graph_kwargs)

    if len(G.nodes) == 0:
        raise ValueError("OSMnx returned an empty graph for the requested area.")

    # Project to a metric CRS for nearest-node search and path lengths.
    G_proj = ox.project_graph(G)

    if args.routing_direction == "undirected":
        G_paths = ox.convert.to_undirected(G_proj)
    else:
        G_paths = G_proj

    if args.save_graphml:
        graph_fp = paths["processed"] / "osmnx_graph.graphml"
        ox.save_graphml(G_proj, graph_fp)

    # ---------------------------------------------------------
    # Snap depot and stops to the graph
    # ---------------------------------------------------------
    points_gdf = gpd.GeoDataFrame(
        nodes_df.copy(),
        geometry=[Point(xy) for xy in zip(nodes_df["Lng"], nodes_df["Lat"])],
        crs="EPSG:4326",
    )

    graph_crs = G_proj.graph["crs"]
    points_proj = points_gdf.to_crs(graph_crs)

    nearest_node_ids, snap_dist_m = ox.distance.nearest_nodes(
        G_proj,
        X=points_proj.geometry.x.to_numpy(),
        Y=points_proj.geometry.y.to_numpy(),
        return_dist=True,
    )

    snap_lookup = points_gdf.copy()
    snap_lookup["graph_node"] = np.asarray(nearest_node_ids)
    snap_lookup["snap_dist_m"] = np.asarray(snap_dist_m, dtype=float)

    # ---------------------------------------------------------
    # Build shortest-path matrix on the graph
    # ---------------------------------------------------------
    meters_per_min = args.travel_speed_mph * 1609.344 / 60.0

    node_id_to_graph_node = dict(zip(snap_lookup["node_id"], snap_lookup["graph_node"]))

    rows = []
    unreachable_pairs = []

    ordered_node_ids = nodes_df["node_id"].astype(int).tolist()

    for from_node_id in ordered_node_ids:
        source_graph_node = node_id_to_graph_node[from_node_id]

        lengths = nx.single_source_dijkstra_path_length(
            G_paths,
            source_graph_node,
            weight="length",
        )

        from_label = str(nodes_df.loc[nodes_df["node_id"] == from_node_id, "label"].iloc[0])

        for to_node_id in ordered_node_ids:
            target_graph_node = node_id_to_graph_node[to_node_id]
            to_label = str(nodes_df.loc[nodes_df["node_id"] == to_node_id, "label"].iloc[0])

            if source_graph_node == target_graph_node:
                dist_m = 0.0
            else:
                dist_m = lengths.get(target_graph_node, np.nan)

            if pd.isna(dist_m):
                unreachable_pairs.append((from_node_id, to_node_id))
                travel_min = np.nan
            else:
                travel_min = float(dist_m) / meters_per_min

            rows.append(
                {
                    "from_node": int(from_node_id),
                    "to_node": int(to_node_id),
                    "from_label": from_label,
                    "to_label": to_label,
                    "distance_m": None if pd.isna(dist_m) else round(float(dist_m), 2),
                    "travel_min": None if pd.isna(travel_min) else round(float(travel_min), 2),
                }
            )

    if unreachable_pairs:
        sample = unreachable_pairs[:10]
        raise ValueError(
            "Some origin-destination pairs are unreachable on the downloaded OSMnx graph. "
            f"Sample unreachable pairs: {sample}. "
            "Try rerunning with --routing-direction undirected, a larger --query-margin-m, "
            "or a different --network-type/custom-filter."
        )

    long_df = pd.DataFrame(rows).sort_values(["from_node", "to_node"]).reset_index(drop=True)
    wide_df = long_df.pivot(index="from_node", columns="to_node", values="travel_min").sort_index()
    wide_df = wide_df.sort_index(axis=1)

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------
    nodes_fp = paths["processed"] / "routing_nodes.csv"
    lookup_fp = paths["processed"] / "bin_stop_lookup.csv"
    snap_fp = paths["processed"] / "routing_snap_lookup.csv"
    long_fp = paths["processed"] / "travel_matrix_long.csv"
    wide_fp = paths["processed"] / "travel_matrix_wide.csv"

    nodes_df.to_csv(nodes_fp, index=False)
    bin_stop_lookup.to_csv(lookup_fp, index=False)
    snap_lookup.drop(columns="geometry").to_csv(snap_fp, index=False)
    long_df.to_csv(long_fp, index=False)
    wide_df.to_csv(wide_fp)

    print(f"[OK] Wrote: {nodes_fp}")
    print(f"[OK] Wrote: {lookup_fp}")
    print(f"[OK] Wrote: {snap_fp}")
    print(f"[OK] Wrote: {long_fp}")
    print(f"[OK] Wrote: {wide_fp}")
    if args.save_graphml:
        print(f"[OK] Wrote: {paths['processed'] / 'osmnx_graph.graphml'}")

    print(f"[INFO] bins included = {len(bin_stop_lookup):,}")
    print(f"[INFO] stop nodes included (excluding depot) = {len(nodes_df) - 1:,}")
    print(f"[INFO] graph nodes = {len(G_proj.nodes):,}")
    print(f"[INFO] graph edges = {len(G_proj.edges):,}")
    print(f"[INFO] center_point = ({center_point[0]:.6f}, {center_point[1]:.6f})")
    print(f"[INFO] query_dist_m = {query_dist_m:.1f}")
    print(f"[INFO] routing_direction = {args.routing_direction}")
    print(f"[INFO] network_type = {args.network_type}")


if __name__ == "__main__":
    main()
