from __future__ import annotations

"""
solve_daily_routing.py
-------------------------------------------------------------
Purpose
-------
Build same-day routes from the Phase 1 Bigbelly schedule.

This script is the Phase 2 routing layer that follows:
- build_projected_fill.py
- solve_7day_schedule.py
- build_travel_matrix.py

Modeling approach
-----------------
For each (day, stream) pair that has scheduled pickups:
1. Read the bins scheduled for that day/stream.
2. Read which trucks are assigned to that stream on that day.
3. Map bins -> Stop_ID using bin_stop_lookup.csv.
4. Map Stop_ID -> routing node using routing_nodes.csv.
5. Aggregate bin-level pickup demand to stop-level demand.
6. If there is one active truck and capacity is not binding, solve a
   single-truck route.
7. Otherwise solve a capacitated VRP with OR-Tools.

Important note
--------------
This is a practical routing layer that uses actual route sequencing.
It is still a first practical implementation, not a perfect multi-trip VRP with
intermediate dump returns. Extra dump cycles from Phase 1 are handled
approximately by enlarging the effective truck capacity for that stream-day.
A later enhancement can model explicit dump returns inside the route.

Inputs
------
Expected files in data/processed:
- small_instance_service_schedule.csv
- small_instance_truck_streams.csv
- bin_7day_projection_inputs.parquet or .csv
- routing_nodes.csv
- bin_stop_lookup.csv
- travel_matrix_wide.csv

Outputs
-------
- daily_route_plan.csv
- daily_route_stops.csv
- daily_route_summary.csv
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires OR-Tools. Install with: pip install ortools"
    ) from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


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


def normalize_stop_id(x: object) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else None


def read_projection(paths: dict[str, Path], input_path: str | None = None) -> pd.DataFrame:
    if input_path:
        fp = Path(input_path)
    else:
        fp = paths["processed"] / "bin_7day_projection_inputs.parquet"
        if not fp.exists():
            fp = paths["processed"] / "bin_7day_projection_inputs.csv"
    if not fp.exists():
        raise FileNotFoundError("Could not find bin_7day_projection_inputs.")
    return pd.read_csv(fp) if fp.suffix.lower() == ".csv" else pd.read_parquet(fp)


def load_matrix(wide_fp: Path) -> pd.DataFrame:
    mat = pd.read_csv(wide_fp, index_col=0)
    mat.index = mat.index.astype(int)
    mat.columns = mat.columns.astype(int)
    return mat.sort_index().sort_index(axis=1)


def stop_map_from_nodes(nodes_df: pd.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in nodes_df.itertuples(index=False):
        stop_id = normalize_stop_id(getattr(row, "Stop_ID", None))
        if str(row.node_type).strip().lower() == "stop" and stop_id is not None:
            out[stop_id] = int(row.node_id)
    return out


def effective_volume_cap(stream: str, base_caps: dict[str, float]) -> float:
    return float(base_caps[stream])


def solve_ortools_vrp(
    matrix_minutes: list[list[int]],
    service_minutes: list[int],
    gallon_demands: list[int],
    pound_demands: list[int],
    vehicle_gal_caps: list[int],
    vehicle_lb_caps: list[int],
    vehicle_time_caps: list[int],
    depot_index: int = 0,
) -> dict[str, Any]:
    num_vehicles = len(vehicle_gal_caps)
    manager = pywrapcp.RoutingIndexManager(len(matrix_minutes), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix_minutes[from_node][to_node] + service_minutes[from_node])

    transit_cb_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    def gal_callback(from_index: int) -> int:
        return gallon_demands[manager.IndexToNode(from_index)]

    gal_cb_idx = routing.RegisterUnaryTransitCallback(gal_callback)
    routing.AddDimensionWithVehicleCapacity(
        gal_cb_idx,
        0,
        vehicle_gal_caps,
        True,
        "Gallons",
    )

    def lb_callback(from_index: int) -> int:
        return pound_demands[manager.IndexToNode(from_index)]

    lb_cb_idx = routing.RegisterUnaryTransitCallback(lb_callback)
    routing.AddDimensionWithVehicleCapacity(
        lb_cb_idx,
        0,
        vehicle_lb_caps,
        True,
        "Pounds",
    )

    routing.AddDimension(
        transit_cb_idx,
        0,
        max(vehicle_time_caps),
        True,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for vehicle_id, cap in enumerate(vehicle_time_caps):
        routing.solver().Add(time_dim.CumulVar(routing.End(vehicle_id)) <= cap)

    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search.time_limit.seconds = 20

    solution = routing.SolveWithParameters(search)
    if solution is None:
       return None

    routes: list[list[int]] = []
    route_minutes: list[int] = []

    for v in range(num_vehicles):
        idx = routing.Start(v)
        route_nodes = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route_nodes.append(node)
            idx = solution.Value(routing.NextVar(idx))
        route_nodes.append(manager.IndexToNode(idx))
        routes.append(route_nodes)
        end_time = solution.Value(time_dim.CumulVar(routing.End(v)))
        route_minutes.append(int(end_time))

    return {
        "routes": routes,
        "route_minutes": route_minutes,
    }

def solve_with_stop_dropping(
    stop_agg: pd.DataFrame,
    stop_to_local: dict[str, int],
    matrix_minutes_full: list[list[int]],
    vehicle_gal_caps: list[int],
    vehicle_lb_caps: list[int],
    vehicle_time_caps: list[int],
    min_stops_to_keep: int = 3,
):
    """
    Retry VRP by dropping the smallest-demand stop until feasible.

    Returns:
      result,
      kept_stop_ids,
      dropped_stop_ids,
      service_minutes,
      gallon_demands,
      pound_demands
    """
    working = stop_agg.copy()
    dropped_stop_ids: list[str] = []

    while len(working) >= min_stops_to_keep:
        kept_stop_ids = working["Stop_ID"].astype(str).tolist()
        kept_local_nodes = [0] + [stop_to_local[s] for s in kept_stop_ids]

        old_to_new = {old: new for new, old in enumerate(kept_local_nodes)}

        submatrix = []
        for old_i in kept_local_nodes:
            row = []
            for old_j in kept_local_nodes:
                row.append(matrix_minutes_full[old_i][old_j])
            submatrix.append(row)

        service_minutes = [0] * len(kept_local_nodes)
        gallon_demands = [0] * len(kept_local_nodes)
        pound_demands = [0] * len(kept_local_nodes)

        for _, r in working.iterrows():
            old_ln = stop_to_local[str(r.Stop_ID)]
            new_ln = old_to_new[old_ln]
            service_minutes[new_ln] = int(round(float(r.stop_service_min)))
            gallon_demands[new_ln] = int(round(float(r.stop_pickup_gal)))
            pound_demands[new_ln] = int(round(float(r.stop_pickup_lb)))

        result = solve_ortools_vrp(
            matrix_minutes=submatrix,
            service_minutes=service_minutes,
            gallon_demands=gallon_demands,
            pound_demands=pound_demands,
            vehicle_gal_caps=vehicle_gal_caps,
            vehicle_lb_caps=vehicle_lb_caps,
            vehicle_time_caps=vehicle_time_caps,
            depot_index=0,
        )

        if result is not None:
            result["kept_local_nodes_original"] = kept_local_nodes
            return (
                result,
                kept_stop_ids,
                dropped_stop_ids,
                service_minutes,
                gallon_demands,
                pound_demands,
            )

        working = working.sort_values(
            ["stop_pickup_gal", "stop_pickup_lb", "stop_service_min"],
            ascending=[True, True, True],
        ).copy()

        drop_stop = str(working.iloc[0]["Stop_ID"])
        dropped_stop_ids.append(drop_stop)
        working = working.iloc[1:].copy()

    return None, [], dropped_stop_ids, [], [], []

def main() -> None:
    parser = argparse.ArgumentParser(description="Solve daily Bigbelly routing from Phase 1 schedule.")
    parser.add_argument("--projection-input", type=str, default=None)
    parser.add_argument("--schedule", type=str, default=None)
    parser.add_argument("--truck-streams", type=str, default=None)
    parser.add_argument("--routing-nodes", type=str, default=None)
    parser.add_argument("--bin-stop-lookup", type=str, default=None)
    parser.add_argument("--travel-matrix", type=str, default=None)
    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)
    parser.add_argument("--truck-work-min", type=float, default=750.0)
    parser.add_argument("--depot-node", type=int, default=0)
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    projection = read_projection(paths, args.projection_input)
    projection = projection.copy()
    projection["Serial"] = projection["Serial"].astype(str).str.strip()
    projection["stream"] = projection["stream"].apply(canonical_stream)

    schedule_fp = Path(args.schedule) if args.schedule else paths["processed"] / "small_instance_service_schedule.csv"
    truck_stream_fp = (
        Path(args.truck_streams) if args.truck_streams else paths["processed"] / "small_instance_truck_streams.csv"
    )
    routing_nodes_fp = Path(args.routing_nodes) if args.routing_nodes else paths["processed"] / "routing_nodes.csv"
    bin_stop_lookup_fp = Path(args.bin_stop_lookup) if args.bin_stop_lookup else paths["processed"] / "bin_stop_lookup.csv"
    travel_matrix_fp = Path(args.travel_matrix) if args.travel_matrix else paths["processed"] / "travel_matrix_wide.csv"

    if not schedule_fp.exists():
        raise FileNotFoundError("Could not find small_instance_service_schedule.csv")
    if not truck_stream_fp.exists():
        raise FileNotFoundError("Could not find small_instance_truck_streams.csv")
    if not routing_nodes_fp.exists():
        raise FileNotFoundError("Could not find routing_nodes.csv")
    if not bin_stop_lookup_fp.exists():
        raise FileNotFoundError("Could not find bin_stop_lookup.csv")
    if not travel_matrix_fp.exists():
        raise FileNotFoundError("Could not find travel_matrix_wide.csv")

    schedule = pd.read_csv(schedule_fp)
    truck_streams = pd.read_csv(truck_stream_fp)
    nodes_df = pd.read_csv(routing_nodes_fp)
    bin_stop_lookup = pd.read_csv(bin_stop_lookup_fp)
    matrix_df = load_matrix(travel_matrix_fp)

    if schedule.empty:
        raise ValueError("Service schedule is empty. Solve Phase 1 first.")

    schedule = schedule.dropna(subset=["Serial", "service_day", "truck", "stream"]).copy()
    schedule = schedule[schedule["service_day"] == 0].copy()

    if schedule.empty:
        raise ValueError("No day-0 service assignments found for routing.")
    schedule["Serial"] = schedule["Serial"].astype(str).str.strip()
    schedule["stream"] = schedule["stream"].apply(canonical_stream)
    schedule["service_day"] = schedule["service_day"].astype(int)
    schedule["pickup_gal"] = pd.to_numeric(schedule["pickup_gal"], errors="coerce").fillna(0.0)
    schedule["pickup_lb"] = pd.to_numeric(schedule["pickup_lb"], errors="coerce").fillna(0.0)

    truck_streams = truck_streams.copy()
    truck_streams["assigned_stream"] = truck_streams["assigned_stream"].apply(
        lambda x: canonical_stream(x) if pd.notna(x) else x
    )
    truck_streams["day"] = pd.to_numeric(truck_streams["day"], errors="coerce").astype("Int64")
    truck_streams["extra_dumps"] = pd.to_numeric(truck_streams["extra_dumps"], errors="coerce").fillna(0).astype(int)

    bin_stop_lookup = bin_stop_lookup.copy()
    bin_stop_lookup["Serial"] = bin_stop_lookup["Serial"].astype(str).str.strip()
    bin_stop_lookup["Stop_ID"] = bin_stop_lookup["Stop_ID"].apply(normalize_stop_id)

    projection_small = projection[[
        "Serial",
        "avg_service_min",
        "density_lb_per_gal",
        "stream",
        "Description",
    ]].drop_duplicates(subset=["Serial"]).copy()

    enriched = schedule.merge(projection_small, on=["Serial", "stream"], how="left")
    enriched["avg_service_min"] = pd.to_numeric(enriched["avg_service_min"], errors="coerce").fillna(4.0)
    enriched["density_lb_per_gal"] = pd.to_numeric(enriched["density_lb_per_gal"], errors="coerce").fillna(1.0)
    if "Description" not in enriched.columns:
        enriched["Description"] = enriched["Serial"]
    else:
        enriched["Description"] = enriched["Description"].fillna(enriched["Serial"])

    stop_to_node = stop_map_from_nodes(nodes_df)
    serial_to_stop = dict(zip(bin_stop_lookup["Serial"], bin_stop_lookup["Stop_ID"]))

    enriched["Stop_ID"] = enriched["Serial"].map(serial_to_stop)
    enriched["Stop_ID"] = enriched["Stop_ID"].apply(normalize_stop_id)

    missing_stops = sorted(
        enriched.loc[enriched["Stop_ID"].isna(), "Serial"].astype(str).unique().tolist()
    )
    if missing_stops:
        raise KeyError(
            f"The following scheduled bins are missing from bin_stop_lookup.csv: {missing_stops[:10]}"
        )

    missing_node_stops = sorted(
        enriched.loc[~enriched["Stop_ID"].astype(str).isin(stop_to_node.keys()), "Stop_ID"]
        .astype(str)
        .unique()
        .tolist()
    )
    if missing_node_stops:
        raise KeyError(
            f"The following scheduled Stop_ID values are missing from routing_nodes.csv: {missing_node_stops[:10]}"
        )

    stream_volume_cap = {
        "Waste": args.waste_volume_gal,
        "Compostables": args.compost_volume_gal,
        "Bottles/Cans": args.recycling_volume_gal,
    }

    route_plan_rows: list[dict[str, Any]] = []
    route_stop_rows: list[dict[str, Any]] = []
    route_summary_rows: list[dict[str, Any]] = []

    grouped = enriched.groupby(["service_day", "stream"], dropna=False)
    for (day, stream), grp in grouped:
        grp = grp.sort_values(["truck", "Serial"]).reset_index(drop=True)
        active = truck_streams[
            (truck_streams["day"] == int(day)) & (truck_streams["assigned_stream"] == stream)
        ].copy()
        active = active.sort_values("truck").reset_index(drop=True)

        if active.empty:
            active = pd.DataFrame(
                {
                    "day": [int(day)] * grp["truck"].nunique(),
                    "truck": sorted(grp["truck"].dropna().astype(str).unique().tolist()),
                    "assigned_stream": [stream] * grp["truck"].nunique(),
                    "extra_dumps": [0] * grp["truck"].nunique(),
                }
            )

        trucks = active["truck"].astype(str).tolist()

        local_nodes = [{
            "local_node": 0,
            "global_node": int(args.depot_node),
            "Stop_ID": None,
            "Serial": None,
            "label": "DEPOT",
        }]
        stop_rows = grp[["Stop_ID", "Description"]].drop_duplicates(subset=["Stop_ID"]).copy()

        for _, r in stop_rows.iterrows():
            local_nodes.append(
                {
                    "local_node": len(local_nodes),
                    "global_node": int(stop_to_node[str(r.Stop_ID)]),
                    "Stop_ID": str(r.Stop_ID),
                    "Serial": None,
                    "label": str(r.Description),
                }
            )
        local_nodes_df = pd.DataFrame(local_nodes)

        local_to_global = dict(zip(local_nodes_df["local_node"], local_nodes_df["global_node"]))
        stop_to_local = {
            row.Stop_ID: int(row.local_node)
            for row in local_nodes_df.itertuples(index=False)
            if pd.notna(row.Stop_ID)
        }

        local_ids = local_nodes_df["local_node"].tolist()
        matrix_minutes = []
        for i in local_ids:
            row = []
            gi = local_to_global[i]
            for j in local_ids:
                gj = local_to_global[j]
                row.append(int(round(float(matrix_df.loc[gi, gj]))))
            matrix_minutes.append(row)

        stop_agg = (
            grp.groupby("Stop_ID", as_index=False)
            .agg(
                stop_pickup_gal=("pickup_gal", "sum"),
                stop_pickup_lb=("pickup_lb", "sum"),
                stop_service_min=("avg_service_min", "sum"),
            )
            .copy()
        )

        service_minutes = [0] * len(local_ids)
        gallon_demands = [0] * len(local_ids)
        pound_demands = [0] * len(local_ids)
        stop_lookup: dict[int, str] = {}
        label_lookup: dict[int, str] = {0: "DEPOT"}

        stop_label_map = grp.groupby("Stop_ID")["Description"].first().to_dict()
        stop_bins_map = grp.groupby("Stop_ID")["Serial"].apply(
            lambda x: ";".join(sorted(set(x.astype(str))))
        ).to_dict()

        for _, r in stop_agg.iterrows():
            ln = stop_to_local[str(r.Stop_ID)]
            service_minutes[ln] = int(round(float(r.stop_service_min)))
            gallon_demands[ln] = int(round(float(r.stop_pickup_gal)))
            pound_demands[ln] = int(round(float(r.stop_pickup_lb)))
            stop_lookup[ln] = str(r.Stop_ID)
            label_lookup[ln] = str(stop_label_map.get(str(r.Stop_ID), r.Stop_ID))

        total_gal = sum(gallon_demands)
        total_lb = sum(pound_demands)
        total_service_min = sum(service_minutes)
        num_trucks = len(trucks)
        base_vol_cap = effective_volume_cap(stream, stream_volume_cap)

        vehicle_gal_caps = []
        vehicle_lb_caps = []
        vehicle_time_caps = []
        truck_extra_dumps = {}
        for row in active.itertuples(index=False):
            extra = int(row.extra_dumps)
            truck_extra_dumps[str(row.truck)] = extra
            vehicle_gal_caps.append(int(round(base_vol_cap * (1 + extra))))
            vehicle_lb_caps.append(int(round(args.truck_mass_lb * (1 + extra))))
            vehicle_time_caps.append(int(round(args.truck_work_min)))

        total_effective_gal_cap = sum(vehicle_gal_caps)
        total_effective_lb_cap = sum(vehicle_lb_caps)

        use_single_truck_route = False
        if num_trucks == 1:
            nonbinding_capacity = (total_gal <= base_vol_cap) and (total_lb <= args.truck_mass_lb)
            use_single_truck_route = nonbinding_capacity

        route_summary_rows.append(
            {
                "day": int(day),
                "stream": stream,
                "scheduled_bins": int(len(grp)),
                "scheduled_stops": int(len(stop_rows)),
                "active_trucks": int(num_trucks),
                "selected_mode": "SingleTruckRoute" if use_single_truck_route and num_trucks == 1 else "VRP",
                "total_pickup_gal": int(total_gal),
                "total_pickup_lb": int(total_lb),
                "total_service_min_only": int(total_service_min),
                "total_effective_volume_cap": int(total_effective_gal_cap),
                "total_effective_mass_cap": int(total_effective_lb_cap),
            }
        )

        result, kept_stop_ids, dropped_stop_ids, service_minutes, gallon_demands, pound_demands = solve_with_stop_dropping(
            stop_agg=stop_agg,
            stop_to_local=stop_to_local,
            matrix_minutes_full=matrix_minutes,
            vehicle_gal_caps=vehicle_gal_caps,
            vehicle_lb_caps=vehicle_lb_caps,
            vehicle_time_caps=vehicle_time_caps,
        )

        if result is None:
            print(
                f"[WARN] Could not find feasible routing solution for day={day}, stream={stream}. "
                "Skipping this stream-day."
            )
            continue

        if dropped_stop_ids:
            print(
                f"[WARN] Routing infeasible for day={day}, stream={stream}. "
                f"Dropped {len(dropped_stop_ids)} low-demand stop(s): {dropped_stop_ids[:10]}"
                + (" ..." if len(dropped_stop_ids) > 10 else "")
            )

        kept_local_nodes_original = result.get(
            "kept_local_nodes_original",
            list(range(len(matrix_minutes)))
        )

        for vehicle_idx, route in enumerate(result["routes"]):
            truck = trucks[vehicle_idx]

            original_route = [kept_local_nodes_original[n] for n in route]

            route_node_count = sum(1 for n in original_route if n != 0)
            route_gal = sum(gallon_demands[n] for n in route)
            route_lb = sum(pound_demands[n] for n in route)
            route_min = result["route_minutes"][vehicle_idx]
            mode = "SingleTruckRoute" if use_single_truck_route and num_trucks == 1 else "VRP"

            if route_node_count == 0:
                continue

            route_plan_rows.append(
                {
                    "day": int(day),
                    "stream": stream,
                    "truck": truck,
                    "routing_mode": mode,
                    "num_stops": int(route_node_count),
                    "route_gal": int(route_gal),
                    "route_lb": int(route_lb),
                    "route_minutes": int(route_min),
                    "volume_capacity_effective": int(vehicle_gal_caps[vehicle_idx]),
                    "mass_capacity_effective": int(vehicle_lb_caps[vehicle_idx]),
                    "extra_dumps_from_phase1": int(truck_extra_dumps[truck]),
                }
            )

            stop_order = 0
            for reduced_node, original_node in zip(route, original_route):
                stop_order += 1
                stop_id = stop_lookup.get(original_node)
                serials_here = stop_bins_map.get(stop_id, None) if stop_id is not None else None
                route_stop_rows.append(
                    {
                        "day": int(day),
                        "stream": stream,
                        "truck": truck,
                        "routing_mode": mode,
                        "stop_order": stop_order,
                        "local_node": int(original_node),
                        "Stop_ID": stop_id,
                        "Serials_at_stop": serials_here,
                        "label": label_lookup.get(original_node, "DEPOT"),
                        "is_depot": int(original_node == 0),
                        "pickup_gal": int(gallon_demands[reduced_node]),
                        "pickup_lb": int(pound_demands[reduced_node]),
                        "service_min": int(service_minutes[reduced_node]),
                    }
                )

    route_plan_df = pd.DataFrame(route_plan_rows)
    route_stops_df = pd.DataFrame(route_stop_rows)
    route_summary_df = pd.DataFrame(route_summary_rows)

    plan_fp = paths["processed"] / "daily_route_plan.csv"
    stops_fp = paths["processed"] / "daily_route_stops.csv"
    summary_fp = paths["processed"] / "daily_route_summary.csv"

    route_plan_df.to_csv(plan_fp, index=False)
    route_stops_df.to_csv(stops_fp, index=False)
    route_summary_df.to_csv(summary_fp, index=False)

    print(f"[OK] Wrote: {plan_fp}")
    print(f"[OK] Wrote: {stops_fp}")
    print(f"[OK] Wrote: {summary_fp}")

    if not route_plan_df.empty:
        print("\nRoute plan preview:")
        print(route_plan_df.head(15).to_string(index=False))
    else:
        print("\nNo active routes were generated.")


if __name__ == "__main__":
    main()