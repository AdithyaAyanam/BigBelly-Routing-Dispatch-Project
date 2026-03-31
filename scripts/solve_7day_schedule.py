from __future__ import annotations

"""Solve a 7-day Bigbelly scheduling instance without geographic zoning.

Revisions in this version:
- no truck is pinned to a geographic zone;
- stream-specific threshold logic is expected upstream;
- both mass and volume limits are enforced;
- extra dump cycles are allowed within a shift via integer variables;
- recycling can use a lower effective volume cap than landfill/compost.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


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


def choose_instance(df: pd.DataFrame, max_bins: int | None, required_only: bool) -> pd.DataFrame:
    out = df.copy()
    if required_only:
        out = out[out["must_service_within_horizon"]].copy()

    if "service_deadline" not in out.columns:
        out["service_deadline"] = np.nan

    required = out[out["service_deadline"].notna()].copy()
    optional = out[out["service_deadline"].isna()].copy()

    if required.empty:
        chosen = optional.copy()
        if max_bins is not None:
            chosen = chosen.head(max_bins)
        return chosen.reset_index(drop=True)

    required["service_deadline"] = required["service_deadline"].astype(int)

    if max_bins is None:
        return required.sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    picks = []
    per_deadline = max(1, max_bins // 7)
    for d in range(7):
        bucket = required[required["service_deadline"] == d].sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        picks.append(bucket.head(per_deadline))

    chosen = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["Serial"])

    if len(chosen) < max_bins:
        remaining_required = required[~required["Serial"].isin(chosen["Serial"])].sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        )
        chosen = pd.concat([chosen, remaining_required.head(max_bins - len(chosen))], ignore_index=True)

    if len(chosen) < max_bins:
        remaining_optional = optional.sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        chosen = pd.concat([chosen, remaining_optional.head(max_bins - len(chosen))], ignore_index=True)

    return chosen.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a 7-day Bigbelly scheduling instance.")
    parser.add_argument("--input", type=str, default=None, help="Path to bin_7day_projection_inputs.parquet or .csv")
    parser.add_argument("--max-bins", type=int, default=18, help="Prototype instance size")
    parser.add_argument("--num-trucks", type=int, default=3)
    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)
    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--dump-turnaround-min", type=float, default=17.0)
    parser.add_argument("--max-extra-dumps", type=int, default=8)
    parser.add_argument("--required-only", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    if args.input:
        input_fp = Path(args.input)
    else:
        input_fp = paths["processed"] / "bin_7day_projection_inputs.parquet"

    if not input_fp.exists():
        csv_fp = paths["processed"] / "bin_7day_projection_inputs.csv"
        if csv_fp.exists():
            input_fp = csv_fp
        else:
            raise FileNotFoundError(
                "Missing bin_7day_projection_inputs. Run python scripts/build_projected_fill.py first."
            )

    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)

    required_cols = [
        "Serial",
        "stream",
        "days_since_last_service",
        "bin_capacity_gal",
        "density_lb_per_gal",
        "avg_service_min",
        "avg_travel_proxy_min",
        "must_service_within_horizon",
    ] + [f"fill_day_{d}" for d in range(7)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input file missing columns: {missing}")

    if "service_deadline" not in df.columns:
        df["service_deadline"] = np.nan

    df["stream"] = df["stream"].apply(canonical_stream)
    df["must_service_within_horizon"] = df["must_service_within_horizon"].fillna(False).astype(bool)

    instance = choose_instance(df, max_bins=args.max_bins, required_only=args.required_only)
    if instance.empty:
        raise ValueError("No bins available for solve after filtering.")

    bins = instance["Serial"].astype(str).tolist()
    streams = [s for s in sorted(instance["stream"].dropna().astype(str).unique().tolist()) if s != "Unknown"]
    days = list(range(7))
    trucks = [f"T{k + 1}" for k in range(args.num_trucks)]

    bin_stream = dict(zip(instance["Serial"], instance["stream"]))
    service_deadline = dict(zip(instance["Serial"], instance["service_deadline"]))
    service_min = dict(zip(instance["Serial"], instance["avg_service_min"]))
    travel_min = dict(zip(instance["Serial"], instance["avg_travel_proxy_min"]))
    bin_capacity_gal = dict(zip(instance["Serial"], instance["bin_capacity_gal"]))
    density_lb_per_gal = dict(zip(instance["Serial"], instance["density_lb_per_gal"]))

    stream_volume_cap = {
        "Waste": args.waste_volume_gal,
        "Compostables": args.compost_volume_gal,
        "Bottles/Cans": args.recycling_volume_gal,
    }

    projected_fill: dict[tuple[str, int], float] = {}
    pickup_gal: dict[tuple[str, int], float] = {}
    pickup_lb: dict[tuple[str, int], float] = {}

    for _, row in instance.iterrows():
        i = row["Serial"]
        for d in days:
            pct = float(row[f"fill_day_{d}"])
            gal = float(bin_capacity_gal[i]) * pct / 100.0
            lb = gal * float(density_lb_per_gal[i])
            projected_fill[(i, d)] = pct
            pickup_gal[(i, d)] = gal
            pickup_lb[(i, d)] = lb

    model = LpProblem("Bigbelly_7Day_Schedule_No_Zones", LpMinimize)

    x = LpVariable.dicts("x", (bins, days), cat=LpBinary)
    y = LpVariable.dicts("y", (bins, trucks, days), cat=LpBinary)
    z = LpVariable.dicts("z", (trucks, streams, days), cat=LpBinary)
    extra_dumps = LpVariable.dicts(
        "extra_dumps",
        (trucks, streams, days),
        lowBound=0,
        upBound=args.max_extra_dumps,
        cat=LpInteger,
    )

    epsilon = 0.001
    dump_penalty = 0.01
    model += (
        lpSum(x[i][d] for i in bins for d in days)
        + epsilon * lpSum((6 - d) * x[i][d] for i in bins for d in days)
        + dump_penalty * lpSum(extra_dumps[k][s][d] for k in trucks for s in streams for d in days)
    )

    for i in bins:
        model += lpSum(x[i][d] for d in days) <= 1, f"at_most_once_{i}"

    for i in bins:
        deadline = service_deadline[i]
        if pd.notna(deadline):
            deadline = int(deadline)
            model += lpSum(x[i][d] for d in range(deadline + 1)) >= 1, f"deadline_{i}"

    for i in bins:
        for d in days:
            model += lpSum(y[i][k][d] for k in trucks) == x[i][d], f"truck_assign_{i}_{d}"

    for k in trucks:
        for d in days:
            model += lpSum(z[k][s][d] for s in streams) <= 1, f"one_stream_{k}_{d}"

    for i in bins:
        s_i = bin_stream[i]
        for k in trucks:
            for d in days:
                if s_i in streams:
                    model += y[i][k][d] <= z[k][s_i][d], f"compat_{i}_{k}_{d}"
                else:
                    model += y[i][k][d] == 0, f"unknown_stream_block_{i}_{k}_{d}"

    # Allow multiple load cycles during a shift by paying extra dump turnaround time.
    for k in trucks:
        for s in streams:
            for d in days:
                assigned_gal = lpSum(pickup_gal[(i, d)] * y[i][k][d] for i in bins if bin_stream[i] == s)
                assigned_lb = lpSum(pickup_lb[(i, d)] * y[i][k][d] for i in bins if bin_stream[i] == s)
                model += assigned_gal <= stream_volume_cap[s] * (z[k][s][d] + extra_dumps[k][s][d]), f"volcap_{k}_{s}_{d}"
                model += assigned_lb <= args.truck_mass_lb * (z[k][s][d] + extra_dumps[k][s][d]), f"masscap_{k}_{s}_{d}"
                model += extra_dumps[k][s][d] <= args.max_extra_dumps * z[k][s][d], f"dumpactivate_{k}_{s}_{d}"

    for k in trucks:
        for d in days:
            service_and_travel = lpSum((service_min[i] + travel_min[i]) * y[i][k][d] for i in bins)
            dump_minutes = args.dump_turnaround_min * lpSum(extra_dumps[k][s][d] for s in streams)
            model += service_and_travel + dump_minutes <= args.truck_work_min, f"workload_{k}_{d}"

    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = LpStatus[model.status]
    print(f"[STATUS] {status}")
    if status != "Optimal":
        print("Model did not solve to optimality. Try reducing max_bins or increasing daily truck resources.")
        return

    schedule_rows = []
    for i in bins:
        picked = False
        for d in days:
            if value(x[i][d]) > 0.5:
                picked = True
                truck = None
                for k in trucks:
                    if value(y[i][k][d]) > 0.5:
                        truck = k
                        break
                schedule_rows.append(
                    {
                        "Serial": i,
                        "stream": bin_stream[i],
                        "service_day": d,
                        "truck": truck,
                        "projected_fill_pct_that_day": round(projected_fill[(i, d)], 2),
                        "projected_pickup_gal": round(pickup_gal[(i, d)], 2),
                        "projected_pickup_lb": round(pickup_lb[(i, d)], 2),
                        "service_deadline": None if pd.isna(service_deadline[i]) else int(service_deadline[i]),
                    }
                )
        if not picked:
            schedule_rows.append(
                {
                    "Serial": i,
                    "stream": bin_stream[i],
                    "service_day": None,
                    "truck": None,
                    "projected_fill_pct_that_day": None,
                    "projected_pickup_gal": None,
                    "projected_pickup_lb": None,
                    "service_deadline": None if pd.isna(service_deadline[i]) else int(service_deadline[i]),
                }
            )

    truck_stream_rows = []
    for d in days:
        for k in trucks:
            chosen_stream = None
            chosen_extra_dumps = 0
            for s in streams:
                if value(z[k][s][d]) > 0.5:
                    chosen_stream = s
                    chosen_extra_dumps = int(round(value(extra_dumps[k][s][d])))
                    break
            truck_stream_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "assigned_stream": chosen_stream,
                    "extra_dumps": chosen_extra_dumps,
                }
            )

    load_rows = []
    for d in days:
        for k in trucks:
            total_gal = sum(pickup_gal[(i, d)] * value(y[i][k][d]) for i in bins)
            total_lb = sum(pickup_lb[(i, d)] * value(y[i][k][d]) for i in bins)
            total_min = sum((service_min[i] + travel_min[i]) * value(y[i][k][d]) for i in bins)
            assigned_stream = None
            extra_dump_count = 0
            volume_cap = None
            for s in streams:
                if value(z[k][s][d]) > 0.5:
                    assigned_stream = s
                    extra_dump_count = int(round(value(extra_dumps[k][s][d])))
                    volume_cap = stream_volume_cap[s] * (1 + extra_dump_count)
                    break
            load_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "assigned_stream": assigned_stream,
                    "pickup_gal_used": round(total_gal, 2),
                    "pickup_gal_capacity_effective": volume_cap,
                    "pickup_lb_used": round(total_lb, 2),
                    "pickup_lb_capacity_effective": args.truck_mass_lb * (1 + extra_dump_count),
                    "extra_dumps": extra_dump_count,
                    "minutes_used_without_dumps": round(total_min, 2),
                    "minutes_used_total": round(total_min + args.dump_turnaround_min * extra_dump_count, 2),
                    "minutes_capacity": args.truck_work_min,
                }
            )

    schedule_df = pd.DataFrame(schedule_rows).sort_values(["service_day", "Serial"], na_position="last")
    truck_stream_df = pd.DataFrame(truck_stream_rows)
    load_df = pd.DataFrame(load_rows)

    schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
    truck_stream_fp = paths["processed"] / "small_instance_truck_streams.csv"
    load_fp = paths["processed"] / "small_instance_truck_load_check.csv"

    schedule_df.to_csv(schedule_fp, index=False)
    truck_stream_df.to_csv(truck_stream_fp, index=False)
    load_df.to_csv(load_fp, index=False)

    print(f"[OK] Wrote: {schedule_fp}")
    print(f"[OK] Wrote: {truck_stream_fp}")
    print(f"[OK] Wrote: {load_fp}")
    print("\nSchedule preview:")
    print(schedule_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
