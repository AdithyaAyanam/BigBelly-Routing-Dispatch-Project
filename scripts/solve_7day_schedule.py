from __future__ import annotations

"""
solve_7day_schedule.py
-------------------------------------------------------------
Purpose
-------
Solve a multi-day Bigbelly scheduling instance without geographic zoning.

This version reflects Professor Yano's feedback:
1. Do NOT divide campus into fixed truck zones.
2. Solve a higher-level planning problem over a horizon of about a week.
3. Allow a bin to be serviced more than once within the horizon.
4. Use inventory-style state variables:
   - inventory_gal[i,d] = start-of-day fill,
   - pickup resets bin inventory after service,
   - inventory then grows according to daily fill rate.
5. Enforce realistic truck constraints:
   - one stream per truck per day,
   - mass limit,
   - volume limit,
   - shift-time limit,
   - extra dump cycles when needed.
6. Allow overtime at a penalty cost.
7. Filter to bins that are routable when routing consistency is required.
8. Support optional weekly pickup bounds for sensitivity/compliance runs.
9. Support hard overflow-bin-day caps for zero-overflow feasibility testing.
10. Support tunable dump-cycle penalty.

Inputs
------
Expected input file:
    data/processed/bin_7day_projection_inputs.parquet
or:
    data/processed/bin_7day_projection_inputs.csv

Required columns:
- Serial
- stream
- threshold_pct
- days_since_last_service
- current_fill_pct_est
- daily_fill_growth_pct
- bin_capacity_gal
- density_lb_per_gal
- avg_service_min
- avg_travel_proxy_min
- must_service_within_horizon

Outputs
-------
- small_instance_service_schedule.csv
- small_instance_truck_streams.csv
- small_instance_truck_load_check.csv
- small_instance_inventory_trajectory.csv
- small_instance_planning_summary.csv
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

MAX_FILL_FACTOR = 1.50

NORMAL_LIFT_TRUCKS = 3
BACKUP_NONLIFT_TRUCKS = 1

SHIFT_WINDOWS = [
    ("S1", 4 * 60, 12 * 60 + 30),   # 4:00–12:30
    ("S2", 6 * 60, 14 * 60 + 30),   # 6:00–2:30
    ("S3", 8 * 60, 16 * 60 + 30),   # 8:00–4:30
]

FLEET_DAY_START_MIN = 4 * 60
FLEET_DAY_END_MIN = 16 * 60 + 30
FLEET_DAY_SPAN_MIN = FLEET_DAY_END_MIN - FLEET_DAY_START_MIN  # 750 min


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "processed": processed_dir}


def load_stop_lookup_serials(root: Path) -> set[str]:
    fp = root / "data" / "processed" / "bin_stop_lookup.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")

    df_lookup = pd.read_csv(fp)

    if "Serial" not in df_lookup.columns:
        raise KeyError("bin_stop_lookup.csv missing 'Serial'")

    return set(df_lookup["Serial"].astype(str).str.strip())


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


def interval_deadline(days_since_last_service: float, horizon_days: int) -> float:
    remaining = 7 - float(days_since_last_service)

    if remaining <= 0:
        return 0.0

    if 1 <= remaining <= horizon_days - 1:
        return float(int(remaining))

    return np.nan


def safe_value(var: object) -> float:
    v = value(var)
    return 0.0 if v is None else float(v)


def choose_instance(
    df: pd.DataFrame,
    max_bins: int | None,
    required_only: bool,
    horizon_days: int,
) -> pd.DataFrame:
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
        optional_sorted = optional.sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )

        return (
            pd.concat(
                [
                    required.sort_values(
                        ["service_deadline", "days_since_last_service", "Serial"],
                        ascending=[True, False, True],
                    ),
                    optional_sorted,
                ],
                ignore_index=True,
            )
            .drop_duplicates(subset=["Serial"])
            .reset_index(drop=True)
        )

    picks = []
    per_deadline = max(1, max_bins // max(1, horizon_days))

    for d in range(horizon_days):
        bucket = required[required["service_deadline"] == d].sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        picks.append(bucket.head(per_deadline))

    chosen = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["Serial"])

    if len(chosen) < max_bins:
        remaining_required = required[
            ~required["Serial"].isin(chosen["Serial"])
        ].sort_values(
            ["service_deadline", "days_since_last_service", "Serial"],
            ascending=[True, False, True],
        )
        chosen = pd.concat(
            [chosen, remaining_required.head(max_bins - len(chosen))],
            ignore_index=True,
        )

    if len(chosen) < max_bins:
        remaining_optional = optional.sort_values(
            ["days_since_last_service", "Serial"],
            ascending=[False, True],
        )
        chosen = pd.concat(
            [chosen, remaining_optional.head(max_bins - len(chosen))],
            ignore_index=True,
        )

    return chosen.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a weekly Bigbelly scheduling instance.")

    parser.add_argument("--input", type=str, default=None)
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help="Optional instance size cap; default keeps all eligible bins.",
    )
    parser.add_argument("--num-trucks", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=7)

    parser.add_argument("--truck-mass-lb", type=float, default=1300.0)
    parser.add_argument("--waste-volume-gal", type=float, default=500.0)
    parser.add_argument("--compost-volume-gal", type=float, default=500.0)
    parser.add_argument("--recycling-volume-gal", type=float, default=450.0)

    parser.add_argument("--truck-work-min", type=float, default=480.0)

    parser.add_argument(
        "--use-observed-shift-span",
        action="store_true",
        help=(
            "Sensitivity option only: approximate staggered driver shifts by using "
            "the observed fleet-day span of 750 minutes instead of the default 480."
        ),
    )

    parser.add_argument("--dump-turnaround-min", type=float, default=17.0)
    parser.add_argument("--max-extra-dumps", type=int, default=8)

    parser.add_argument(
        "--dump-penalty",
        type=float,
        default=0.01,
        help="Penalty per extra dump cycle.",
    )

    parser.add_argument("--max-overtime-min", type=float, default=180.0)
    parser.add_argument("--overtime-penalty-per-min", type=float, default=0.02)

    parser.add_argument("--cbc-time-limit-sec", type=int, default=180)
    parser.add_argument("--cbc-gap-rel", type=float, default=0.05)

    parser.add_argument(
        "--min-weekly-pickups",
        type=int,
        default=None,
        help="Optional lower bound on total pickups over the planning horizon.",
    )
    parser.add_argument(
        "--max-weekly-pickups",
        type=int,
        default=None,
        help="Optional upper bound on total pickups over the planning horizon.",
    )

    parser.add_argument(
        "--min-day0-pickups",
        type=int,
        default=None,
        help="Optional lower bound on pickups executed on Day 0 of the planning horizon.",
    )
    parser.add_argument(
        "--max-day0-pickups",
        type=int,
        default=None,
        help="Optional upper bound on pickups executed on Day 0 of the planning horizon.",
    )

    parser.add_argument(
        "--overflow-penalty",
        type=float,
        default=5.0,
        help="Penalty per overflow bin-day.",
    )

    parser.add_argument(
        "--max-overflow-bin-days",
        type=int,
        default=None,
        help="Optional hard upper bound on total overflow bin-days.",
    )

    parser.add_argument(
        "--tiny-pickup-threshold-gal",
        type=float,
        default=5.0,
        help="Threshold for tiny pickups.",
    )
    parser.add_argument(
        "--tiny-pickup-penalty",
        type=float,
        default=0.25,
        help="Penalty for scheduling a tiny pickup.",
    )

    parser.add_argument("--required-only", action="store_true")
    parser.add_argument(
        "--require-routable",
        action="store_true",
        help=(
            "If set, keep only bins that exist in bin_stop_lookup.csv for downstream "
            "routing consistency."
        ),
    )

    args = parser.parse_args()

    if (
        args.min_day0_pickups is not None
        and args.max_day0_pickups is not None
        and args.min_day0_pickups > args.max_day0_pickups
    ):
        raise ValueError("--min-day0-pickups cannot exceed --max-day0-pickups")

    if (
        args.min_weekly_pickups is not None
        and args.max_weekly_pickups is not None
        and args.min_weekly_pickups > args.max_weekly_pickups
    ):
        raise ValueError("--min-weekly-pickups cannot exceed --max-weekly-pickups")

    if args.max_overflow_bin_days is not None and args.max_overflow_bin_days < 0:
        raise ValueError("--max-overflow-bin-days cannot be negative")

    root = repo_root()
    paths = ensure_dirs(root)

    if args.num_trucks != NORMAL_LIFT_TRUCKS:
        print(
            f"[WARN] Field observation suggests {NORMAL_LIFT_TRUCKS} normal lift trucks, "
            f"but num_trucks={args.num_trucks}"
        )

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
                "Missing bin_7day_projection_inputs. "
                "Run python scripts/build_projected_fill.py first."
            )

    if input_fp.suffix.lower() == ".csv":
        df = pd.read_csv(input_fp)
    else:
        df = pd.read_parquet(input_fp)

    required_cols = [
        "Serial",
        "stream",
        "threshold_pct",
        "days_since_last_service",
        "current_fill_pct_est",
        "daily_fill_growth_pct",
        "bin_capacity_gal",
        "density_lb_per_gal",
        "avg_service_min",
        "avg_travel_proxy_min",
        "must_service_within_horizon",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input file missing columns: {missing}")

    if "service_deadline" not in df.columns:
        df["service_deadline"] = np.nan

    if "deadline_interval" not in df.columns:
        df["deadline_interval"] = df["days_since_last_service"].apply(
            lambda x: interval_deadline(x, args.horizon_days)
        )

    df["Serial"] = df["Serial"].astype(str).str.strip()
    df["stream"] = df["stream"].apply(canonical_stream)
    df["must_service_within_horizon"] = (
        df["must_service_within_horizon"].fillna(False).astype(bool)
    )

    if args.require_routable:
        stop_lookup_serials = load_stop_lookup_serials(root)

        before = df["Serial"].nunique()
        df = df[df["Serial"].isin(stop_lookup_serials)].copy()
        after = df["Serial"].nunique()

        print(f"[INFO] require_routable: {after}/{before} bins retained (aligned with routing)")

        if df.empty:
            raise ValueError(
                "No routable bins remain after applying --require-routable filter. "
                "Increase routing coverage."
            )

    instance = choose_instance(
        df=df,
        max_bins=args.max_bins,
        required_only=args.required_only,
        horizon_days=args.horizon_days,
    )

    if instance.empty:
        raise ValueError("No bins available for solve after filtering.")

    bins = instance["Serial"].astype(str).tolist()
    streams = [
        s
        for s in sorted(instance["stream"].dropna().astype(str).unique().tolist())
        if s != "Unknown"
    ]
    days = list(range(args.horizon_days))
    trucks = [f"T{k + 1}" for k in range(args.num_trucks)]

    bin_stream = dict(zip(instance["Serial"], instance["stream"]))
    threshold_pct = dict(zip(instance["Serial"], instance["threshold_pct"]))
    deadline_interval = dict(zip(instance["Serial"], instance["deadline_interval"]))
    service_min = dict(zip(instance["Serial"], instance["avg_service_min"]))
    travel_min = dict(zip(instance["Serial"], instance["avg_travel_proxy_min"]))
    bin_capacity_gal = dict(zip(instance["Serial"], instance["bin_capacity_gal"]))
    density_lb_per_gal = dict(zip(instance["Serial"], instance["density_lb_per_gal"]))

    effective_truck_work_min = args.truck_work_min
    resource_model = "single_shift_truck_day"

    if args.use_observed_shift_span:
        effective_truck_work_min = FLEET_DAY_SPAN_MIN
        resource_model = "observed_shift_span_truck_day"

    print(f"[INFO] resource_model = {resource_model}")
    print(f"[INFO] truck_work_min_input = {args.truck_work_min}")
    print(f"[INFO] effective_truck_work_min = {effective_truck_work_min}")

    max_inventory_gal = {
        i: float(bin_capacity_gal[i]) * MAX_FILL_FACTOR
        for i in bins
    }

    current_fill_gal = {
        i: float(bin_capacity_gal[i])
        * float(instance.loc[instance["Serial"] == i, "current_fill_pct_est"].iloc[0])
        / 100.0
        for i in bins
    }

    growth_gal_per_day = {
        i: float(bin_capacity_gal[i])
        * float(instance.loc[instance["Serial"] == i, "daily_fill_growth_pct"].iloc[0])
        / 100.0
        for i in bins
    }

    threshold_gal = {
        i: (
            np.nan
            if pd.isna(threshold_pct[i])
            else float(bin_capacity_gal[i]) * float(threshold_pct[i]) / 100.0
        )
        for i in bins
    }

    stream_volume_cap = {
        "Waste": args.waste_volume_gal,
        "Compostables": args.compost_volume_gal,
        "Bottles/Cans": args.recycling_volume_gal,
    }

    model = LpProblem("Bigbelly_Weekly_Schedule_No_Zones", LpMinimize)

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

    overtime_min = LpVariable.dicts(
        "overtime_min",
        (trucks, days),
        lowBound=0,
        upBound=args.max_overtime_min,
    )

    inventory_gal = LpVariable.dicts("inventory_gal", (bins, days), lowBound=0)
    post_service_inventory_gal = LpVariable.dicts(
        "post_service_inventory_gal",
        (bins, days),
        lowBound=0,
    )
    pickup_gal = LpVariable.dicts("pickup_gal", (bins, days), lowBound=0)
    pickup_gal_truck = LpVariable.dicts(
        "pickup_gal_truck",
        (bins, trucks, days),
        lowBound=0,
    )

    tiny_pickup = LpVariable.dicts("tiny_pickup", (bins, days), cat=LpBinary)
    overflow_flag = LpVariable.dicts("overflow_flag", (bins, days), cat=LpBinary)
    threshold_violation = LpVariable.dicts(
        "threshold_violation",
        (bins, days),
        lowBound=0,
    )

    epsilon = 0.001
    dump_penalty = args.dump_penalty

    total_pickups_expr = lpSum(x[i][d] for i in bins for d in days)
    day0_pickups_expr = lpSum(x[i][0] for i in bins)

    model += (
        total_pickups_expr
        + epsilon * lpSum((days[-1] - d) * x[i][d] for i in bins for d in days)
        + dump_penalty * lpSum(
            extra_dumps[k][s][d]
            for k in trucks
            for s in streams
            for d in days
        )
        + args.overtime_penalty_per_min * lpSum(
            overtime_min[k][d]
            for k in trucks
            for d in days
        )
        + args.overflow_penalty * lpSum(
            overflow_flag[i][d]
            for i in bins
            for d in days
        )
        + 0.5 * lpSum(
            threshold_violation[i][d]
            for i in bins
            for d in days
        )
        + args.tiny_pickup_penalty * lpSum(
            tiny_pickup[i][d]
            for i in bins
            for d in days
        )
    )

    if args.min_weekly_pickups is not None:
        model += total_pickups_expr >= args.min_weekly_pickups, "min_weekly_pickups"

    if args.max_weekly_pickups is not None:
        model += total_pickups_expr <= args.max_weekly_pickups, "max_weekly_pickups"

    if args.min_day0_pickups is not None:
        model += day0_pickups_expr >= args.min_day0_pickups, "min_day0_pickups"

    if args.max_day0_pickups is not None:
        model += day0_pickups_expr <= args.max_day0_pickups, "max_day0_pickups"

    if args.max_overflow_bin_days is not None:
        model += (
            lpSum(overflow_flag[i][d] for i in bins for d in days)
            <= args.max_overflow_bin_days
        ), "max_overflow_bin_days"

    for i in bins:
        model += inventory_gal[i][0] == current_fill_gal[i], f"initial_inventory_{i}"
        model += inventory_gal[i][0] <= max_inventory_gal[i], f"initial_cap_{i}"

        interval_due = deadline_interval[i]
        if pd.notna(interval_due):
            interval_due = int(interval_due)
            model += (
                lpSum(x[i][d] for d in range(interval_due + 1)) >= 1
            ), f"interval_deadline_{i}"

        for d in days:
            model += inventory_gal[i][d] <= max_inventory_gal[i], f"inventory_cap_{i}_{d}"
            model += (
                post_service_inventory_gal[i][d] <= inventory_gal[i][d]
            ), f"post_le_inventory_{i}_{d}"

            model += (
                post_service_inventory_gal[i][d]
                <= max_inventory_gal[i] * (1 - x[i][d])
            ), f"reset_if_pick_{i}_{d}"

            model += (
                post_service_inventory_gal[i][d]
                >= inventory_gal[i][d] - max_inventory_gal[i] * x[i][d]
            ), f"carry_if_no_pick_{i}_{d}"

            model += (
                pickup_gal[i][d]
                == inventory_gal[i][d] - post_service_inventory_gal[i][d]
            ), f"pickup_def_{i}_{d}"

            model += (
                pickup_gal[i][d] <= max_inventory_gal[i] * x[i][d]
            ), f"pickup_activate_{i}_{d}"

            model += (
                inventory_gal[i][d]
                <= float(bin_capacity_gal[i])
                + (max_inventory_gal[i] - float(bin_capacity_gal[i]))
                * overflow_flag[i][d]
            ), f"overflow_flag_link_{i}_{d}"

            model += (
                pickup_gal[i][d]
                <= args.tiny_pickup_threshold_gal
                + max_inventory_gal[i] * (1 - tiny_pickup[i][d])
            ), f"tiny_pickup_upper_{i}_{d}"

            model += (
                pickup_gal[i][d]
                >= args.tiny_pickup_threshold_gal * (x[i][d] - tiny_pickup[i][d])
            ), f"tiny_pickup_lower_{i}_{d}"

            model += x[i][d] >= tiny_pickup[i][d], f"tiny_pickup_only_if_service_{i}_{d}"

            if pd.notna(threshold_gal[i]):
                model += (
                    inventory_gal[i][d] <= threshold_gal[i] + threshold_violation[i][d]
                ), f"soft_threshold_{i}_{d}"

            if d < days[-1]:
                model += (
                    inventory_gal[i][d + 1]
                    == post_service_inventory_gal[i][d] + growth_gal_per_day[i]
                ), f"inventory_flow_{i}_{d}"

    for i in bins:
        for d in days:
            model += (
                lpSum(y[i][k][d] for k in trucks) == x[i][d]
            ), f"truck_assign_{i}_{d}"

            for k in trucks:
                model += (
                    pickup_gal_truck[i][k][d] <= pickup_gal[i][d]
                ), f"truck_pickup_le_total_{i}_{k}_{d}"

                model += (
                    pickup_gal_truck[i][k][d] <= max_inventory_gal[i] * y[i][k][d]
                ), f"truck_pickup_activate_{i}_{k}_{d}"

                model += (
                    pickup_gal_truck[i][k][d]
                    >= pickup_gal[i][d] - max_inventory_gal[i] * (1 - y[i][k][d])
                ), f"truck_pickup_lower_{i}_{k}_{d}"

            model += (
                lpSum(pickup_gal_truck[i][k][d] for k in trucks) == pickup_gal[i][d]
            ), f"truck_pickup_balance_{i}_{d}"

    for k in trucks:
        for d in days:
            model += (
                lpSum(z[k][s][d] for s in streams) <= 1
            ), f"one_stream_{k}_{d}"

    for d in days:
        for t in range(len(trucks) - 1):
            model += (
                lpSum(z[trucks[t]][s][d] for s in streams)
                >= lpSum(z[trucks[t + 1]][s][d] for s in streams)
            ), f"sym_use_{d}_{t}"

    for i in bins:
        s_i = bin_stream[i]
        for k in trucks:
            for d in days:
                if s_i in streams:
                    model += y[i][k][d] <= z[k][s_i][d], f"compat_{i}_{k}_{d}"
                else:
                    model += y[i][k][d] == 0, f"unknown_stream_block_{i}_{k}_{d}"

    for k in trucks:
        for s in streams:
            for d in days:
                assigned_gal = lpSum(
                    pickup_gal_truck[i][k][d]
                    for i in bins
                    if bin_stream[i] == s
                )

                assigned_lb = lpSum(
                    density_lb_per_gal[i] * pickup_gal_truck[i][k][d]
                    for i in bins
                    if bin_stream[i] == s
                )

                model += (
                    assigned_gal
                    <= stream_volume_cap[s] * (z[k][s][d] + extra_dumps[k][s][d])
                ), f"volcap_{k}_{s}_{d}"

                model += (
                    assigned_lb
                    <= args.truck_mass_lb * (z[k][s][d] + extra_dumps[k][s][d])
                ), f"masscap_{k}_{s}_{d}"

                model += (
                    extra_dumps[k][s][d] <= args.max_extra_dumps * z[k][s][d]
                ), f"dumpactivate_{k}_{s}_{d}"

    for k in trucks:
        for d in days:
            service_and_travel = lpSum(
                (service_min[i] + travel_min[i]) * y[i][k][d]
                for i in bins
            )

            dump_minutes = args.dump_turnaround_min * lpSum(
                extra_dumps[k][s][d]
                for s in streams
            )

            model += (
                service_and_travel + dump_minutes
                <= effective_truck_work_min + overtime_min[k][d]
            ), f"workload_{k}_{d}"

    solver = PULP_CBC_CMD(
        msg=True,
        timeLimit=args.cbc_time_limit_sec,
        gapRel=args.cbc_gap_rel,
    )
    model.solve(solver)

    status = LpStatus[model.status]
    objective_value = safe_value(model.objective)

    print(f"[STATUS] {status}")
    print(f"[OBJECTIVE] {objective_value:.3f}")

    if status not in {"Optimal", "Integer Feasible"}:
        print("Model did not find a usable integer-feasible solution. Outputs will not be trusted.")
        return

    schedule_rows = []

    for i in bins:
        for d in days:
            if safe_value(x[i][d]) > 0.5 and safe_value(pickup_gal[i][d]) > 1e-6:
                truck = None

                for k in trucks:
                    if safe_value(y[i][k][d]) > 0.5:
                        truck = k
                        break

                fill_pct_before = (
                    100.0 * safe_value(inventory_gal[i][d]) / float(bin_capacity_gal[i])
                )

                schedule_rows.append(
                    {
                        "Serial": i,
                        "stream": bin_stream[i],
                        "service_day": d,
                        "truck": truck,
                        "inventory_gal_before_service": round(safe_value(inventory_gal[i][d]), 2),
                        "inventory_pct_before_service": round(fill_pct_before, 2),
                        "pickup_gal": round(safe_value(pickup_gal[i][d]), 2),
                        "pickup_lb": round(
                            safe_value(pickup_gal[i][d]) * density_lb_per_gal[i],
                            2,
                        ),
                        "interval_deadline": (
                            None
                            if pd.isna(deadline_interval[i])
                            else int(deadline_interval[i])
                        ),
                    }
                )

    if not schedule_rows:
        schedule_rows.append(
            {
                "Serial": None,
                "stream": None,
                "service_day": None,
                "truck": None,
                "inventory_gal_before_service": None,
                "inventory_pct_before_service": None,
                "pickup_gal": None,
                "pickup_lb": None,
                "interval_deadline": None,
            }
        )

    truck_stream_rows = []

    for d in days:
        for k in trucks:
            chosen_stream = None
            chosen_extra_dumps = 0

            for s in streams:
                if safe_value(z[k][s][d]) > 0.5:
                    chosen_stream = s
                    chosen_extra_dumps = int(round(safe_value(extra_dumps[k][s][d])))
                    break

            truck_stream_rows.append(
                {
                    "day": d,
                    "truck": k,
                    "assigned_stream": chosen_stream,
                    "extra_dumps": chosen_extra_dumps,
                    "resource_model": resource_model,
                }
            )

    load_rows = []

    for d in days:
        for k in trucks:
            total_gal = sum(safe_value(pickup_gal_truck[i][k][d]) for i in bins)
            total_lb = sum(
                density_lb_per_gal[i] * safe_value(pickup_gal_truck[i][k][d])
                for i in bins
            )
            total_min = sum(
                (service_min[i] + travel_min[i]) * safe_value(y[i][k][d])
                for i in bins
            )

            assigned_stream = None
            extra_dump_count = 0
            volume_cap = None

            for s in streams:
                if safe_value(z[k][s][d]) > 0.5:
                    assigned_stream = s
                    extra_dump_count = int(round(safe_value(extra_dumps[k][s][d])))
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
                    "minutes_used_total": round(
                        total_min + args.dump_turnaround_min * extra_dump_count,
                        2,
                    ),
                    "overtime_min": round(safe_value(overtime_min[k][d]), 2),
                    "minutes_capacity_regular": effective_truck_work_min,
                    "minutes_capacity_with_overtime": round(
                        effective_truck_work_min + safe_value(overtime_min[k][d]),
                        2,
                    ),
                    "resource_model": resource_model,
                }
            )

    inventory_rows = []

    for i in bins:
        for d in days:
            inventory_rows.append(
                {
                    "Serial": i,
                    "stream": bin_stream[i],
                    "day": d,
                    "inventory_gal_start": round(safe_value(inventory_gal[i][d]), 2),
                    "inventory_pct_start": round(
                        100.0 * safe_value(inventory_gal[i][d]) / float(bin_capacity_gal[i]),
                        2,
                    ),
                    "pickup_flag": int(round(safe_value(x[i][d]))),
                    "pickup_gal": round(safe_value(pickup_gal[i][d]), 2),
                    "inventory_gal_after_service": round(
                        safe_value(post_service_inventory_gal[i][d]),
                        2,
                    ),
                    "overflow_flag": int(round(safe_value(overflow_flag[i][d]))),
                    "threshold_violation_gal": round(
                        safe_value(threshold_violation[i][d]),
                        2,
                    ),
                }
            )

    total_pickups = sum(safe_value(x[i][d]) for i in bins for d in days)
    total_extra_dumps = sum(
        safe_value(extra_dumps[k][s][d])
        for k in trucks
        for s in streams
        for d in days
    )
    total_overtime = sum(
        safe_value(overtime_min[k][d])
        for k in trucks
        for d in days
    )
    overflow_bin_days = sum(
        1
        for i in bins
        for d in days
        if safe_value(inventory_gal[i][d]) > float(bin_capacity_gal[i]) + 1e-6
    )
    overflow_flag_days = sum(
        int(round(safe_value(overflow_flag[i][d])))
        for i in bins
        for d in days
    )

    planning_summary_df = pd.DataFrame(
        [
            {
                "status": status,
                "objective_value": round(objective_value, 3),
                "horizon_days": args.horizon_days,
                "num_bins_in_instance": len(bins),
                "num_trucks": args.num_trucks,
                "truck_work_min_input": args.truck_work_min,
                "effective_truck_work_min": effective_truck_work_min,
                "resource_model": resource_model,
                "normal_lift_trucks": NORMAL_LIFT_TRUCKS,
                "backup_nonlift_trucks": BACKUP_NONLIFT_TRUCKS,
                "use_observed_shift_span": bool(args.use_observed_shift_span),
                "total_pickups": round(total_pickups, 2),
                "min_weekly_pickups": args.min_weekly_pickups,
                "max_weekly_pickups": args.max_weekly_pickups,
                "min_day0_pickups": args.min_day0_pickups,
                "max_day0_pickups": args.max_day0_pickups,
                "total_extra_dumps": round(total_extra_dumps, 2),
                "total_overtime_min": round(total_overtime, 2),
                "overflow_bin_days": int(overflow_bin_days),
                "overflow_flag_days": int(overflow_flag_days),
                "max_overflow_bin_days": args.max_overflow_bin_days,
                "dump_penalty": args.dump_penalty,
                "overflow_penalty": args.overflow_penalty,
                "cbc_time_limit_sec": args.cbc_time_limit_sec,
                "cbc_gap_rel": args.cbc_gap_rel,
                "require_routable": bool(args.require_routable),
            }
        ]
    )

    schedule_df = pd.DataFrame(schedule_rows).sort_values(
        ["service_day", "Serial"],
        na_position="last",
    )
    truck_stream_df = pd.DataFrame(truck_stream_rows)
    load_df = pd.DataFrame(load_rows)
    inventory_df = pd.DataFrame(inventory_rows)

    if not schedule_df.empty and schedule_df["truck"].isna().any():
        missing = schedule_df[schedule_df["truck"].isna()][["Serial", "service_day", "stream"]]
        print("[ERROR] Schedule contains serviced bins without truck assignments:")
        print(missing.head(20).to_string(index=False))
        return

    schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
    truck_stream_fp = paths["processed"] / "small_instance_truck_streams.csv"
    load_fp = paths["processed"] / "small_instance_truck_load_check.csv"
    inventory_fp = paths["processed"] / "small_instance_inventory_trajectory.csv"
    planning_summary_fp = paths["processed"] / "small_instance_planning_summary.csv"

    schedule_df.to_csv(schedule_fp, index=False)
    truck_stream_df.to_csv(truck_stream_fp, index=False)
    load_df.to_csv(load_fp, index=False)
    inventory_df.to_csv(inventory_fp, index=False)
    planning_summary_df.to_csv(planning_summary_fp, index=False)

    print(f"[OK] Wrote: {schedule_fp}")
    print(f"[OK] Wrote: {truck_stream_fp}")
    print(f"[OK] Wrote: {load_fp}")
    print(f"[OK] Wrote: {inventory_fp}")
    print(f"[OK] Wrote: {planning_summary_fp}")

    print("\nSchedule preview:")
    print(schedule_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()