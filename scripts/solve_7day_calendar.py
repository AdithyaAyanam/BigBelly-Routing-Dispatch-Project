from __future__ import annotations

"""
solve_7day_schedule.py
-------------------------------------------------------------
Purpose
-------
Solve a seven-day Bigbelly collection planning model without fixed geographic zones.

This version is aligned with the professor-confirmation assumption table:

Operational assumptions
-----------------------
- Planning horizon: 7 calendar days for inventory lookahead.
- Service days: 5 weekday operating days; no pickups are allowed on weekend/non-working days.
- Primary truck-day resource: 480 minutes, representing one 8-hour shift.
- Sensitivity alternative: 750 minutes, retained only as a prior extended-span comparison.
- Primary fleet: 4 trucks, with one driver assigned to each truck.
- Backup vehicle: not separately modeled in the baseline.
- Depot: Edwards Track.
- Payload limit: 1,300 lb per trip.
- Volume capacities:
  Waste = 500 gallons,
  Compostables = 500 gallons,
  Bottles/Cans = 450 gallons.
- No hidden 99% capacity safety factor.
- Dump turnaround time: 17 minutes.
- Physical service time per bin: 4 minutes.
- Average travel time between stops: 8 minutes.
- Driver wage: $40/hour.
- Base labor rate: $40/60 = $0.667/minute.
- Overtime rate: 1.5 * $40/60 = $1.00/minute.
- Dump-cycle cost: 17 * $40/60 = $11.33.
- Approximate pickup cost: (4 + 8) * $40/60 = $8.00.
- Overflow penalty: default $20 per overflow bin-day, with $15 as a sensitivity value.
- Overflow penalty is applied across all seven calendar days, including weekend/non-service days.
- 60% threshold penalty: 0; threshold is diagnostic/planning signal only.
- Tiny pickup penalty: 0; realistic pickup labor cost discourages unnecessary pickups.
- Maximum extra dumps: default 3 per truck-stream-day.
- Compost weekday hygiene rule: optional flag requires every compost bin to be serviced at least once within the 5 service days.
- CBC time limit and optimality gap are configurable.
"""

import argparse
from datetime import datetime, timedelta
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

# ---------------------------------------------------------------------
# Report-aligned constants
# ---------------------------------------------------------------------

MAX_FILL_FACTOR = 1.50

NORMAL_LIFT_TRUCKS = 4
BACKUP_NONLIFT_TRUCKS = 0

SHIFT_WINDOWS = "4 trucks, one driver per truck, 8-hour shift"
DEPOT_NAME = "Edwards Track"


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
        "bottles and cans": "Bottles/Cans",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a seven-day Bigbelly scheduling instance."
    )

    parser.add_argument("--input", type=str, default=None)
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help="Optional instance size cap; default keeps all eligible bins.",
    )

    # ------------------------------------------------------------------
    # Core planning assumptions matched to the professor-confirmation table
    # ------------------------------------------------------------------

    parser.add_argument(
        "--num-trucks",
        type=int,
        default=4,
        help="Primary modeled fleet: 4 trucks, with one driver assigned to each truck.",
    )

    parser.add_argument(
        "--horizon-days",
        type=int,
        default=7,
        help="Seven-calendar-day planning horizon used for inventory lookahead.",
    )

    parser.add_argument(
        "--num-service-days",
        type=int,
        default=5,
        help=(
            "Expected number of weekday operating days within the planning horizon. "
            "Actual service days are calculated from --start-date and the calendar."
        ),
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help=(
            "Calendar start date of the planning horizon in YYYY-MM-DD format. "
            "Example: --start-date 2026-02-04."
        ),
    )

    parser.add_argument(
        "--truck-work-min",
        type=float,
        default=480.0,
        help="Primary truck-day resource: 480 minutes, representing one 8-hour shift.",
    )

    parser.add_argument(
        "--sensitivity-truck-work-min",
        type=float,
        default=750.0,
        help="Sensitivity alternative: 750-minute extended-span comparison from earlier model version.",
    )

    parser.add_argument(
        "--truck-mass-lb",
        type=float,
        default=1300.0,
        help="Payload limit W = 1,300 lb per trip.",
    )

    parser.add_argument(
        "--waste-volume-gal",
        type=float,
        default=500.0,
        help="Effective volume limit for Waste, V_Waste = 500 gallons.",
    )

    parser.add_argument(
        "--compost-volume-gal",
        type=float,
        default=500.0,
        help="Effective volume limit for Compostables, V_Comp = 500 gallons.",
    )

    parser.add_argument(
        "--recycling-volume-gal",
        type=float,
        default=450.0,
        help="Effective volume limit for Bottles/Cans, V_BC = 450 gallons.",
    )

    parser.add_argument(
        "--service-min-per-bin",
        type=float,
        default=4.0,
        help="Planning-stage service time per bin, tau_i approximately 4 minutes.",
    )

    parser.add_argument(
        "--travel-min-between-stops",
        type=float,
        default=8.0,
        help="Average travel time between stops, approximately 8 minutes.",
    )

    parser.add_argument(
        "--use-input-service-travel",
        action="store_true",
        help=(
            "Optional: use avg_service_min and avg_travel_proxy_min from input instead "
            "of the table assumptions of 4 minutes service + 8 minutes travel."
        ),
    )

    parser.add_argument(
        "--dump-turnaround-min",
        type=float,
        default=17.0,
        help="Dump turnaround time at Edwards Track, t_dump = 17 minutes.",
    )

    parser.add_argument(
        "--max-extra-dumps",
        type=int,
        default=3,
        help=(
            "Maximum extra dump cycles per truck-stream-day. "
            "Default 3 is a more defensible operational cap than 8."
        ),
    )

    parser.add_argument(
        "--driver-wage-per-hour",
        type=float,
        default=40.0,
        help="Base hourly wage used to convert service, travel, and dump time into dollars.",
    )

    parser.add_argument(
        "--dump-penalty",
        type=float,
        default=None,
        help=(
            "Cost per extra dump cycle. If omitted, computed as "
            "driver_wage_per_hour / 60 * dump_turnaround_min."
        ),
    )

    parser.add_argument(
        "--max-overtime-min",
        type=float,
        default=0.0,
        help="Baseline overtime allowance per truck-day. Default is 0 for four 8-hour driver shifts.",
    )

    parser.add_argument(
        "--overtime-penalty-per-min",
        type=float,
        default=None,
        help=(
            "Overtime cost per minute. If omitted, computed as "
            "1.5 * driver_wage_per_hour / 60."
        ),
    )

    parser.add_argument(
        "--overflow-penalty",
        type=float,
        default=20.0,
        help="Main-case penalty per overflow bin-day, in dollars. Professor suggested about $15-$20.",
    )

    parser.add_argument(
        "--tiny-pickup-threshold-gal",
        type=float,
        default=5.0,
        help="Diagnostic threshold for tiny pickups.",
    )

    parser.add_argument(
        "--tiny-pickup-penalty",
        type=float,
        default=0.0,
        help="Keep at 0 because realistic pickup labor cost discourages unnecessary pickups.",
    )

    parser.add_argument("--required-only", action="store_true")

    parser.add_argument(
        "--enforce-compost-weekday-hygiene",
        action="store_true",
        help=(
            "If set, require every compost bin to be serviced at least once "
            "during the 5 weekday service days, reflecting that drivers "
            "do not work weekends."
        ),
    )

    parser.add_argument(
        "--require-routable",
        action="store_true",
        help=(
            "Optional legacy filter. Do not use for the baseline planning model, "
            "because this stage is a scheduling/planning model rather than a routing model."
        ),
    )

    parser.add_argument(
        "--cbc-time-limit-sec",
        type=int,
        default=1200,
        help="CBC time limit in seconds for the final run.",
    )

    parser.add_argument(
        "--cbc-gap-rel",
        type=float,
        default=0.10,
        help="CBC relative MIP gap tolerance.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    if args.num_trucks != NORMAL_LIFT_TRUCKS:
        print(
            f"[WARN] Professor-aligned baseline uses {NORMAL_LIFT_TRUCKS} trucks, "
            f"but num_trucks={args.num_trucks}"
        )

    if args.num_service_days > args.horizon_days:
        raise ValueError("--num-service-days cannot exceed --horizon-days")

    if args.num_service_days < 1:
        raise ValueError("--num-service-days must be at least 1")
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("--start-date must be in YYYY-MM-DD format, e.g., 2026-02-04") from exc


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

    if "is_active_bin" in df.columns:
        before_active = df["Serial"].nunique()
        df["is_active_bin"] = df["is_active_bin"].fillna(False).astype(bool)
        df = df[df["is_active_bin"]].copy()
        after_active = df["Serial"].nunique()
        print(f"[INFO] active-bin filter: {after_active}/{before_active} bins retained")

        if df.empty:
            raise ValueError("No active bins remain after applying is_active_bin filter.")

    if args.require_routable:
        stop_lookup_serials = load_stop_lookup_serials(root)

        before = df["Serial"].nunique()
        df = df[df["Serial"].isin(stop_lookup_serials)].copy()
        after = df["Serial"].nunique()

        print(f"[INFO] require_routable: {after}/{before} bins retained")

        if df.empty:
            raise ValueError(
                "No routable bins remain after applying --require-routable filter."
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
    # ------------------------------------------------------------------
    # Calendar-correct service-day logic
    # ------------------------------------------------------------------
    # IMPORTANT: Do not assume days 0-4 are weekdays and days 5-6 are weekend.
    # That assumption is only valid when the horizon starts on a Monday.
    # Example: if start_date is 2026-02-04, then:
    #   day 0 = Wed 2026-02-04
    #   day 1 = Thu 2026-02-05
    #   day 2 = Fri 2026-02-06
    #   day 3 = Sat 2026-02-07  non-service
    #   day 4 = Sun 2026-02-08  non-service
    #   day 5 = Mon 2026-02-09
    #   day 6 = Tue 2026-02-10
    # ------------------------------------------------------------------
    days = list(range(args.horizon_days))
    date_by_day = {d: start_date + timedelta(days=d) for d in days}
    weekday_by_day = {d: date_by_day[d].strftime("%A") for d in days}

    service_days = [d for d in days if date_by_day[d].weekday() < 5]
    nonservice_days = [d for d in days if date_by_day[d].weekday() >= 5]
    if len(service_days) != args.num_service_days:
        print(
            f"[WARN] Calendar-derived service days count is {len(service_days)}, "
            f"but --num-service-days={args.num_service_days}. "
            "Using calendar-derived service_days."
        )
    trucks = [f"T{k + 1}" for k in range(args.num_trucks)]

    bin_stream = dict(zip(instance["Serial"], instance["stream"]))
    threshold_pct = dict(zip(instance["Serial"], instance["threshold_pct"]))
    deadline_interval = dict(zip(instance["Serial"], instance["deadline_interval"]))

    if args.use_input_service_travel:
        service_min = dict(zip(instance["Serial"], instance["avg_service_min"]))
        travel_min = dict(zip(instance["Serial"], instance["avg_travel_proxy_min"]))
        service_travel_source = "input columns avg_service_min and avg_travel_proxy_min"
    else:
        service_min = {i: float(args.service_min_per_bin) for i in instance["Serial"]}
        travel_min = {i: float(args.travel_min_between_stops) for i in instance["Serial"]}
        service_travel_source = "fixed assumptions: 4 min service + 8 min average travel"

    bin_capacity_gal = dict(zip(instance["Serial"], instance["bin_capacity_gal"]))
    density_lb_per_gal = dict(zip(instance["Serial"], instance["density_lb_per_gal"]))

    effective_truck_work_min = args.truck_work_min

    if abs(effective_truck_work_min - 480.0) < 1e-6:
        resource_model = "four_trucks_one_driver_each_8hr_shift"
    elif abs(effective_truck_work_min - 750.0) < 1e-6:
        resource_model = "extended_span_sensitivity"
    else:
        resource_model = "custom_truck_day"

    print(f"[INFO] resource_model = {resource_model}")
    print(f"[INFO] truck_work_min = {args.truck_work_min}")
    print(f"[INFO] sensitivity_truck_work_min = {args.sensitivity_truck_work_min}")
    print(f"[INFO] service_days = {service_days}")
    print(f"[INFO] nonservice_days = {nonservice_days}")
    print("[INFO] calendar_map = " + "; ".join(
        f"day {d}: {date_by_day[d].isoformat()} {weekday_by_day[d]} "
        f"({'service' if d in service_days else 'non-service'})"
        for d in days
    ))
    print(f"[INFO] shift_windows = {SHIFT_WINDOWS}")
    print(f"[INFO] depot = {DEPOT_NAME}")
    print(f"[INFO] service_travel_source = {service_travel_source}")

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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

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

    overflow_slack_gal = LpVariable.dicts(
        "overflow_slack_gal",
        (bins, days),
        lowBound=0,
    )

    threshold_violation = LpVariable.dicts(
        "threshold_violation",
        (bins, days),
        lowBound=0,
    )

    # ------------------------------------------------------------------
    # Dollar-based objective coefficients
    # ------------------------------------------------------------------

    wage_per_min = args.driver_wage_per_hour / 60.0

    dump_penalty = (
        args.dump_penalty
        if args.dump_penalty is not None
        else wage_per_min * args.dump_turnaround_min
    )

    overtime_penalty_per_min = (
        args.overtime_penalty_per_min
        if args.overtime_penalty_per_min is not None
        else 1.5 * wage_per_min
    )

    approx_total_min_per_pickup = args.service_min_per_bin + args.travel_min_between_stops
    approx_cost_per_pickup = wage_per_min * approx_total_min_per_pickup

    print(f"[INFO] driver_wage_per_hour = {args.driver_wage_per_hour}")
    print(f"[INFO] wage_per_min = {wage_per_min:.4f}")
    print(f"[INFO] approx_total_min_per_pickup = {approx_total_min_per_pickup:.2f}")
    print(f"[INFO] approx_cost_per_pickup = {approx_cost_per_pickup:.2f}")
    print(f"[INFO] dump_penalty = {dump_penalty:.3f}")
    print(f"[INFO] overtime_penalty_per_min = {overtime_penalty_per_min:.3f}")
    print(f"[INFO] overflow_penalty = {args.overflow_penalty}")
    print("[INFO] overflow_penalty_applies_to = all 7 calendar days including non-service days")
    print("[INFO] threshold_penalty = 0.0")
    print(f"[INFO] tiny_pickup_penalty = {args.tiny_pickup_penalty}")

    # Small tie-breaker only. It should not dominate dollar costs.
    epsilon = 0.001

    pickup_labor_cost_expr = wage_per_min * lpSum(
        (float(service_min[i]) + float(travel_min[i])) * y[i][k][d]
        for i in bins
        for k in trucks
        for d in service_days
    )

    dump_cost_expr = dump_penalty * lpSum(
        extra_dumps[k][s][d]
        for k in trucks
        for s in streams
        for d in service_days
    )

    overtime_cost_expr = overtime_penalty_per_min * lpSum(
        overtime_min[k][d]
        for k in trucks
        for d in service_days
    )

    # Overflow is penalized over the full seven-calendar-day lookahead,
    # including calendar weekend/non-service days. Drivers cannot collect on
    # weekend days, but weekend overflow still matters operationally and should
    # influence service decisions before the weekend.
    overflow_cost_expr = args.overflow_penalty * lpSum(
        overflow_flag[i][d]
        for i in bins
        for d in days
    )

    model += (
        pickup_labor_cost_expr
        + dump_cost_expr
        + overtime_cost_expr
        + overflow_cost_expr
        + epsilon * lpSum((service_days[-1] - d) * x[i][d] for i in bins for d in service_days)
    )

    # ------------------------------------------------------------------
    # Inventory, service, overflow, threshold diagnostics
    # ------------------------------------------------------------------

    for i in bins:
        model += inventory_gal[i][0] == current_fill_gal[i], f"initial_inventory_{i}"
        model += inventory_gal[i][0] <= max_inventory_gal[i], f"initial_cap_{i}"

        interval_due = deadline_interval[i]
        if pd.notna(interval_due):
            interval_due = int(interval_due)
            model += (
                lpSum(x[i][d] for d in range(interval_due + 1)) >= 1
            ), f"interval_deadline_{i}"

        # Practical compost hygiene rule:
        # Drivers do not work weekends, so the operational rule is effectively
        # service within 5 weekdays. In the 7-day planning horizon, enforce that
        # every compost bin receives at least one service when this option is enabled.
        if args.enforce_compost_weekday_hygiene and bin_stream[i] == "Compostables":
            model += (
                lpSum(x[i][d] for d in service_days) >= 1
            ), f"compost_weekday_hygiene_{i}"

        # Drivers do not work weekends/non-service days. Inventory is still tracked
        # across the full seven-calendar-day lookahead, but service decisions,
        # pickup volume, and truck assignment are forced to zero on non-service days.
        for d in nonservice_days:
            model += x[i][d] == 0, f"no_service_nonworking_day_{i}_{d}"
            model += pickup_gal[i][d] == 0, f"no_pickup_volume_nonworking_day_{i}_{d}"

        for d in days:
            model += inventory_gal[i][d] <= max_inventory_gal[i], f"inventory_cap_{i}_{d}"

            model += (
                post_service_inventory_gal[i][d] <= inventory_gal[i][d]
            ), f"post_le_inventory_{i}_{d}"

            model += (
                post_service_inventory_gal[i][d]
                <= max_inventory_gal[i] * (1 - x[i][d])
                + (max_inventory_gal[i] - float(bin_capacity_gal[i])) * x[i][d]
            ), f"reset_if_pick_{i}_{d}"

            model += (
                post_service_inventory_gal[i][d]
                >= inventory_gal[i][d] - float(bin_capacity_gal[i]) * x[i][d]
            ), f"carry_if_no_pick_{i}_{d}"

            model += (
                pickup_gal[i][d]
                == inventory_gal[i][d] - post_service_inventory_gal[i][d]
            ), f"pickup_def_{i}_{d}"

            model += (
                pickup_gal[i][d] <= float(bin_capacity_gal[i]) * x[i][d]
            ), f"pickup_activate_{i}_{d}"

            model += (
                inventory_gal[i][d]
                <= float(bin_capacity_gal[i]) + overflow_slack_gal[i][d]
            ), f"overflow_slack_def_{i}_{d}"

            model += (
                overflow_slack_gal[i][d]
                <= (max_inventory_gal[i] - float(bin_capacity_gal[i])) * overflow_flag[i][d]
            ), f"overflow_flag_link_{i}_{d}"

            # Tiny pickup is diagnostic only because tiny_pickup_penalty defaults to 0.
            model += (
                pickup_gal[i][d]
                <= args.tiny_pickup_threshold_gal
                + float(bin_capacity_gal[i]) * (1 - tiny_pickup[i][d])
            ), f"tiny_pickup_upper_{i}_{d}"

            model += (
                pickup_gal[i][d]
                >= args.tiny_pickup_threshold_gal * (x[i][d] - tiny_pickup[i][d])
            ), f"tiny_pickup_lower_{i}_{d}"

            model += x[i][d] >= tiny_pickup[i][d], f"tiny_pickup_only_if_service_{i}_{d}"

            # Threshold violation is diagnostic only; it is not penalized in the objective.
            if pd.notna(threshold_gal[i]):
                model += (
                    inventory_gal[i][d] <= threshold_gal[i] + threshold_violation[i][d]
                ), f"soft_threshold_{i}_{d}"

            if d < days[-1]:
                model += (
                    inventory_gal[i][d + 1]
                    == post_service_inventory_gal[i][d] + growth_gal_per_day[i]
                ), f"inventory_flow_{i}_{d}"

    # ------------------------------------------------------------------
    # Truck assignment and stream compatibility
    # ------------------------------------------------------------------

    for i in bins:
        for d in nonservice_days:
            for k in trucks:
                model += y[i][k][d] == 0, f"no_truck_assignment_nonworking_day_{i}_{k}_{d}"
                model += pickup_gal_truck[i][k][d] == 0, f"no_truck_pickup_nonworking_day_{i}_{k}_{d}"

    for k in trucks:
        for s in streams:
            for d in nonservice_days:
                model += z[k][s][d] == 0, f"no_stream_assignment_nonworking_day_{k}_{s}_{d}"
                model += extra_dumps[k][s][d] == 0, f"no_extra_dumps_nonworking_day_{k}_{s}_{d}"

    for k in trucks:
        for d in nonservice_days:
            model += overtime_min[k][d] == 0, f"no_overtime_nonworking_day_{k}_{d}"

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
                    pickup_gal_truck[i][k][d] <= float(bin_capacity_gal[i]) * y[i][k][d]
                ), f"truck_pickup_activate_{i}_{k}_{d}"

                model += (
                    pickup_gal_truck[i][k][d]
                    >= pickup_gal[i][d] - float(bin_capacity_gal[i]) * (1 - y[i][k][d])
                ), f"truck_pickup_lower_{i}_{k}_{d}"

            model += (
                lpSum(pickup_gal_truck[i][k][d] for k in trucks) == pickup_gal[i][d]
            ), f"truck_pickup_balance_{i}_{d}"

    for k in trucks:
        for d in days:
            model += (
                lpSum(z[k][s][d] for s in streams) <= 1
            ), f"one_stream_{k}_{d}"

    # Symmetry breaking: use earlier truck labels first when possible.
    for d in service_days:
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

    # ------------------------------------------------------------------
    # Truck capacity constraints
    # ------------------------------------------------------------------
    # Uses stated stream-specific volume capacities directly:
    #   Waste:        500 gal
    #   Compostables: 500 gal
    #   Bottles/Cans: 450 gal
    #
    # No 99% volume safety factor is applied.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Truck-day workload constraints
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    schedule_rows = []

    for i in bins:
        for d in days:
            if safe_value(x[i][d]) > 0.5:
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
                        "service_date": date_by_day[d].isoformat(),
                        "weekday": weekday_by_day[d],
                        "is_service_day": d in service_days,
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
                "service_date": None,
                "weekday": None,
                "is_service_day": None,
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
                    "date": date_by_day[d].isoformat(),
                    "weekday": weekday_by_day[d],
                    "is_service_day": d in service_days,
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
                    "date": date_by_day[d].isoformat(),
                    "weekday": weekday_by_day[d],
                    "is_service_day": d in service_days,
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
                    "date": date_by_day[d].isoformat(),
                    "weekday": weekday_by_day[d],
                    "is_service_day": d in service_days,
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
                    "overflow_flag": int(safe_value(overflow_slack_gal[i][d]) > 1e-6),
                    "overflow_slack_gal": round(safe_value(overflow_slack_gal[i][d]), 2),
                    "threshold_violation_gal": round(
                        safe_value(threshold_violation[i][d]),
                        2,
                    ),
                }
            )

    total_pickups = sum(safe_value(x[i][d]) for i in bins for d in service_days)

    total_extra_dumps = sum(
        safe_value(extra_dumps[k][s][d])
        for k in trucks
        for s in streams
        for d in service_days
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
        1
        for i in bins
        for d in days
        if safe_value(overflow_slack_gal[i][d]) > 1e-6
    )

    total_overflow_slack_gal = sum(
        safe_value(overflow_slack_gal[i][d])
        for i in bins
        for d in days
    )

    pickup_labor_cost_value = safe_value(pickup_labor_cost_expr)
    dump_cost_value = safe_value(dump_cost_expr)
    overtime_cost_value = safe_value(overtime_cost_expr)
    overflow_cost_value = safe_value(overflow_cost_expr)

    planning_summary_df = pd.DataFrame(
        [
            {
                "status": status,
                "objective_value": round(objective_value, 3),
                "horizon_days": args.horizon_days,
                "start_date": start_date.isoformat(),
                "end_date": date_by_day[days[-1]].isoformat(),
                "calendar_map": "; ".join(
                    f"day {d}: {date_by_day[d].isoformat()} {weekday_by_day[d]} "
                    f"({'service' if d in service_days else 'non-service'})"
                    for d in days
                ),
                "num_service_days": args.num_service_days,
                "calendar_derived_num_service_days": len(service_days),
                "service_days": ",".join(str(d) for d in service_days),
                "nonservice_days": ",".join(str(d) for d in nonservice_days),
                "num_bins_in_instance": len(bins),
                "num_trucks": args.num_trucks,
                "primary_fleet": "4 trucks, one driver assigned to each",
                "backup_vehicle": "not separately modeled in professor-aligned baseline",
                "truck_work_min_input": args.truck_work_min,
                "effective_truck_work_min": effective_truck_work_min,
                "sensitivity_truck_work_min": args.sensitivity_truck_work_min,
                "resource_model": resource_model,
                "staggered_shift_windows": SHIFT_WINDOWS,
                "depot": DEPOT_NAME,
                "normal_lift_trucks": NORMAL_LIFT_TRUCKS,
                "backup_nonlift_trucks": BACKUP_NONLIFT_TRUCKS,
                "total_pickups": round(total_pickups, 2),
                "total_extra_dumps": round(total_extra_dumps, 2),
                "total_overtime_min": round(total_overtime, 2),
                "overflow_bin_days": int(overflow_bin_days),
                "overflow_flag_days": int(overflow_flag_days),
                "total_overflow_slack_gal": round(total_overflow_slack_gal, 2),
                "driver_wage_per_hour": args.driver_wage_per_hour,
                "wage_per_min": round(wage_per_min, 4),
                "pickup_cost_method": "wage_per_min * (service_min_per_bin + travel_min_between_stops)",
                "service_travel_source": service_travel_source,
                "service_min_per_bin": args.service_min_per_bin,
                "travel_min_between_stops": args.travel_min_between_stops,
                "approx_total_min_per_pickup": round(approx_total_min_per_pickup, 2),
                "approx_cost_per_pickup": round(approx_cost_per_pickup, 2),
                "pickup_labor_cost": round(pickup_labor_cost_value, 2),
                "dump_turnaround_min": args.dump_turnaround_min,
                "dump_penalty": round(dump_penalty, 3),
                "dump_cost": round(dump_cost_value, 2),
                "max_overtime_min": args.max_overtime_min,
                "overtime_penalty_per_min": round(overtime_penalty_per_min, 3),
                "overtime_cost": round(overtime_cost_value, 2),
                "overflow_penalty": args.overflow_penalty,
                "overflow_cost": round(overflow_cost_value, 2),
                "expected_overflow_cost_from_model_flags": round(
                    args.overflow_penalty * overflow_flag_days,
                    2,
                ),
                "overflow_penalty_applies_to": "all 7 calendar days including non-service days",
                "max_fill_factor": MAX_FILL_FACTOR,
                "volume_capacity_factor": 1.0,
                "volume_capacity_note": "Uses stated capacities directly; no 99 percent safety factor.",
                "waste_volume_gal": args.waste_volume_gal,
                "compost_volume_gal": args.compost_volume_gal,
                "recycling_volume_gal": args.recycling_volume_gal,
                "truck_mass_lb": args.truck_mass_lb,
                "max_extra_dumps": args.max_extra_dumps,
                "one_stream_per_truck_day": True,
                "fixed_geographic_zones": False,
                "enforce_compost_weekday_hygiene": bool(args.enforce_compost_weekday_hygiene),
                "weekday_service_only": True,
                "weekend_service_allowed": False,
                "threshold_penalty": 0.0,
                "tiny_pickup_penalty": args.tiny_pickup_penalty,
                "cbc_time_limit_sec": args.cbc_time_limit_sec,
                "cbc_gap_rel": args.cbc_gap_rel,
                "legacy_require_routable_filter_used": bool(args.require_routable),
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
        missing_assignments = schedule_df[schedule_df["truck"].isna()][
            ["Serial", "service_day", "stream"]
        ]
        print("[ERROR] Schedule contains serviced bins without truck assignments:")
        print(missing_assignments.head(20).to_string(index=False))
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
