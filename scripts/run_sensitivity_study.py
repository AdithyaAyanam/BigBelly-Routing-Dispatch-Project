from __future__ import annotations

"""
run_sensitivity_study.py
-------------------------------------------------------------
Purpose
-------
Automate a batch sensitivity study for the Bigbelly weekly planning model.

This script runs solve_7day_schedule.py multiple times across different:
- truck counts
- overtime limits
- overflow penalties

For each run, it:
1. calls solve_7day_schedule.py with a chosen parameter set
2. reads the resulting planning summary CSV
3. copies key outputs into a dedicated scenario folder
4. writes one master sensitivity summary table

Expected upstream files
-----------------------
- data/processed/bin_7day_projection_inputs.parquet or .csv
- scripts/solve_7day_schedule.py

Outputs
-------
- data/processed/sensitivity_runs/<scenario_name>/...
- data/processed/sensitivity_study_summary.csv
"""

import argparse
import shutil
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_dir = processed_dir / "sensitivity_runs"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "processed": processed_dir,
        "sensitivity": sensitivity_dir,
    }


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def scenario_name(trucks: int, overtime: float, overflow_penalty: float, max_bins: int | None) -> str:
    bins_part = "all" if max_bins is None else f"bins{max_bins}"
    return f"t{trucks}_ot{overtime:g}_ovp{overflow_penalty:g}_{bins_part}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sensitivity study for weekly Bigbelly planning model.")
    parser.add_argument("--max-bins", type=int, default=100, help="Instance size for study; use none by omitting or setting manually in code")
    parser.add_argument("--truck-work-min", type=float, default=480.0)
    parser.add_argument("--cbc-time-limit-sec", type=int, default=180)
    parser.add_argument("--cbc-gap-rel", type=float, default=0.10)
    parser.add_argument("--require-routable", action="store_true")
    parser.add_argument("--trucks", type=str, default="3,4,5", help="Comma-separated truck counts")
    parser.add_argument("--overtime-values", type=str, default="0,60,180", help="Comma-separated overtime limits")
    parser.add_argument("--overflow-penalties", type=str, default="1,5,10", help="Comma-separated overflow penalties")
    parser.add_argument("--tiny-pickup-threshold-gal", type=float, default=5.0)
    parser.add_argument("--tiny-pickup-penalty", type=float, default=0.25)
    args = parser.parse_args()

    root = repo_root()
    paths = ensure_dirs(root)

    trucks_list = [int(x.strip()) for x in args.trucks.split(",") if x.strip()]
    overtime_list = [float(x.strip()) for x in args.overtime_values.split(",") if x.strip()]
    overflow_penalty_list = [float(x.strip()) for x in args.overflow_penalties.split(",") if x.strip()]

    solve_script = root / "scripts" / "solve_7day_schedule.py"
    if not solve_script.exists():
        raise FileNotFoundError("Could not find scripts/solve_7day_schedule.py")

    summary_rows: list[dict[str, object]] = []

    for trucks, overtime, overflow_penalty in product(trucks_list, overtime_list, overflow_penalty_list):
        scen = scenario_name(
            trucks=trucks,
            overtime=overtime,
            overflow_penalty=overflow_penalty,
            max_bins=args.max_bins,
        )
        scen_dir = paths["sensitivity"] / scen
        scen_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(solve_script),
            "--num-trucks", str(trucks),
            "--truck-work-min", str(args.truck_work_min),
            "--max-overtime-min", str(overtime),
            "--overflow-penalty", str(overflow_penalty),
            "--tiny-pickup-threshold-gal", str(args.tiny_pickup_threshold_gal),
            "--tiny-pickup-penalty", str(args.tiny_pickup_penalty),
            "--cbc-time-limit-sec", str(args.cbc_time_limit_sec),
            "--cbc-gap-rel", str(args.cbc_gap_rel),
        ]

        if args.max_bins is not None:
            cmd.extend(["--max-bins", str(args.max_bins)])

        if args.require_routable:
            cmd.append("--require-routable")

        print(f"\n[RUNNING] {scen}")
        result = run_command(cmd, cwd=root)

        stdout_fp = scen_dir / "stdout.txt"
        stderr_fp = scen_dir / "stderr.txt"
        stdout_fp.write_text(result.stdout or "", encoding="utf-8")
        stderr_fp.write_text(result.stderr or "", encoding="utf-8")

        planning_summary_fp = paths["processed"] / "small_instance_planning_summary.csv"
        schedule_fp = paths["processed"] / "small_instance_service_schedule.csv"
        truck_streams_fp = paths["processed"] / "small_instance_truck_streams.csv"
        load_check_fp = paths["processed"] / "small_instance_truck_load_check.csv"
        inventory_fp = paths["processed"] / "small_instance_inventory_trajectory.csv"

        copy_if_exists(planning_summary_fp, scen_dir / "planning_summary.csv")
        copy_if_exists(schedule_fp, scen_dir / "service_schedule.csv")
        copy_if_exists(truck_streams_fp, scen_dir / "truck_streams.csv")
        copy_if_exists(load_check_fp, scen_dir / "truck_load_check.csv")
        copy_if_exists(inventory_fp, scen_dir / "inventory_trajectory.csv")

        row: dict[str, object] = {
            "scenario": scen,
            "num_trucks": trucks,
            "max_overtime_min": overtime,
            "overflow_penalty": overflow_penalty,
            "max_bins": args.max_bins,
            "return_code": result.returncode,
        }

        if planning_summary_fp.exists():
            plan = pd.read_csv(planning_summary_fp)
            if not plan.empty:
                for col in plan.columns:
                    row[col] = plan.iloc[0][col]

        # quantify zero / near-zero pickups for this scenario
        if schedule_fp.exists():
            sched = pd.read_csv(schedule_fp)
            if "pickup_gal" in sched.columns:
                pickup_gal = pd.to_numeric(sched["pickup_gal"], errors="coerce").fillna(0.0)
                row["scheduled_visits"] = int(len(sched))
                row["zero_pickup_visits"] = int((pickup_gal <= 1e-9).sum())
                row["near_zero_under_5gal"] = int(((pickup_gal > 0) & (pickup_gal < 5)).sum())
                row["near_zero_under_10gal"] = int(((pickup_gal > 0) & (pickup_gal < 10)).sum())
                row["share_zero_pickup"] = float((pickup_gal <= 1e-9).mean()) if len(sched) else 0.0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["num_trucks", "max_overtime_min", "overflow_penalty"],
        ascending=[True, True, True],
    )

    out_fp = paths["processed"] / "sensitivity_study_summary.csv"
    summary_df.to_csv(out_fp, index=False)

    print(f"\n[OK] Wrote: {out_fp}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()