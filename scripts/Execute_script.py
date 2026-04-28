from __future__ import annotations

"""
Execute_script.py
-------------------------------------------------------------
Runs the full Bigbelly workflow in order:

1. Compile/check scripts
2. Build projected fill inputs
3. Build travel matrix
4. Run bounded 7-day weekly optimization
5. Run Day-0 routing
6. Run 5-day rolling horizon
7. Print final summaries
8. Show git status

Final target:
- Use all routable bins
- Use 3 trucks
- Use 480-minute truck-day
- Keep weekly pickups in acceptable range: 250–300
- Keep Day 0 pickups in range: 45–55
- Increase routing time limit to 300 seconds
- Do NOT use observed 750-minute shift span

Run from project root:
    python scripts/Execute_script.py
"""

import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 90)
    print("[RUN]", " ".join(cmd))
    print("=" * 90)

    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def print_file_if_exists(root: Path, rel_fp: str) -> None:
    fp = root / rel_fp

    print("\n" + "-" * 90)
    print(rel_fp)
    print("-" * 90)

    if fp.exists():
        print(fp.read_text())
    else:
        print(f"[WARN] Missing file: {rel_fp}")


def main() -> None:
    root = repo_root()

    print("\nBIGBELLY FULL WORKFLOW")
    print(f"Project root: {root}")

    # ------------------------------------------------------------------
    # 1. Compile/check scripts
    # ------------------------------------------------------------------
    scripts_to_check = [
        "scripts/build_projected_fill.py",
        "scripts/build_travel_matrix.py",
        "scripts/solve_7day_schedule.py",
        "scripts/solve_daily_routing.py",
        "scripts/run_5day_rolling_horizon.py",
        "scripts/print_summary.py",
    ]

    for script in scripts_to_check:
        fp = root / script
        if fp.exists():
            run_cmd(["python", "-m", "py_compile", script], cwd=root)
        else:
            print(f"[WARN] Skipping compile check because missing: {script}")

    # ------------------------------------------------------------------
    # 2. Build projected fill inputs
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/build_projected_fill.py",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 3. Build travel matrix
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/build_travel_matrix.py",
            "--depot-lat",
            "37.868998",
            "--depot-lng",
            "-122.264800",
            "--network-type",
            "drive_service",
            "--routing-direction",
            "undirected",
            "--query-margin-m",
            "1500",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 4. Run bounded 7-day weekly model
    #
    # Main planning target:
    # - all routable bins, expected around 232
    # - 3 trucks
    # - 480-minute truck-day
    # - weekly pickups between 250 and 300
    # - Day 0 pickups between 45 and 55 to improve routing feasibility
    #
    # Important:
    # Current solve_7day_schedule.py treats overflow as a soft penalty.
    # It does not hard-force zero overflow.
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/solve_7day_schedule.py",
            "--num-trucks",
            "3",
            "--truck-work-min",
            "480",
            "--max-overtime-min",
            "300",
            "--overtime-penalty-per-min",
            "0.02",
            "--tiny-pickup-threshold-gal",
            "5",
            "--tiny-pickup-penalty",
            "0",
            "--overflow-penalty",
            "100",
            "--require-routable",
            "--min-weekly-pickups",
            "250",
            "--max-weekly-pickups",
            "300",
            "--min-day0-pickups",
            "45",
            "--max-day0-pickups",
            "55",
            "--cbc-time-limit-sec",
            "1200",
            "--cbc-gap-rel",
            "0.10",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 5. Run Day-0 routing
    #
    # Increased routing time limit from 90 to 300 seconds because
    # Bottles/Cans previously failed routing under the 90-second setting.
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/solve_daily_routing.py",
            "--truck-work-min",
            "480",
            "--routing-time-limit-sec",
            "300",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 6. Run 5-day rolling horizon
    #
    # Keep this consistent with the final 480-minute truck-day assumption.
    # Routing time limit is also increased to 300 seconds.
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/run_5day_rolling_horizon.py",
            "--num-trucks",
            "3",
            "--truck-work-min",
            "480",
            "--max-overtime-min",
            "300",
            "--tiny-pickup-threshold-gal",
            "10",
            "--require-routable",
            "--routing-time-limit-sec",
            "300",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 7. Print important final output files
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL OUTPUT CHECKS")
    print("=" * 90)

    files_to_print = [
        "data/processed/small_instance_planning_summary.csv",
        "data/processed/daily_route_summary.csv",
        "data/processed/daily_route_plan.csv",
        "data/processed/rolling_horizon_5day/rolling_5day_summary.csv",
        "data/processed/rolling_horizon_5day/rolling_day_metrics.csv",
    ]

    for rel_fp in files_to_print:
        print_file_if_exists(root, rel_fp)

    # ------------------------------------------------------------------
    # 8. Optional print_summary.py
    # ------------------------------------------------------------------
    print_summary_fp = root / "scripts/print_summary.py"

    if print_summary_fp.exists():
        run_cmd(
            [
                "python",
                "scripts/print_summary.py",
            ],
            cwd=root,
        )
    else:
        print("[WARN] scripts/print_summary.py not found, skipping.")

    # ------------------------------------------------------------------
    # 9. Git status
    # ------------------------------------------------------------------
    run_cmd(
        [
            "git",
            "status",
        ],
        cwd=root,
    )

    print("\n" + "=" * 90)
    print("WORKFLOW COMPLETE")
    print("=" * 90)

    print(
        "\nExpected good weekly result:\n"
        "  status = Optimal or Integer Feasible\n"
        "  num_bins_in_instance = 232 or all routable bins available\n"
        "  num_trucks = 3\n"
        "  truck_work_min_input = 480\n"
        "  effective_truck_work_min = 480\n"
        "  resource_model = single_shift_truck_day\n"
        "  use_observed_shift_span = False\n"
        "  require_routable = True\n"
        "  total_pickups between 250 and 300\n"
        "  Day 0 pickups between 45 and 55\n"
        "  total_overtime_min low or acceptable\n"
        "  daily routing should complete without dropped/failed stream routes\n"
        "\nImportant interpretation:\n"
        "  This is a bounded weekly service scenario.\n"
        "  Pickup range is an in-model policy constraint, not post-optimization forcing.\n"
        "  Overflow is penalized but not hard-forbidden in the current solve_7day_schedule.py.\n"
    )


if __name__ == "__main__":
    main()