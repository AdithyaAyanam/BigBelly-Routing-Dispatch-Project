from __future__ import annotations

"""
run_full_bigbelly_workflow.py
-------------------------------------------------------------
Runs the full Bigbelly workflow in order:

1. Compile/check scripts
2. Build projected fill inputs
3. Build travel matrix
4. Run relaxed 7-day weekly optimization
5. Run Day-0 routing
6. Run 5-day rolling horizon
7. Print final summaries
8. Show git status

Run from project root:
    python scripts/run_full_bigbelly_workflow.py
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
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


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
        ["python", "scripts/build_projected_fill.py"],
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
    # 4. Run final relaxed 7-day weekly model
    #    Main operational result, no forced pickup bounds.
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
            "180",
            "--tiny-pickup-threshold-gal",
            "5",
            "--require-routable",
            "--cbc-time-limit-sec",
            "600",
            "--cbc-gap-rel",
            "0.10",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 5. Run Day-0 routing
    # ------------------------------------------------------------------
    run_cmd(
        [
            "python",
            "scripts/solve_daily_routing.py",
            "--truck-work-min",
            "480",
            "--routing-time-limit-sec",
            "90",
        ],
        cwd=root,
    )

    # ------------------------------------------------------------------
    # 6. Run 5-day rolling horizon
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
            "90",
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
        "data/processed/rolling_horizon_5day/rolling_5day_summary.csv",
        "data/processed/rolling_horizon_5day/rolling_day_metrics.csv",
    ]

    for rel_fp in files_to_print:
        fp = root / rel_fp
        print("\n" + "-" * 90)
        print(rel_fp)
        print("-" * 90)

        if fp.exists():
            print(fp.read_text())
        else:
            print(f"[WARN] Missing file: {rel_fp}")

    # ------------------------------------------------------------------
    # 8. Optional print_summary.py
    # ------------------------------------------------------------------
    print_summary_fp = root / "scripts/print_summary.py"
    if print_summary_fp.exists():
        run_cmd(["python", "scripts/print_summary.py"], cwd=root)
    else:
        print("[WARN] scripts/print_summary.py not found, skipping.")

    # ------------------------------------------------------------------
    # 9. Git status
    # ------------------------------------------------------------------
    run_cmd(["git", "status"], cwd=root)

    print("\n" + "=" * 90)
    print("WORKFLOW COMPLETE")
    print("=" * 90)
    print(
        "\nExpected good rolling-horizon result should be close to:\n"
        "  total_pickups = 249\n"
        "  total_routes = 15\n"
        "  total_route_minutes = 1474\n"
        "  total_overtime = 19\n"
        "  avg_overflow_bins_start_of_day = 1.8\n"
    )


if __name__ == "__main__":
    main()