# 1) rebuild processed data
python scripts/run_pipeline.py

# 2) rebuild projected fill inputs
python scripts/build_projected_fill.py

# 3) build / refresh travel matrix
python scripts/build_travel_matrix.py --depot-lat 37.868998 --depot-lng -122.264800

# 4) test the heuristic 7-day planner alone
python scripts/solve_7day_schedule_heuristic.py --num-trucks 3 --truck-work-min 480 --max-overtime-min 180 --tiny-pickup-threshold-gal 5 --require-routable

# 5) test daily routing alone on that planner output
python scripts/solve_daily_routing.py --truck-work-min 480

# 6) run the 5-day rolling-horizon experiment
python scripts/run_5day_rolling_horizon.py --num-trucks 3 --truck-work-min 480 --max-overtime-min 180 --tiny-pickup-threshold-gal 5 --require-routable

# 7) inspect final 5-day outputs
cat data/processed/rolling_horizon_5day/rolling_5day_summary.csv
cat data/processed/rolling_horizon_5day/rolling_day_metrics.csv

python scripts/check_tiny_pickups.py
python scripts/evaluate_week_against_actual.py --actual-start-date 2024-02-05 --actual-end-date 2024-02-11
python scripts/run_sensitivity_study.py --max-bins 100 --require-routable