python scripts/run_pipeline.py
python scripts/build_projected_fill.py
python scripts/solve_7day_schedule.py --num-trucks 3 --truck-work-min 480 --max-bins 100 --max-overtime-min 180 --overflow-penalty 5 --tiny-pickup-threshold-gal 5 --tiny-pickup-penalty 0.25 --cbc-time-limit-sec 180 --cbc-gap-rel 0.10 --require-routable
python scripts/build_travel_matrix.py --depot-lat 37.868998 --depot-lng -122.264800
python scripts/solve_daily_routing.py
python scripts/evaluate_week_against_actual.py --actual-start-date 2024-02-05 --actual-end-date 2024-02-11
python scripts/check_tiny_pickups.py
python scripts/run_sensitivity_study.py --max-bins 100 --require-routable