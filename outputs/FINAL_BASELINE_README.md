# Final Frozen Baseline Before Rolling Horizon

This snapshot preserves the validated baseline before running the 5-day rolling horizon.

## Final 7-calendar-day / 5-service-day planning result

- Solver status: Optimal
- Objective value: 2499.13
- Horizon: 7 calendar days
- Service days: Days 0-4
- Non-service days: Days 5-6
- Bins in instance: 210
- Trucks: 4
- Truck work limit: 480 minutes
- Total pickups: 242
- Total extra dumps: 2
- Total overtime: 0
- Overflow bin-days: 27
- Overflow cost: $540
- Compost bins serviced within weekday window: 74/74
- Capacity/time violations: 0
- Weekday service-only satisfied: True
- Overflow cost consistency check: True

## Final scaled Day 0 routing result

- Scheduled Day 0 bins: 41
- Routed Day 0 bins: 41
- Dropped stops: 0
- Missing scheduled bins: 0
- Extra routed bins: 0
- Volume violations: 0
- Mass violations: 0
- Time violations: 0
- Day 0 routing feasible: True

## Scaled Day 0 route times

- Bottles/Cans, T2: 14 stops, 407 gal, 123 lb, 109 min
- Compostables, T3: 10 stops, 500 gal, 601 lb, 70 min
- Waste, T1: 11 stops, 499 gal, 499 lb, 95 min
- Waste, T4: 6 stops, 497 gal, 497 lb, 37 min

## Notes

The Day 0 travel matrix was scaled so the average non-depot inter-stop travel time is closer to the 8-minute planning assumption. These route times are calibrated approximations, not observed driver shift durations.
