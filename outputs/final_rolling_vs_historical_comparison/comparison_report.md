# Rolling Horizon vs Historical Collection Comparison

## Scope

This comparison evaluates the rolling-horizon model output against historical collection activity.

- Rolling days compared: 2
- Historical dates used: 2026-02-04, 2026-02-05
- Planning horizon per rolling solve: 7 calendar days
- Service days per rolling solve: 5 weekday service days
- Travel-time calibration target: 8-minute average non-depot inter-stop travel time
- Historical route minutes are a proxy: historical pickups × 12.0 minutes

## Overall Results

- Total model pickups: 108
- Total historical pickups: 181
- Difference, model minus historical: -73
- Model / historical pickup ratio: 0.60
- Total model route minutes: 749.0
- Total historical route-minutes proxy: 2172.0
- Difference, model minus historical route-minutes proxy: -1423.0
- Total model pickup gallons: 4432.86
- Total model pickup pounds: 4160.64
- Capacity/time violations: 0
- Dropped stops: 0
- Missing routed serials: 0
- Extra routed serials: 0

## Interpretation Notes

The rolling-horizon model is not expected to exactly match historical operations. The model optimizes collection based on projected fill levels, overflow penalties, compost hygiene, truck capacity, and routing feasibility. Historical operations may reflect additional constraints that are not observed in the dataset, such as driver judgment, real-time requests, campus traffic, access restrictions, staff availability, special events, or manual prioritization.

The historical route-minutes value is only a proxy because actual driver route times are not directly observed. It uses the same planning-stage assumption of 4 minutes service time plus 8 minutes travel time per pickup.

The most important validation question is whether the model produces a feasible and operationally reasonable dispatch pattern: all scheduled bins are routed, no stops are dropped, no truck exceeds volume or mass capacity, and route times remain within the 480-minute truck-day limit.
