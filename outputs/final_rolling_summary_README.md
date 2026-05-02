# Final Rolling Horizon and Historical Comparison Results

## Rolling Horizon Scope

- Rolling days executed: 2
- Planning horizon per solve: 7 calendar days
- Service days per solve: 5 weekday service days
- Travel matrix calibration: average non-depot inter-stop travel time scaled to 8 minutes
- Trucks: 4
- Truck work limit: 480 minutes
- Overtime: 0

## 2-Day Rolling Horizon Validation

- Total pickups: 108
- Total pickup gallons: 4,432.86
- Total pickup pounds: 4,160.64
- Total routes: 8
- Total route minutes: 749
- Total overtime: 0
- Capacity/time violations: 0
- Dropped stops: 0
- Missing routed serials: 0
- Extra routed serials: 0
- Routing feasible all days: True
- Rolling horizon valid: True

## Historical Comparison

Historical comparison dates:
- 2026-02-04
- 2026-02-05

Model vs historical:
- Model pickups: 108
- Historical pickups: 181
- Difference: -73
- Model/historical pickup ratio: 0.5967
- Model route minutes: 749
- Historical route-minutes proxy: 2,172
- Difference: -1,423
- Capacity/time violations: 0
- Dropped stops: 0

## Interpretation

The rolling-horizon model produced fewer pickups than historical operations while maintaining routing feasibility and eliminating start-of-day overflow by the second rolling day. Historical route minutes are a proxy calculated as historical pickups multiplied by 12 minutes, using the same planning assumption of 4 minutes service time plus 8 minutes travel time per pickup.
