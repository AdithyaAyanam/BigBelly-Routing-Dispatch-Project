# Period 1 Re-optimization Comparison

This comparison addresses Professor Yano's request to show how the planned service set for period 1 changes after rolling the horizon forward.

## Comparison

- Original plan: 7-day plan over periods 0-6, using service_day == 1
- Rolling re-solve: second rolling solve over shifted periods 1-7, using Rolling Day 2 Day 0 execution set

## Results

- Original period-1 bins: 60
- Re-solved Rolling Day 2 bins: 52
- Bins unchanged in both plans: 35
- Bins removed after re-solving: 25
- Bins newly added after re-solving: 17
- Jaccard similarity: 0.455

## Interpretation

The next-day service set changed after executing period 0, updating bin states, and shifting the planning window from periods 0-6 to periods 1-7. This supports the rolling-horizon approach: future-day schedules should remain provisional, and only the current Day 0 selections should be routed and executed.
