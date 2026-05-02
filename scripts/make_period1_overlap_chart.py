from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "period1_reoptimization_comparison"
OUTPUT_DIR = ROOT / "outputs" / "final_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_fp = INPUT_DIR / "period1_reoptimization_summary.csv"
details_fp = INPUT_DIR / "period1_reoptimization_details.csv"

if not summary_fp.exists():
    raise FileNotFoundError(f"Missing file: {summary_fp}")

summary = pd.read_csv(summary_fp)

row = summary.iloc[0]

original_total = int(row["original_period1_bins"])
rolling_total = int(row["rolling_day2_bins"])
unchanged = int(row["unchanged_bins"])
removed = int(row["removed_from_period1_after_resolve"])
added = int(row["added_to_period1_after_resolve"])
jaccard = float(row["jaccard_similarity"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect("equal")
ax.axis("off")

# Circle positions
left_center = (0.42, 0.5)
right_center = (0.62, 0.5)
radius = 0.23

left_circle = Circle(left_center, radius, fill=False, linewidth=2)
right_circle = Circle(right_center, radius, fill=False, linewidth=2)

ax.add_patch(left_circle)
ax.add_patch(right_circle)

# Labels above circles
ax.text(
    left_center[0],
    0.80,
    f"Original Plan\nPeriod 1\nTotal = {original_total}",
    ha="center",
    va="center",
    fontsize=12,
)

ax.text(
    right_center[0],
    0.80,
    f"Re-solved Plan\nRolling Day 2\nTotal = {rolling_total}",
    ha="center",
    va="center",
    fontsize=12,
)

# Region counts
ax.text(0.31, 0.50, f"{removed}\nRemoved", ha="center", va="center", fontsize=13)
ax.text(0.52, 0.50, f"{unchanged}\nUnchanged", ha="center", va="center", fontsize=13)
ax.text(0.73, 0.50, f"{added}\nAdded", ha="center", va="center", fontsize=13)

# Title
ax.set_title(
    "Period-1 Re-optimization Overlap",
    fontsize=15,
    fontweight="bold",
    pad=20,
)

# Bottom annotation
annotation = (
    f"Jaccard similarity = {jaccard:.3f}\n"
    "Comparison: original 7-day plan service_day = 1 vs re-solved Rolling Day 2 Day 0 set"
)
ax.text(0.5, 0.13, annotation, ha="center", va="center", fontsize=11)

out_fp = OUTPUT_DIR / "08_period1_reoptimization_overlap_chart.png"
plt.tight_layout()
plt.savefig(out_fp, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Wrote: {out_fp}")