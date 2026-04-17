import pandas as pd

df = pd.read_csv("data/processed/small_instance_service_schedule.csv")
df["pickup_gal"] = pd.to_numeric(df["pickup_gal"], errors="coerce").fillna(0)

print("total_visits:", len(df))
print("zero_pickup_visits:", int((df["pickup_gal"] <= 1e-9).sum()))
print("under_5_gal:", int(((df["pickup_gal"] > 0) & (df["pickup_gal"] < 5)).sum()))
print("under_10_gal:", int(((df["pickup_gal"] > 0) & (df["pickup_gal"] < 10)).sum()))

print("\nSmallest pickups:")
print(
    df.sort_values("pickup_gal")[
        ["Serial", "stream", "service_day", "truck", "pickup_gal", "interval_deadline"]
    ].head(20).to_string(index=False)
)