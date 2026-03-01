## Data Analysis and Forecasting 
### Introduction 
This project analyzes historical Bigbelly smart-bin collection data, to uncover collection patterns, seasonality, and alert behavior across waste streams.The analysis produces cleaned, merged datasets and performance summaries that feed forecasting models and routing/dispatch optimization workflows.

## Data Sources

The analysis integrates multiple datasets:

### Collections Dataset (3 Years)
Historical collection events including:
1.Collection_Time
2.Serial (Bin ID)
3.Stream_Type (Waste, Compostables, Bottles/Cans, Single Stream)
4.Is_Alert
5.Fullness_Pct
6.Lat, Lng

### Asset Account Dataset
1.Bin metadata
2.Stream mapping
3.Operational attributes

### UCB Location Dataset
1.Geographic location of bins
2.Used for spatial validation and routing preparation

## Data Cleaning & Preparation
All preprocessing was performed in:
## code
scripts/run_pipeline.py

## Cleaning Steps:
1.Converted Collection_Time → datetime
2.Removed null timestamps
3.Created time features:
    1.date
    2.week
    3.month
    4.day_of_week
    5.month_name
    6.Filled missing:
    7.Is_Alert → False
    8.Stream_Type → "Unknown"
4.Generated daily aggregated dataset:
    data/processed/daily_counts_by_stream.parquet
Time series were completed by filling missing dates with 0 collections to support forecasting.

## Exploratory Data Analysis (EDA)
All visualizations generated using:
scripts/make_viz.py
### Libraries used:
1.pandas
2.matplotlib
3.rolling averages (7-day, 30-day smoothing)

## Key EDA Findings
### Alert Rate by Day of Week
## Highest alert rate: Wednesday
Lowest alert rate: Weekend (Saturday & Sunday)
Inference:
Midweek operational pressure is higher. Route balancing or staffing adjustments may improve efficiency.

### Daily Collections (Overall)
Observations:
Clear upward trend (2023 → 2026)
Strong seasonal cycles
High daily volatility
7-day and 30-day smoothing reveal structure
Inference:
Time-series modeling is appropriate due to:
Trend
Weekly seasonality
Annual seasonality

## Collections by Stream
### Waste
Largest volume contributor
Strong seasonal pattern
Drives operational workload

### Compostables
Highest volatility
Large amplitude seasonal swings
Influenced by academic calendar

### Bottles/Cans
Moderate growth
More stable pattern

### Single Stream
Very low volume
Minimal operational impact

### Operational Insight:
Routing optimization should prioritize:
Waste
Compostables
Bottles/Cans

## Monthly Timeline (All Bins)
Strong annual cycles
Peaks in Fall and Spring
Lower volumes during summer/winter breaks

Inference:
Collections are correlated with:
    1.Academic calendar
    2.Campus activity levels
Seasonal planning is essential.

## Daily Alert Rate Over Time
Generally low baseline alert rate
Occasional spikes (anomaly days)
Some days exceed 40–50%

Inference:
Dispatch system must handle:
    1.Rare surge events
    2.Event-driven waste spikes

## Fullness % Distribution
Discrete values: 0, 20, 40, 60, 80, 100
Most common: 60% and 80%
Inference:
    1.Sensor data is banded (not continuous).
    2.Alerts likely triggered near 80–100%.

scripts/
│
├── run_pipeline.py
├── make_viz.py

data/
├── raw/
├── interim/
└── processed/
outputs/
├── eda/

Key Takeaways

1. Demand is increasing year-over-year
2. Strong weekly + annual seasonality
3. Compostables drive volatility
4. Waste drives operational load
5. Midweek alert concentration
6. Sensor readings are discrete bands