# 📋 KPI FUNCTIONS QUICK REFERENCE CARD

## 🚀 QUICK START

```python
# Import the library
import pandas as pd
from ev_kpi_functions import *

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate ALL KPIs at once (easiest way!)
df = calculate_all_kpis(df, current_year=2024)

# Save result
df.to_csv('data_with_kpis.csv', index=False)
```

---

## 📊 PRIMARY KPI FUNCTIONS (Must Use)

### 1. **calculate_ev_adoption_rate()**
**What it does:** Calculates % of vehicles that are EVs  
**Formula:** `(EV / (EV + ICE)) × 100`

```python
df = calculate_ev_adoption_rate(
    df,
    ev_col='ev_vehicle_registrations',
    ice_col='ice_vehicle_registrations',
    output_col='ev_adoption_rate'
)
```

**Returns:** Adds columns:
- `total_registrations` 
- `ev_adoption_rate`

**Use in dashboard:** Maps, trend lines, state rankings

---

### 2. **calculate_ice_to_ev_conversion_rate()**
**What it does:** Measures speed of ICE→EV transition  
**Formula:** `((EV_current - EV_previous) / ICE_previous) × 100`

```python
df = calculate_ice_to_ev_conversion_rate(
    df,
    ev_col='ev_vehicle_registrations',
    ice_col='ice_vehicle_registrations',
    groupby_cols=['state', 'vehicle_segment'],
    output_col='ice_to_ev_conversion_rate'
)
```

**Returns:** Adds columns:
- `prev_year_ev`
- `prev_year_ice`
- `ice_to_ev_conversion_rate`

**Use in dashboard:** Waterfall charts, state comparisons

---

### 3. **calculate_yoy_growth_rate()**
**What it does:** Year-over-year % change  
**Formula:** `((Value_t - Value_t-1) / Value_t-1) × 100`

```python
df = calculate_yoy_growth_rate(
    df,
    value_col='ev_vehicle_registrations',  # Can be any column!
    groupby_cols=['state', 'vehicle_segment'],
    output_col='yoy_growth_rate'
)
```

**Returns:** Adds column:
- `yoy_growth_rate`

**Use in dashboard:** Bar charts, trend analysis

---

### 4. **calculate_market_share_by_segment()**
**What it does:** % share of 2W/3W/4W in total EVs  
**Formula:** `(Segment_EV / Total_EV) × 100`

```python
df = calculate_market_share_by_segment(
    df,
    segment_col='vehicle_segment',
    ev_col='ev_vehicle_registrations',
    groupby_cols=['state', 'year'],
    output_col='segment_market_share'
)
```

**Returns:** Adds column:
- `segment_market_share`

**Use in dashboard:** Pie charts, stacked area charts

---

### 5. **calculate_infrastructure_adequacy_ratio()**
**What it does:** Charging stations per 1000 EVs  
**Formula:** `(Stations / EVs) × 1000`

```python
df = calculate_infrastructure_adequacy_ratio(
    df,
    stations_col='charging_stations',
    ev_col='ev_vehicle_registrations',
    output_col='stations_per_1000_evs',
    benchmark=10.0  # International standard
)
```

**Returns:** Adds columns:
- `stations_per_1000_evs`
- `infrastructure_gap`
- `infrastructure_status` (categorical)

**Use in dashboard:** Heatmaps, scatter plots

---

### 6. **calculate_policy_effectiveness_score()**
**What it does:** Composite policy strength + ROI  
**Formula:** 
- `Score = (Subsidy/10000) + (Tax×2) + (Reg×2)`
- `ROI = Adoption_Rate / Score`

```python
df = calculate_policy_effectiveness_score(
    df,
    subsidy_col='avg_ev_subsidy_rs',
    tax_exemption_col='road_tax_exemption',
    reg_waiver_col='registration_fee_waiver',
    adoption_col='ev_adoption_rate',
    output_score_col='policy_score',
    output_roi_col='policy_roi'
)
```

**Returns:** Adds columns:
- `policy_score`
- `policy_roi`
- `policy_strength_category`

**Use in dashboard:** Bubble charts, policy comparison

---

### 7. **predict_future_ev_share()**
**What it does:** Forecasts EV adoption for 2025-2027  
**Method:** Trend extrapolation using historical growth

```python
predictions_df = predict_future_ev_share(
    df,
    current_year=2024,
    forecast_years=[2025, 2026, 2027],
    state_col='state',
    segment_col='vehicle_segment',
    adoption_col='ev_adoption_rate',
    growth_col='yoy_growth_rate',
    output_col='predicted_ev_share'
)
```

**Returns:** New dataframe with:
- `state`, `segment`, `year`
- `predicted_ev_share`
- `prediction_method`
- `confidence_level`

**Use in dashboard:** Forecast lines, prediction maps

---

## 📈 SECONDARY KPI FUNCTIONS (Optional)

### 8. **calculate_cagr()**
**What it does:** Compound Annual Growth Rate  
**Formula:** `((End/Start)^(1/years) - 1) × 100`

```python
cagr_df = calculate_cagr(
    df,
    value_col='ev_vehicle_registrations',
    groupby_cols=['state', 'vehicle_segment'],
    start_year=2016,
    end_year=2024,
    output_col='cagr'
)
```

**Use:** Long-term trend analysis

---

### 9. **calculate_fast_charging_availability()**
**What it does:** Quality of infrastructure  
**Formula:** `Stations × Fast_Charger_Percentage`

```python
df = calculate_fast_charging_availability(
    df,
    stations_col='charging_stations',
    fast_pct_col='fast_charger_pct',
    output_col='fast_charging_score'
)
```

**Use:** Infrastructure quality assessment

---

### 10. **calculate_economic_viability_index()**
**What it does:** Economic favorability for EVs  
**Formula:** `(Income_Index × 100) / Fuel_Price`

```python
df = calculate_economic_viability_index(
    df,
    income_col='avg_income_index',
    fuel_price_col='fuel_price_rs_per_litre',
    output_col='economic_viability_index'
)
```

**Use:** Affordability analysis

---

### 11. **calculate_policy_maturity()**
**What it does:** Years since policy started  
**Formula:** `Current_Year - Policy_Start_Year`

```python
df = calculate_policy_maturity(
    df,
    current_year=2024,
    policy_start_col='policy_start_year',
    output_col='policy_age_years'
)
```

**Use:** Policy effectiveness over time

---

## 🎯 UTILITY FUNCTIONS

### **calculate_all_kpis()** ⭐ RECOMMENDED
**What it does:** Calculates ALL KPIs in one call!

```python
df = calculate_all_kpis(
    df,
    current_year=2024,
    groupby_cols=['state', 'vehicle_segment']
)
```

**This runs all 11 KPI functions automatically!**

---

### **summarize_kpis_by_state()**
**What it does:** Creates state ranking table

```python
summary = summarize_kpis_by_state(
    df,
    year=2024,
    top_n=10
)
```

**Returns:** Top N states with key metrics

---

### **export_kpis_for_dashboard()**
**What it does:** Exports multiple CSV files for dashboards

```python
export_kpis_for_dashboard(
    df,
    output_prefix='dashboard'
)
```

**Creates:**
- `dashboard_state_year.csv`
- `dashboard_segment_year.csv`
- `dashboard_latest_2024.csv`

---

## 🔧 COMMON PARAMETERS

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `df` | Input dataframe | Required | Must have relevant columns |
| `groupby_cols` | Columns to group by | `['state', 'vehicle_segment']` | For time-series KPIs |
| `output_col` | Name of output column | Varies | Customize as needed |
| `current_year` | Most recent year | `2024` | For predictions |

---

## 📝 TYPICAL WORKFLOW

```python
# Step 1: Load and merge data
df = pd.read_csv('india_ev_ice_adoption_large.csv')
infra = pd.read_csv('ev_charging_infrastructure_india.csv')
policy = pd.read_csv('ev_policy_incentives_india.csv')

df = df.merge(infra[['state', 'year', 'fast_charger_pct', 'urban_coverage_pct']], 
              on=['state', 'year'], how='left')
df = df.merge(policy[['state', 'policy_start_year', 'road_tax_exemption', 
                       'registration_fee_waiver']], 
              on='state', how='left')

# Step 2: Calculate all KPIs
df = calculate_all_kpis(df, current_year=2024)

# Step 3: Generate predictions
predictions = predict_future_ev_share(
    df, 
    current_year=2024, 
    forecast_years=[2025, 2026, 2027]
)

# Step 4: Create summaries
state_summary = summarize_kpis_by_state(df, year=2024, top_n=15)

# Step 5: Export for dashboard
export_kpis_for_dashboard(df, output_prefix='tableau_ready')
predictions.to_csv('predictions.csv', index=False)
state_summary.to_csv('state_rankings.csv', index=False)
```

---

## ⚠️ COMMON ERRORS & FIXES

### Error: `KeyError: 'column_name'`
**Cause:** Missing required column  
**Fix:** Ensure all required columns exist in your dataframe

### Error: `TypeError: unsupported operand type`
**Cause:** Column has wrong data type  
**Fix:** Convert to numeric: `df['col'] = pd.to_numeric(df['col'])`

### Error: Division by zero warnings
**Cause:** Some rows have 0 values  
**Fix:** Already handled in functions with `.fillna(0)` and `.replace(0, 0.1)`

### First year has NaN for growth rates
**Cause:** No previous year to compare  
**Fix:** This is expected - filter or fill: `df = df[df['year'] > 2016]`

---

## 💡 PRO TIPS

1. **Always use `calculate_all_kpis()` first** - easiest way to get everything
2. **Check for missing values** after merging datasets
3. **Sort by year** before time-series calculations
4. **Round decimals** for readability: `df.round(2)`
5. **Filter to latest year** for snapshot views: `df[df['year'] == 2024]`
6. **Group and aggregate** for summary stats: `df.groupby('state').mean()`
7. **Use `.nlargest()` and `.nsmallest()`** for top/bottom N
8. **Export to CSV** with: `df.to_csv('file.csv', index=False)`

---

## 📚 FUNCTION DEPENDENCY TREE

```
calculate_all_kpis()
├── calculate_ev_adoption_rate()          [Runs first - creates base metrics]
├── calculate_ice_to_ev_conversion_rate() [Needs year sorting]
├── calculate_yoy_growth_rate()           [Needs year sorting]
├── calculate_market_share_by_segment()   [Independent]
├── calculate_infrastructure_adequacy_ratio() [Independent]
├── calculate_policy_effectiveness_score()    [Needs ev_adoption_rate]
├── calculate_fast_charging_availability()    [Optional - if fast_pct exists]
├── calculate_economic_viability_index()      [Optional - if income exists]
└── calculate_policy_maturity()               [Optional - if policy_start exists]
```

---

## 🎨 DASHBOARD VISUALIZATION MAPPING

| KPI | Best Visualization | Why |
|-----|-------------------|-----|
| EV Adoption Rate | Line chart, Map | Show trends and geography |
| Conversion Rate | Waterfall, Heatmap | Show transition dynamics |
| YoY Growth | Bar chart | Compare states/segments |
| Market Share | Pie, Stacked area | Show composition |
| Infrastructure Adequacy | Scatter, Heatmap | Correlate with adoption |
| Policy Score | Bubble chart | Multi-variable comparison |
| Predictions | Line with forecast | Show future scenarios |

---

## 📞 NEED HELP?

**Check column names:** `print(df.columns)`  
**Check data types:** `print(df.dtypes)`  
**Check for NaN:** `print(df.isnull().sum())`  
**View sample:** `print(df.head())`  
**Test on subset:** `df_test = df.head(1000)`

---

**Happy analyzing! 🚀**
