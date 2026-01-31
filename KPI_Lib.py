"""
EV ADOPTION KPI FUNCTIONS LIBRARY
==================================
Comprehensive collection of functions to calculate all Key Performance Indicators
for India's EV Adoption and ICE-to-EV Transition Analysis.

Author: Data Science Case Study
Date: January 2026
Version: 1.0

Dependencies:
    - pandas
    - numpy

Usage:
    import pandas as pd
    from ev_kpi_functions import *
    
    df = pd.read_csv('master_dataset.csv')
    df = calculate_all_kpis(df)

Functions included:
PRIMARY KPIs (7):

calculate_ev_adoption_rate() - Market penetration %
calculate_ice_to_ev_conversion_rate() - Transition speed
calculate_yoy_growth_rate() - Annual growth momentum
calculate_market_share_by_segment() - 2W/3W/4W distribution
calculate_infrastructure_adequacy_ratio() - Stations per 1000 EVs
calculate_policy_effectiveness_score() - Policy strength & ROI
predict_future_ev_share() - 2025-2027 forecasts

SECONDARY KPIs (4):
8. calculate_cagr() - Long-term compound growth
9. calculate_fast_charging_availability() - Infrastructure quality
10. calculate_economic_viability_index() - Affordability metric
11. calculate_policy_maturity() - Years since policy started
UTILITY FUNCTIONS:

calculate_all_kpis() - ONE function to calculate everything! ⭐
summarize_kpis_by_state() - Create ranking tables
export_kpis_for_dashboard() - Export dashboard-ready CSVs
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List


# ==============================================================================
# PRIMARY KPIs (MUST-HAVE)
# ==============================================================================

def calculate_ev_adoption_rate(df: pd.DataFrame, 
                                ev_col: str = 'ev_vehicle_registrations',
                                ice_col: str = 'ice_vehicle_registrations',
                                output_col: str = 'ev_adoption_rate') -> pd.DataFrame:
    """
    Calculate the EV Adoption Rate - the percentage of total vehicle registrations 
    that are electric vehicles.
    
    This is the PRIMARY metric for measuring EV market penetration. It shows what 
    proportion of new vehicles being registered are EVs versus ICE vehicles.
    
    Formula:
        EV Adoption Rate = (EV Registrations / Total Registrations) × 100
        where Total Registrations = EV Registrations + ICE Registrations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing vehicle registration data
    ev_col : str, default='ev_vehicle_registrations'
        Column name containing EV registration counts
    ice_col : str, default='ice_vehicle_registrations'
        Column name containing ICE registration counts
    output_col : str, default='ev_adoption_rate'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - 'total_registrations': Sum of EV and ICE registrations
        - output_col: EV adoption rate as percentage (0-100)
    
    Interpretation:
    ---------------
    - 0-5%: Early adoption stage - strong intervention needed
    - 5-15%: Growth phase - infrastructure critical
    - 15-25%: Acceleration phase - policy refinement needed
    - 25%+: Advanced stage - focus on sustainability
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Mumbai'],
    ...     'ev_vehicle_registrations': [1000, 1500],
    ...     'ice_vehicle_registrations': [9000, 8500]
    ... })
    >>> df = calculate_ev_adoption_rate(df)
    >>> print(df['ev_adoption_rate'])
    0    10.0
    1    15.0
    
    Notes:
    ------
    - Handles division by zero by returning 0 for empty markets
    - Returns percentage (not decimal), so 10% adoption = 10.0
    - Creates intermediate 'total_registrations' column for transparency
    """
    # Calculate total registrations (denominator)
    df['total_registrations'] = df[ev_col] + df[ice_col]
    
    # Calculate EV adoption rate as percentage
    # fillna(0) handles cases where total_registrations might be 0
    df[output_col] = (
        (df[ev_col] / df['total_registrations']) * 100
    ).fillna(0)
    
    return df


def calculate_ice_to_ev_conversion_rate(df: pd.DataFrame,
                                          ev_col: str = 'ev_vehicle_registrations',
                                          ice_col: str = 'ice_vehicle_registrations',
                                          groupby_cols: List[str] = ['state', 'vehicle_segment'],
                                          output_col: str = 'ice_to_ev_conversion_rate') -> pd.DataFrame:
    """
    Calculate the ICE-to-EV Conversion Rate - measures how many ICE vehicle buyers 
    are switching to EVs year-over-year.
    
    This metric shows the SPEED of transition from ICE to EV vehicles. It answers:
    "What percentage of last year's ICE buyers have switched to EV this year?"
    
    Formula:
        Conversion Rate = ((EV_current - EV_previous) / ICE_previous) × 100
    
    This represents the net change in EV registrations as a percentage of the 
    previous year's ICE market size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing vehicle registration data sorted by year
    ev_col : str, default='ev_vehicle_registrations'
        Column name containing EV registration counts
    ice_col : str, default='ice_vehicle_registrations'
        Column name containing ICE registration counts
    groupby_cols : List[str], default=['state', 'vehicle_segment']
        Columns to group by before calculating year-over-year changes
    output_col : str, default='ice_to_ev_conversion_rate'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - 'prev_year_ev': Previous year's EV registrations
        - 'prev_year_ice': Previous year's ICE registrations
        - output_col: Conversion rate as percentage
    
    Interpretation:
    ---------------
    - <2%: Slow conversion - market barriers exist
    - 2-5%: Steady conversion - normal transition pace
    - 5-10%: Rapid conversion - strong market shift
    - >10%: Accelerated conversion - market disruption
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Delhi'],
    ...     'year': [2023, 2024],
    ...     'ev_vehicle_registrations': [1000, 1500],
    ...     'ice_vehicle_registrations': [9000, 8000]
    ... })
    >>> df = calculate_ice_to_ev_conversion_rate(df, groupby_cols=['state'])
    >>> # For 2024: (1500-1000)/9000 * 100 = 5.56%
    
    Notes:
    ------
    - Requires data to be sorted by year within each group
    - First year in each group will have NaN (no previous year data)
    - Negative values indicate EV market contraction (rare but possible)
    - Uses shift(1) to get previous year values within each group
    """
    # Ensure data is sorted by year within groups
    df = df.sort_values(groupby_cols + ['year'])
    
    # Calculate previous year values using shift
    # shift(1) moves values down by 1 row within each group
    df['prev_year_ev'] = df.groupby(groupby_cols)[ev_col].shift(1)
    df['prev_year_ice'] = df.groupby(groupby_cols)[ice_col].shift(1)
    
    # Calculate the conversion rate
    # (Change in EV registrations) / (Previous year's ICE base) × 100
    df[output_col] = (
        ((df[ev_col] - df['prev_year_ev']) / (df['prev_year_ice'] + 1)) * 100
    ).fillna(0)  # fillna(0) for first year in each group
    
    # Note: Adding 1 to denominator prevents division by zero
    # This is a small adjustment that doesn't materially affect results
    
    return df


def calculate_yoy_growth_rate(df: pd.DataFrame,
                               value_col: str = 'ev_vehicle_registrations',
                               groupby_cols: List[str] = ['state', 'vehicle_segment'],
                               output_col: str = 'yoy_growth_rate') -> pd.DataFrame:
    """
    Calculate Year-over-Year (YoY) Growth Rate for any metric.
    
    This measures the percentage change in a metric from one year to the next.
    Most commonly used for EV registrations to show market momentum.
    
    Formula:
        YoY Growth = ((Value_current - Value_previous) / Value_previous) × 100
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing time-series data
    value_col : str, default='ev_vehicle_registrations'
        Column name for which to calculate growth rate
    groupby_cols : List[str], default=['state', 'vehicle_segment']
        Columns to group by before calculating year-over-year changes
    output_col : str, default='yoy_growth_rate'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added column containing YoY growth rate percentage
    
    Interpretation:
    ---------------
    - <0%: Decline (market contraction)
    - 0-20%: Slow growth (mature market)
    - 20-50%: Healthy growth (normal EV adoption)
    - 50-100%: Rapid growth (early adopter phase)
    - >100%: Explosive growth (typical with small base)
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Delhi'],
    ...     'year': [2023, 2024],
    ...     'ev_vehicle_registrations': [1000, 1500]
    ... })
    >>> df = calculate_yoy_growth_rate(df, groupby_cols=['state'])
    >>> # For 2024: (1500-1000)/1000 * 100 = 50%
    
    Notes:
    ------
    - Can be applied to any numeric column (EVs, ICEs, charging stations, etc.)
    - First year in each group will have NaN
    - Very high growth rates (>500%) often indicate starting from a very small base
    - Negative growth indicates market decline
    """
    # Ensure data is sorted by year
    df = df.sort_values(groupby_cols + ['year'])
    
    # Get previous year value
    prev_col = f'prev_{value_col}'
    df[prev_col] = df.groupby(groupby_cols)[value_col].shift(1)
    
    # Calculate YoY growth rate
    df[output_col] = (
        ((df[value_col] - df[prev_col]) / (df[prev_col] + 1)) * 100
    ).fillna(0)
    
    # Drop temporary column to keep dataframe clean
    df = df.drop(columns=[prev_col])
    
    return df


def calculate_market_share_by_segment(df: pd.DataFrame,
                                       segment_col: str = 'vehicle_segment',
                                       ev_col: str = 'ev_vehicle_registrations',
                                       groupby_cols: List[str] = ['state', 'year'],
                                       output_col: str = 'segment_market_share') -> pd.DataFrame:
    """
    Calculate Market Share by Vehicle Segment (2W, 3W, 4W).
    
    This shows what percentage of total EV registrations comes from each vehicle 
    segment. Helps identify which vehicle types are driving EV adoption.
    
    Formula:
        Segment Market Share = (Segment EV Registrations / Total EV Registrations) × 100
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing segment-level data
    segment_col : str, default='vehicle_segment'
        Column name containing segment identifiers (2W, 3W, 4W)
    ev_col : str, default='ev_vehicle_registrations'
        Column name containing EV registration counts
    groupby_cols : List[str], default=['state', 'year']
        Columns defining the aggregation level (usually state and/or year)
    output_col : str, default='segment_market_share'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added column containing segment market share percentage
    
    Interpretation:
    ---------------
    Typical patterns in India:
    - 2W (Two-wheelers): 60-70% - Most affordable, highest adoption
    - 3W (Three-wheelers): 20-30% - Commercial use, good economics
    - 4W (Four-wheelers): 5-15% - Premium segment, growing
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Delhi', 'Delhi'],
    ...     'year': [2024, 2024, 2024],
    ...     'vehicle_segment': ['2W', '3W', '4W'],
    ...     'ev_vehicle_registrations': [700, 200, 100]
    ... })
    >>> df = calculate_market_share_by_segment(df, groupby_cols=['state', 'year'])
    >>> print(df['segment_market_share'])
    0    70.0  # 2W
    1    20.0  # 3W
    2    10.0  # 4W
    
    Notes:
    ------
    - Market shares within each group sum to 100%
    - Can be calculated at different levels (national, state, yearly)
    - Useful for understanding which segments to prioritize
    - Complements total adoption rate by showing composition
    """
    # Calculate total EV registrations for each group
    # This creates a temporary column with the sum for each group
    df['total_ev_in_group'] = df.groupby(groupby_cols)[ev_col].transform('sum')
    
    # Calculate market share as percentage
    df[output_col] = (
        (df[ev_col] / df['total_ev_in_group']) * 100
    ).fillna(0)
    
    # Remove temporary column
    df = df.drop(columns=['total_ev_in_group'])
    
    return df


def calculate_infrastructure_adequacy_ratio(df: pd.DataFrame,
                                             stations_col: str = 'charging_stations',
                                             ev_col: str = 'ev_vehicle_registrations',
                                             output_col: str = 'stations_per_1000_evs',
                                             benchmark: float = 10.0) -> pd.DataFrame:
    """
    Calculate Infrastructure Adequacy Ratio - charging stations per 1000 EVs.
    
    This metric assesses whether charging infrastructure can support the existing 
    EV fleet. International benchmarks suggest 1 charging station per 10-15 EVs.
    
    Formula:
        Adequacy Ratio = (Charging Stations / EV Registrations) × 1000
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing infrastructure and EV data
    stations_col : str, default='charging_stations'
        Column name containing number of charging stations
    ev_col : str, default='ev_vehicle_registrations'
        Column name containing EV registration counts
    output_col : str, default='stations_per_1000_evs'
        Name for the output column
    benchmark : float, default=10.0
        International benchmark for adequate infrastructure (stations per 1000 EVs)
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - output_col: Stations per 1000 EVs
        - 'infrastructure_gap': Difference from benchmark
        - 'infrastructure_status': Categorical assessment
    
    Interpretation:
    ---------------
    - <5 stations/1000 EVs: Severe shortage - urgent investment needed
    - 5-10: Below standard - infrastructure lagging
    - 10-15: Adequate - meets basic requirements
    - >15: Well-served - infrastructure ahead of demand
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Mumbai'],
    ...     'charging_stations': [500, 300],
    ...     'ev_vehicle_registrations': [50000, 20000]
    ... })
    >>> df = calculate_infrastructure_adequacy_ratio(df)
    >>> print(df['stations_per_1000_evs'])
    0    10.0  # Delhi: 500/50000 * 1000 = adequate
    1    15.0  # Mumbai: 300/20000 * 1000 = well-served
    
    Notes:
    ------
    - Higher values indicate better infrastructure coverage
    - Very high values (>50) may indicate over-investment or low adoption
    - Zero/NaN for areas with no EVs yet (handled with replacement)
    - Can identify priority areas for infrastructure investment
    """
    # Calculate stations per 1000 EVs
    # Divide by EV count, multiply by 1000, handle division by zero
    df[output_col] = (
        df[stations_col] / (df[ev_col] / 1000)
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate gap from benchmark (positive = above benchmark, negative = below)
    df['infrastructure_gap'] = df[output_col] - benchmark
    
    # Create categorical status for easy interpretation
    df['infrastructure_status'] = pd.cut(
        df[output_col],
        bins=[0, 5, 10, 15, np.inf],
        labels=['Severe Shortage', 'Below Standard', 'Adequate', 'Well-Served'],
        include_lowest=True
    )
    
    return df


def calculate_policy_effectiveness_score(df: pd.DataFrame,
                                          subsidy_col: str = 'avg_ev_subsidy_rs',
                                          tax_exemption_col: str = 'road_tax_exemption',
                                          reg_waiver_col: str = 'registration_fee_waiver',
                                          adoption_col: str = 'ev_adoption_rate',
                                          output_score_col: str = 'policy_score',
                                          output_roi_col: str = 'policy_roi') -> pd.DataFrame:
    """
    Calculate Policy Effectiveness Score and Return on Investment (ROI).
    
    This composite metric evaluates the strength of EV policy incentives and 
    measures how effectively they translate into adoption. Helps identify which 
    policy levers are most cost-effective.
    
    Policy Score Formula:
        Score = (Subsidy_Amount / 10000) + (Road_Tax_Exemption × 2) + (Registration_Waiver × 2)
    
    Policy ROI Formula:
        ROI = EV_Adoption_Rate / Policy_Score
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing policy and adoption data
    subsidy_col : str, default='avg_ev_subsidy_rs'
        Column containing average EV subsidy amount in rupees
    tax_exemption_col : str, default='road_tax_exemption'
        Binary column (0/1) indicating road tax exemption
    reg_waiver_col : str, default='registration_fee_waiver'
        Binary column (0/1) indicating registration fee waiver
    adoption_col : str, default='ev_adoption_rate'
        Column containing EV adoption rate percentage
    output_score_col : str, default='policy_score'
        Name for the policy score output column
    output_roi_col : str, default='policy_roi'
        Name for the policy ROI output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - output_score_col: Composite policy strength score (0-12+ scale)
        - output_roi_col: Adoption rate per unit of policy score
        - 'policy_strength_category': Categorical classification
    
    Interpretation:
    ---------------
    Policy Score:
    - 0-2: Weak policy support
    - 2-5: Moderate support
    - 5-8: Strong support
    - >8: Very strong support
    
    Policy ROI:
    - <1: Ineffective - high cost, low adoption
    - 1-2: Below average - reassess policy design
    - 2-5: Good - policies working as intended
    - >5: Excellent - very efficient policies
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['State A', 'State B'],
    ...     'avg_ev_subsidy_rs': [50000, 25000],
    ...     'road_tax_exemption': [1, 0],
    ...     'registration_fee_waiver': [1, 1],
    ...     'ev_adoption_rate': [15.0, 8.0]
    ... })
    >>> df = calculate_policy_effectiveness_score(df)
    >>> # State A: Score = 5 + 2 + 2 = 9, ROI = 15/9 = 1.67
    >>> # State B: Score = 2.5 + 0 + 2 = 4.5, ROI = 8/4.5 = 1.78
    
    Notes:
    ------
    - Subsidies are normalized by dividing by 10,000 to bring to similar scale as binary features
    - Tax exemption and registration waiver weighted at 2× due to their significant impact
    - Higher ROI doesn't always mean better - context matters (urban vs rural, etc.)
    - Zero policy score is set to 0.1 to avoid division by zero in ROI calculation
    """
    # Calculate composite policy score
    # Normalize subsidy to similar scale as binary features (divide by 10,000)
    # Weight exemptions and waivers at 2× their binary value
    df[output_score_col] = (
        (df[subsidy_col] / 10000) +  # Normalize large rupee amounts
        (df[tax_exemption_col] * 2) +  # Weight: binary × 2
        (df[reg_waiver_col] * 2)  # Weight: binary × 2
    ).fillna(0)
    
    # Calculate policy ROI (adoption per unit of policy effort)
    # Replace 0 scores with 0.1 to avoid division by zero
    df[output_roi_col] = (
        df[adoption_col] / df[output_score_col].replace(0, 0.1)
    ).fillna(0)
    
    # Create categorical classification for policy strength
    df['policy_strength_category'] = pd.cut(
        df[output_score_col],
        bins=[0, 2, 5, 8, np.inf],
        labels=['Weak', 'Moderate', 'Strong', 'Very Strong'],
        include_lowest=True
    )
    
    return df


def predict_future_ev_share(df: pd.DataFrame,
                             current_year: int,
                             forecast_years: List[int],
                             state_col: str = 'state',
                             segment_col: str = 'vehicle_segment',
                             adoption_col: str = 'ev_adoption_rate',
                             growth_col: str = 'yoy_growth_rate',
                             output_col: str = 'predicted_ev_share') -> pd.DataFrame:
    """
    Predict Future EV Market Share using trend extrapolation.
    
    This function provides a simple trend-based forecast for EV adoption rates.
    It uses historical growth rates to project future adoption levels. For production
    use, this should be replaced with ML models, but it's useful for quick estimates.
    
    Method:
        Uses compound annual growth rate (CAGR) from recent years to project forward.
        Prediction = Current_Adoption × (1 + Avg_Growth_Rate)^Years_Forward
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing historical adoption data
    current_year : int
        Most recent year in the dataset (e.g., 2024)
    forecast_years : List[int]
        Years to forecast (e.g., [2025, 2026, 2027])
    state_col : str, default='state'
        Column name for state identifier
    segment_col : str, default='vehicle_segment'
        Column name for vehicle segment
    adoption_col : str, default='ev_adoption_rate'
        Column containing current adoption rates
    growth_col : str, default='yoy_growth_rate'
        Column containing year-over-year growth rates
    output_col : str, default='predicted_ev_share'
        Name for the prediction output column
    
    Returns:
    --------
    pd.DataFrame
        New dataframe containing predictions with columns:
        - state_col, segment_col, 'year', output_col
        - 'prediction_method': Indicator of forecasting method used
        - 'confidence_level': Simple confidence indicator
    
    Interpretation:
    ---------------
    - Predictions assume growth trends continue
    - Capped at 100% (cannot exceed total market)
    - Higher uncertainty for longer forecasts
    - Should be combined with scenario analysis
    
    Example:
    --------
    >>> historical_df = pd.DataFrame({
    ...     'state': ['Delhi', 'Delhi', 'Delhi'],
    ...     'year': [2022, 2023, 2024],
    ...     'ev_adoption_rate': [5.0, 8.0, 12.0],
    ...     'yoy_growth_rate': [60.0, 50.0, 50.0]
    ... })
    >>> predictions = predict_future_ev_share(
    ...     historical_df, 
    ...     current_year=2024, 
    ...     forecast_years=[2025, 2026, 2027]
    ... )
    
    Notes:
    ------
    - This is a SIMPLIFIED forecasting method for demonstration
    - For actual case study, use ML models (Random Forest, XGBoost, etc.)
    - Consider adding confidence intervals
    - External factors (policy changes, technology shifts) not captured
    - Use this for quick estimates, not final predictions
    """
    # Get the most recent data for each state-segment combination
    latest_data = df[df['year'] == current_year].copy()
    
    # Calculate average growth rate over recent years (use last 3 years if available)
    recent_data = df[df['year'].isin([current_year-2, current_year-1, current_year])]
    avg_growth = recent_data.groupby([state_col, segment_col])[growth_col].mean().reset_index()
    avg_growth = avg_growth.rename(columns={growth_col: 'avg_growth_rate'})
    
    # Merge average growth with latest data
    latest_data = latest_data.merge(avg_growth, on=[state_col, segment_col], how='left')
    
    # Create predictions for each forecast year
    predictions_list = []
    
    for year in forecast_years:
        year_ahead = year - current_year  # How many years into the future
        
        # Make a copy for this forecast year
        forecast = latest_data[[state_col, segment_col, adoption_col, 'avg_growth_rate']].copy()
        forecast['year'] = year
        
        # Calculate predicted adoption using compound growth
        # Prediction = Current × (1 + growth_rate/100)^years_ahead
        forecast[output_col] = (
            forecast[adoption_col] * 
            ((1 + forecast['avg_growth_rate'] / 100) ** year_ahead)
        )
        
        # Cap at 100% (cannot exceed total market)
        forecast[output_col] = forecast[output_col].clip(upper=100)
        
        # Add metadata
        forecast['prediction_method'] = 'Trend Extrapolation'
        
        # Confidence decreases with time
        if year_ahead == 1:
            forecast['confidence_level'] = 'High'
        elif year_ahead == 2:
            forecast['confidence_level'] = 'Medium'
        else:
            forecast['confidence_level'] = 'Low'
        
        predictions_list.append(forecast)
    
    # Combine all predictions
    predictions_df = pd.concat(predictions_list, ignore_index=True)
    
    # Select and order columns
    output_columns = [state_col, segment_col, 'year', output_col, 
                     'prediction_method', 'confidence_level']
    predictions_df = predictions_df[output_columns]
    
    return predictions_df


# ==============================================================================
# SECONDARY KPIs (IMPORTANT BUT NOT CRITICAL)
# ==============================================================================

def calculate_cagr(df: pd.DataFrame,
                   value_col: str = 'ev_vehicle_registrations',
                   groupby_cols: List[str] = ['state', 'vehicle_segment'],
                   start_year: int = 2016,
                   end_year: int = 2024,
                   output_col: str = 'cagr') -> pd.DataFrame:
    """
    Calculate Compound Annual Growth Rate (CAGR) over a specified period.
    
    CAGR shows the smoothed annual growth rate as if growth had been steady 
    over the entire period. Unlike YoY growth which can be volatile, CAGR 
    shows the overall trend.
    
    Formula:
        CAGR = ((Ending_Value / Beginning_Value)^(1 / Number_of_Years) - 1) × 100
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing time-series data
    value_col : str, default='ev_vehicle_registrations'
        Column for which to calculate CAGR
    groupby_cols : List[str], default=['state', 'vehicle_segment']
        Columns defining separate CAGR calculations
    start_year : int, default=2016
        Beginning year for CAGR calculation
    end_year : int, default=2024
        Ending year for CAGR calculation
    output_col : str, default='cagr'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Aggregated dataframe with one row per group containing CAGR
    
    Interpretation:
    ---------------
    - <10%: Slow long-term growth
    - 10-30%: Moderate growth (typical for established markets)
    - 30-50%: Strong growth (early EV adoption phase)
    - >50%: Exceptional growth (very early stage or breakthrough)
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Delhi', 'Delhi'],
    ...     'year': [2020, 2022, 2024],
    ...     'ev_vehicle_registrations': [100, 225, 506]
    ... })
    >>> cagr_df = calculate_cagr(df, groupby_cols=['state'], 
    ...                           start_year=2020, end_year=2024)
    >>> # CAGR = (506/100)^(1/4) - 1 = 0.50 = 50%
    
    Notes:
    ------
    - More stable than YoY growth (not affected by year-to-year volatility)
    - Useful for comparing growth across different time periods
    - Requires both start and end year data for each group
    - Groups missing either start or end year will have NaN
    """
    # Filter data for start and end years
    start_data = df[df['year'] == start_year][groupby_cols + [value_col]].copy()
    end_data = df[df['year'] == end_year][groupby_cols + [value_col]].copy()
    
    # Rename columns to distinguish start and end values
    start_data = start_data.rename(columns={value_col: 'start_value'})
    end_data = end_data.rename(columns={value_col: 'end_value'})
    
    # Merge start and end values
    cagr_df = start_data.merge(end_data, on=groupby_cols, how='inner')
    
    # Calculate number of years
    num_years = end_year - start_year
    
    # Calculate CAGR
    # Formula: (end/start)^(1/years) - 1, then convert to percentage
    # Add 1 to start_value to avoid division by zero
    cagr_df[output_col] = (
        ((cagr_df['end_value'] / (cagr_df['start_value'] + 1)) ** (1 / num_years) - 1) * 100
    )
    
    # Add metadata
    cagr_df['period'] = f'{start_year}-{end_year}'
    cagr_df['years'] = num_years
    
    # Drop intermediate columns
    cagr_df = cagr_df.drop(columns=['start_value', 'end_value'])
    
    return cagr_df


def calculate_fast_charging_availability(df: pd.DataFrame,
                                          stations_col: str = 'charging_stations',
                                          fast_pct_col: str = 'fast_charger_pct',
                                          output_col: str = 'fast_charging_score') -> pd.DataFrame:
    """
    Calculate Fast Charging Availability Score.
    
    This metric assesses the quality of charging infrastructure by considering 
    both quantity and the percentage of fast chargers. Fast chargers are crucial 
    for EV adoption as they reduce charging time anxiety.
    
    Formula:
        Fast Charging Score = Total_Stations × (Fast_Charger_Percentage / 100)
    
    This gives the equivalent number of fast charging stations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing infrastructure data
    stations_col : str, default='charging_stations'
        Column containing total number of charging stations
    fast_pct_col : str, default='fast_charger_pct'
        Column containing percentage of fast chargers (0-1 or 0-100)
    output_col : str, default='fast_charging_score'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - output_col: Number of equivalent fast charging stations
        - 'fast_charging_category': Quality assessment
    
    Interpretation:
    ---------------
    - Combines quantity (total stations) with quality (% fast)
    - Higher scores indicate better infrastructure
    - Prioritizes states with both high coverage and high quality
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['State A', 'State B'],
    ...     'charging_stations': [1000, 500],
    ...     'fast_charger_pct': [0.30, 0.60]  # 30% and 60%
    ... })
    >>> df = calculate_fast_charging_availability(df)
    >>> print(df['fast_charging_score'])
    0    300  # 1000 × 0.30
    1    300  # 500 × 0.60 (equal score despite fewer stations)
    
    Notes:
    ------
    - Fast chargers typically defined as >50kW DC chargers
    - Fast charging reduces range anxiety significantly
    - Important for long-distance travel and commercial vehicles
    - Should be considered alongside total station count
    """
    # If fast_pct_col is in 0-100 scale, convert to 0-1
    # Check by looking at max value
    if df[fast_pct_col].max() > 1:
        fast_pct = df[fast_pct_col] / 100
    else:
        fast_pct = df[fast_pct_col]
    
    # Calculate fast charging score
    df[output_col] = df[stations_col] * fast_pct
    
    # Create categorical assessment
    # Based on quintiles or fixed thresholds
    df['fast_charging_category'] = pd.cut(
        df[output_col],
        bins=[0, 50, 150, 300, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    
    return df


def calculate_economic_viability_index(df: pd.DataFrame,
                                        income_col: str = 'avg_income_index',
                                        fuel_price_col: str = 'fuel_price_rs_per_litre',
                                        output_col: str = 'economic_viability_index') -> pd.DataFrame:
    """
    Calculate Economic Viability Index for EV adoption.
    
    This index captures the economic favorability of EVs by comparing income levels 
    to fuel costs. Higher fuel prices + higher incomes = more economic sense to 
    switch to EVs.
    
    Formula:
        Economic Viability = (Average_Income_Index × 100) / Fuel_Price_per_Litre
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing economic indicators
    income_col : str, default='avg_income_index'
        Column containing income index (normalized, typically 0-2)
    fuel_price_col : str, default='fuel_price_rs_per_litre'
        Column containing fuel price in rupees per litre
    output_col : str, default='economic_viability_index'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - output_col: Economic viability index
        - 'economic_favorability': Categorical assessment
    
    Interpretation:
    ---------------
    - Higher values = more economically favorable for EVs
    - Considers both affordability (income) and incentive (fuel cost)
    - <0.8: Poor economics for EV adoption
    - 0.8-1.2: Neutral economics
    - >1.2: Favorable economics for EVs
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Rich + High Fuel', 'Poor + Low Fuel'],
    ...     'avg_income_index': [1.5, 0.8],
    ...     'fuel_price_rs_per_litre': [110, 90]
    ... })
    >>> df = calculate_economic_viability_index(df)
    >>> print(df['economic_viability_index'])
    0    1.36  # (1.5 × 100) / 110 = favorable
    1    0.89  # (0.8 × 100) / 90 = less favorable
    
    Notes:
    ------
    - Income index is multiplied by 100 to scale appropriately
    - Higher fuel prices increase the savings from switching to EV
    - Higher incomes increase ability to afford upfront EV cost
    - Doesn't account for electricity prices (future enhancement)
    """
    # Calculate economic viability index
    df[output_col] = (df[income_col] * 100) / df[fuel_price_col]
    
    # Create categorical favorability
    df['economic_favorability'] = pd.cut(
        df[output_col],
        bins=[0, 0.8, 1.2, np.inf],
        labels=['Unfavorable', 'Neutral', 'Favorable'],
        include_lowest=True
    )
    
    # Also calculate fuel-to-income ratio (inverse metric)
    df['fuel_to_income_ratio'] = df[fuel_price_col] / (df[income_col] * 100)
    
    return df


def calculate_policy_maturity(df: pd.DataFrame,
                               current_year: int = 2024,
                               policy_start_col: str = 'policy_start_year',
                               output_col: str = 'policy_age_years') -> pd.DataFrame:
    """
    Calculate Policy Maturity - years since EV policy implementation.
    
    This metric helps understand whether low adoption is due to insufficient time 
    for policies to take effect, or if policies are genuinely ineffective.
    
    Formula:
        Policy Age = Current_Year - Policy_Start_Year
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing policy implementation data
    current_year : int, default=2024
        Reference year for calculating policy age
    policy_start_col : str, default='policy_start_year'
        Column containing the year policy was implemented
    output_col : str, default='policy_age_years'
        Name for the output column
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added columns:
        - output_col: Number of years since policy started
        - 'policy_maturity_stage': Categorical life stage
    
    Interpretation:
    ---------------
    - 0-2 years: Early stage - give time to see effects
    - 3-4 years: Mature - should see measurable impact
    - 5+ years: Established - evaluate for renewal or revision
    
    Example:
    --------
    >>> df = pd.DataFrame({
    ...     'state': ['Delhi', 'Karnataka', 'Bihar'],
    ...     'policy_start_year': [2018, 2022, 2022]
    ... })
    >>> df = calculate_policy_maturity(df, current_year=2024)
    >>> print(df['policy_age_years'])
    0    6  # Delhi - established
    1    2  # Karnataka - early
    2    2  # Bihar - early
    
    Notes:
    ------
    - Newer policies may show low adoption simply due to time lag
    - Older policies with low adoption may need redesign
    - Consider in conjunction with policy effectiveness score
    - NaN for states without policies (filled with 0)
    """
    # Calculate years since policy implementation
    df[output_col] = current_year - df[policy_start_col]
    
    # Handle states without policies (NaN) by setting to 0
    df[output_col] = df[output_col].fillna(0)
    
    # Create maturity stage categories
    df['policy_maturity_stage'] = pd.cut(
        df[output_col],
        bins=[-1, 0, 2, 4, np.inf],
        labels=['No Policy', 'Early Stage', 'Mature', 'Established'],
        include_lowest=True
    )
    
    return df


# ==============================================================================
# COMPOSITE FUNCTION - CALCULATE ALL KPIs AT ONCE
# ==============================================================================

def calculate_all_kpis(df: pd.DataFrame,
                       current_year: int = 2024,
                       groupby_cols: List[str] = ['state', 'vehicle_segment']) -> pd.DataFrame:
    """
    Calculate ALL KPIs in one function call.
    
    This is a convenience function that applies all KPI calculations to your 
    dataset in the correct order. Use this for quick analysis or dashboard prep.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all necessary columns
    current_year : int, default=2024
        Most recent year in the dataset
    groupby_cols : List[str], default=['state', 'vehicle_segment']
        Columns for grouping time-series calculations
    
    Returns:
    --------
    pd.DataFrame
        Original dataframe with ALL KPI columns added
    
    KPIs Calculated:
    ----------------
    PRIMARY:
    1. EV Adoption Rate
    2. ICE-to-EV Conversion Rate
    3. Year-over-Year Growth Rate
    4. Market Share by Segment
    5. Infrastructure Adequacy Ratio
    6. Policy Effectiveness Score & ROI
    
    SECONDARY:
    7. Fast Charging Availability
    8. Economic Viability Index
    9. Policy Maturity
    
    Example:
    --------
    >>> df = pd.read_csv('master_dataset.csv')
    >>> df = calculate_all_kpis(df)
    >>> df.to_csv('dataset_with_all_kpis.csv', index=False)
    
    Notes:
    ------
    - Ensure your dataframe has all required columns before calling
    - Some KPIs depend on others, so order matters (handled internally)
    - This may take a minute on large datasets (100k+ rows)
    - Missing columns will cause errors - validate input first
    """
    print("Calculating all KPIs...")
    
    # PRIMARY KPIs
    print("  ✓ Calculating EV Adoption Rate...")
    df = calculate_ev_adoption_rate(df)
    
    print("  ✓ Calculating ICE-to-EV Conversion Rate...")
    df = calculate_ice_to_ev_conversion_rate(df, groupby_cols=groupby_cols)
    
    print("  ✓ Calculating YoY Growth Rate...")
    df = calculate_yoy_growth_rate(df, groupby_cols=groupby_cols)
    
    print("  ✓ Calculating Market Share by Segment...")
    df = calculate_market_share_by_segment(df)
    
    print("  ✓ Calculating Infrastructure Adequacy...")
    df = calculate_infrastructure_adequacy_ratio(df)
    
    print("  ✓ Calculating Policy Effectiveness...")
    df = calculate_policy_effectiveness_score(df)
    
    # SECONDARY KPIs (if columns exist)
    if 'fast_charger_pct' in df.columns:
        print("  ✓ Calculating Fast Charging Availability...")
        df = calculate_fast_charging_availability(df)
    
    if 'avg_income_index' in df.columns and 'fuel_price_rs_per_litre' in df.columns:
        print("  ✓ Calculating Economic Viability Index...")
        df = calculate_economic_viability_index(df)
    
    if 'policy_start_year' in df.columns:
        print("  ✓ Calculating Policy Maturity...")
        df = calculate_policy_maturity(df, current_year=current_year)
    
    print("✅ All KPIs calculated successfully!")
    print(f"📊 Total columns: {len(df.columns)}")
    
    return df


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def summarize_kpis_by_state(df: pd.DataFrame,
                             year: Optional[int] = None,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Create a summary table of key KPIs by state.
    
    Useful for creating state ranking tables in dashboards.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with calculated KPIs
    year : int, optional
        Specific year to summarize (if None, uses most recent)
    top_n : int, default=10
        Number of top states to return
    
    Returns:
    --------
    pd.DataFrame
        Summary table with top states by EV adoption rate
    """
    # Filter to specific year if provided
    if year is not None:
        df_year = df[df['year'] == year].copy()
    else:
        # Use most recent year
        max_year = df['year'].max()
        df_year = df[df['year'] == max_year].copy()
    
    # Aggregate by state
    summary = df_year.groupby('state').agg({
        'ev_vehicle_registrations': 'sum',
        'ice_vehicle_registrations': 'sum',
        'ev_adoption_rate': 'mean',
        'yoy_growth_rate': 'mean',
        'charging_stations': 'mean',
        'policy_score': 'first',
        'stations_per_1000_evs': 'mean'
    }).reset_index()
    
    # Sort by adoption rate
    summary = summary.sort_values('ev_adoption_rate', ascending=False).head(top_n)
    
    # Round for readability
    summary = summary.round(2)
    
    return summary


def export_kpis_for_dashboard(df: pd.DataFrame,
                               output_prefix: str = 'dashboard') -> None:
    """
    Export different views of KPI data optimized for dashboard creation.
    
    Creates multiple CSV files:
    - State-year level aggregation
    - Segment-year level aggregation
    - Latest year snapshot
    - Historical trends
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with all KPIs calculated
    output_prefix : str, default='dashboard'
        Prefix for output filenames
    
    Returns:
    --------
    None (saves CSV files)
    """
    # State-year aggregation
    state_year = df.groupby(['state', 'year']).agg({
        'ev_vehicle_registrations': 'sum',
        'ice_vehicle_registrations': 'sum',
        'ev_adoption_rate': 'mean',
        'yoy_growth_rate': 'mean',
        'charging_stations': 'mean',
        'policy_score': 'first'
    }).reset_index()
    state_year.to_csv(f'{output_prefix}_state_year.csv', index=False)
    print(f"✓ Saved: {output_prefix}_state_year.csv")
    
    # Segment-year aggregation
    segment_year = df.groupby(['vehicle_segment', 'year']).agg({
        'ev_vehicle_registrations': 'sum',
        'ice_vehicle_registrations': 'sum',
        'ev_adoption_rate': 'mean'
    }).reset_index()
    segment_year.to_csv(f'{output_prefix}_segment_year.csv', index=False)
    print(f"✓ Saved: {output_prefix}_segment_year.csv")
    
    # Latest year snapshot
    max_year = df['year'].max()
    latest = df[df['year'] == max_year].copy()
    latest.to_csv(f'{output_prefix}_latest_{max_year}.csv', index=False)
    print(f"✓ Saved: {output_prefix}_latest_{max_year}.csv")
    
    print("✅ All dashboard files exported!")


if __name__ == "__main__":
    """
    Example usage of the KPI functions library.
    """
    print("EV Adoption KPI Functions Library Loaded Successfully!")
    print("\nAvailable Functions:")
    print("  • calculate_ev_adoption_rate()")
    print("  • calculate_ice_to_ev_conversion_rate()")
    print("  • calculate_yoy_growth_rate()")
    print("  • calculate_market_share_by_segment()")
    print("  • calculate_infrastructure_adequacy_ratio()")
    print("  • calculate_policy_effectiveness_score()")
    print("  • predict_future_ev_share()")
    print("  • calculate_cagr()")
    print("  • calculate_fast_charging_availability()")
    print("  • calculate_economic_viability_index()")
    print("  • calculate_policy_maturity()")
    print("  • calculate_all_kpis() - Calculate everything at once!")
    print("\nUtility Functions:")
    print("  • summarize_kpis_by_state()")
    print("  • export_kpis_for_dashboard()")