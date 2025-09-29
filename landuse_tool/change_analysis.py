import rasterio
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from .data_loader import _open_as_raster

def calculate_transition_matrix(file1, file2):
    """
    Calculates a transition matrix and counts between two land cover maps.
    MODIFIED: Now consistently returns three values (matrix, counts, plots)
    to match the signature of the non-linear function.
    """
    try:
        # ... (The core calculation logic remains the same) ...
        with _open_as_raster(file1) as src1, _open_as_raster(file2) as src2:
            arr1 = src1.read(1)
            arr2 = src2.read(1)
            if arr1.shape != arr2.shape:
                st.error("Rasters must have the same dimensions!")
                return None, None, []
            nodata = src1.nodata
            if nodata is None:
                unique_vals = np.unique(arr1)
                nodata = -9999 if -9999 not in unique_vals else np.min(unique_vals) - 1
            valid_mask = (arr1 != nodata) & (arr2 != nodata)
            arr1_flat = arr1[valid_mask]
            arr2_flat = arr2[valid_mask]
            classes = sorted(list(np.unique(np.concatenate([arr1_flat, arr2_flat]))))
            from sklearn.metrics import confusion_matrix
            counts = confusion_matrix(arr1_flat, arr2_flat, labels=classes)
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(row_sums > 0, counts / row_sums, 0)
        matrix_df = pd.DataFrame(matrix, index=classes, columns=classes)
        counts_df = pd.DataFrame(counts, index=classes, columns=classes)
        # Return an empty list for plots to maintain a consistent signature
        return matrix_df, counts_df, []
    except Exception as e:
        st.error(f"Error calculating transition matrix: {e}")
        return None, None, []

def analyze_non_linear_trends(target_files_with_years):
    """
    Analyzes change over multiple periods by fitting a linear regression to the
    rate of change for each transition, then extrapolating to predict future change.
    """
    st.info("Non-linear mode: Fitting a trend line to the rate of change from all historical periods.")
    
    sorted_targets = sorted(target_files_with_years, key=lambda x: x['year'])
    periods = []
    for i in range(len(sorted_targets) - 1):
        start, end = sorted_targets[i], sorted_targets[i+1]
        period_duration = end['year'] - start['year']
        if period_duration <= 0:
            st.warning(f"Skipping invalid period: {start['year']} -> {end['year']}.")
            continue
        midpoint_year = start['year'] + period_duration / 2
        
        # --- THIS IS THE CORRECTED LINE ---
        # The function now correctly unpacks the 3 return values.
        # We don't need the matrix or plots from these intermediate steps, just the counts.
        _, counts, _ = calculate_transition_matrix(start['file'], end['file'])
        
        if counts is not None:
            annual_rate = counts / period_duration
            periods.append({'midpoint': midpoint_year, 'rates': pd.DataFrame(annual_rate, index=counts.index, columns=counts.columns)})

    if len(periods) < 2:
        st.error("Cannot perform non-linear analysis with fewer than two valid time periods (requires at least 3 maps).")
        return None, None, []

    all_classes = sorted(list(set(index for p in periods for index in p['rates'].index)))
    final_counts_template = periods[-1]['rates'].reindex(index=all_classes, columns=all_classes, fill_value=0)
    projected_annual_rates = pd.DataFrame(0.0, index=final_counts_template.index, columns=final_counts_template.columns)
    years = np.array([p['midpoint'] for p in periods]).reshape(-1, 1)

    generated_plots = []
    for from_cls in final_counts_template.index:
        for to_cls in final_counts_template.columns:
            if from_cls == to_cls: continue
            rate_timeseries = [p['rates'].loc[from_cls, to_cls] if from_cls in p['rates'].index and to_cls in p['rates'].columns else 0 for p in periods]
            if np.sum(rate_timeseries) == 0: continue
            
            model = LinearRegression()
            model.fit(years, rate_timeseries)
            
            last_period_duration = periods[-1]['midpoint'] - periods[-2]['midpoint']
            future_midpoint = periods[-1]['midpoint'] + last_period_duration
            predicted_rate = model.predict([[future_midpoint]])[0]
            
            projected_annual_rates.loc[from_cls, to_cls] = max(0, predicted_rate)

            if np.mean(rate_timeseries) > 100:
                fig, ax = plt.subplots()
                ax.scatter(years, rate_timeseries, label='Historical Annual Rate')
                ax.plot(years, model.predict(years), color='red', linestyle='--', label='Trend Line')
                ax.scatter([future_midpoint], [predicted_rate], color='green', zorder=5, label='Predicted Future Rate')
                ax.set_title(f'Trend for Transition: {from_cls} -> {to_cls}')
                ax.set_xlabel('Year'); ax.set_ylabel('Annual Pixel Change Rate'); ax.legend()
                generated_plots.append(fig)

    total_historical_period = sorted_targets[-1]['year'] - sorted_targets[0]['year']
    projected_total_counts = projected_annual_rates * total_historical_period
    
    # Use the matrix from the last period for probability reference
    final_matrix, _, _ = calculate_transition_matrix(sorted_targets[-2]['file'], sorted_targets[-1]['file'])
    
    return final_matrix, projected_total_counts.round().astype(int), generated_plots

