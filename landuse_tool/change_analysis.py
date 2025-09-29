import rasterio
import numpy as np
import pandas as pd
import streamlit as st
from .data_loader import _open_as_raster

def calculate_transition_matrix(file1, file2):
    """Calculates a transition matrix and counts between two land cover maps."""
    try:
        with _open_as_raster(file1) as src1, _open_as_raster(file2) as src2:
            arr1 = src1.read(1)
            arr2 = src2.read(1)

            if arr1.shape != arr2.shape:
                st.error("Rasters must have the same dimensions!")
                return None, None

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
        
        return matrix_df, counts_df
    except Exception as e:
        st.error(f"Error calculating transition matrix: {e}")
        return None, None

def analyze_non_linear_trends(target_files_with_years):
    """
    Analyzes change over multiple periods to detect non-linear trends.
    This implementation uses the rate of change from the most recent period for projection.
    """
    st.info("Non-linear mode: Using the rate of change from the most recent period for projection.")
    
    # Sort by year to ensure correct order
    sorted_targets = sorted(target_files_with_years, key=lambda x: x['year'])
    
    # Use the last two maps to calculate the most recent rate of change
    recent_start_file = sorted_targets[-2]['file']
    recent_end_file = sorted_targets[-1]['file']
    
    start_year_recent = sorted_targets[-2]['year']
    end_year_recent = sorted_targets[-1]['year']
    
    start_year_overall = sorted_targets[0]['year']
    end_year_overall = sorted_targets[-1]['year']
    
    st.write(f"Analyzing trend from the most recent interval: **{start_year_recent} -> {end_year_recent}**")

    matrix, counts = calculate_transition_matrix(recent_start_file, recent_end_file)
    if counts is None:
        return None, None
    
    # Project the recent rate over the total historical period for the simulation
    # This keeps the magnitude of change consistent with the simulation's forecast period
    total_period = end_year_overall - start_year_overall
    recent_period = end_year_recent - start_year_recent
    
    if recent_period > 0:
        # Calculate the average annual rate from the recent period
        annual_rate = counts / recent_period
        # Project this annual rate over the total historical period
        projected_counts = annual_rate * total_period
        return matrix, projected_counts.round().astype(int)
    else:
        # If years are the same, no projection is possible, return the raw counts
        return matrix, counts

