import rasterio
import numpy as np
import pandas as pd
import streamlit as st
from .data_loader import _open_as_raster

def calculate_transition_matrix(file1, file2):
    """
    Calculates a transition matrix and pixel counts between two land cover maps
    in a memory-safe way by processing in windows.
    """
    try:
        with _open_as_raster(file1) as src1, _open_as_raster(file2) as src2:
            if src1.profile != src2.profile:
                st.error("Error: Input rasters for transition analysis do not have the same dimensions or CRS.")
                return None, None

            nodata = src1.nodata or -9999
            
            # Find all unique classes present in both rasters by reading them fully once
            # This is memory intensive but necessary to define the matrix dimensions
            classes1 = np.unique(src1.read(1))
            classes2 = np.unique(src2.read(1))
            all_classes = sorted(list(np.unique(np.concatenate([classes1, classes2]))))
            all_classes = [c for c in all_classes if c != nodata]

            # Initialize a matrix to store counts
            class_map = {val: i for i, val in enumerate(all_classes)}
            counts = np.zeros((len(all_classes), len(all_classes)), dtype=np.int64)

            # Process raster in windows to save memory during counting
            for _, window in src1.block_windows(1):
                arr1 = src1.read(1, window=window)
                arr2 = src2.read(1, window=window)

                valid_mask = (arr1 != nodata) & (arr2 != nodata)
                
                arr1_flat = arr1[valid_mask]
                arr2_flat = arr2[valid_mask]

                # Map values to matrix indices
                idx1 = np.array([class_map.get(v, -1) for v in arr1_flat])
                idx2 = np.array([class_map.get(v, -1) for v in arr2_flat])

                valid_indices = (idx1 != -1) & (idx2 != -1)
                idx1, idx2 = idx1[valid_indices], idx2[valid_indices]

                # Use numpy's fast histogram function to increment counts
                np.add.at(counts, (idx1, idx2), 1)

        # Normalize counts to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            matrix = np.where(row_sums > 0, counts / row_sums, 0)

        matrix_df = pd.DataFrame(matrix, index=all_classes, columns=all_classes)
        counts_df = pd.DataFrame(counts, index=all_classes, columns=all_classes)
        
        return matrix_df, counts_df

    except Exception as e:
        st.error(f"Failed during transition analysis: {e}")
        return None, None

