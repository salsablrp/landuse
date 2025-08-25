from io import BytesIO
from contextlib import contextmanager

import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
import numpy as np
from collections import Counter
from tqdm import tqdm
import streamlit as st # Import streamlit for error messages

from .utils import reproject_raster, align_rasters, create_mask

@contextmanager
def _open_as_raster(file_object_or_path):
    """
    A context manager to safely open a raster from an uploaded file or a path.
    This ensures the underlying MemoryFile is not garbage-collected prematurely.
    """
    if hasattr(file_object_or_path, "read"):
        # Handle in-memory file (Streamlit UploadedFile)
        file_object_or_path.seek(0)
        with MemoryFile(file_object_or_path.read()) as memfile:
            with memfile.open() as src:
                yield src
    else:
        # Handle local file path
        with rasterio.open(str(file_object_or_path)) as src:
            yield src

def load_raster(file_object_or_path):
    """
    Load a single raster and return (array, profile).
    """
    try:
        return _open_as_raster(file_object_or_path)
    except Exception as e:
        st.error(f"Error loading raster file '{getattr(file_object_or_path, 'name', 'unknown')}': {e}")
        return None, None

def load_targets(target_files, align=True):
    """
    MODIFIED: This function now primarily extracts profiles and the final mask.
    It AVOIDS loading all raster arrays into memory.
    """
    raster_info = []
    for f in target_files:
        # Using the memory-safe _open_as_raster that returns a dataset object
        with _open_as_raster(f) as src:
            raster_info.append({'profile': src.profile, 'name': f.name})
    
    if not raster_info:
        st.warning("No valid target rasters were found.")
        return None, None
    
    # You might perform alignment checks on the profiles here without loading data
    # For simplicity, we'll skip the complex align_rasters logic for now.
    
    profiles = [info['profile'] for info in raster_info]
    
    # Only load the MASK of the latest target raster, which is usually small.
    with _open_as_raster(target_files[-1]) as src:
        arr = src.read(1)
        nodata = src.nodata
        mask = create_mask(arr, nodata=nodata) # Assuming create_mask is memory-efficient

    return profiles, mask

def load_predictors(predictor_files, ref_profile=None):
    """
    MODIFIED: This function is now a lightweight validator.
    It checks if predictors can be opened but DOES NOT stack them.
    It returns a success flag instead of a giant numpy array.
    """
    if not predictor_files:
        return False
        
    # Simply check if each file can be opened. You can add more checks here
    # (e.g., comparing profiles to the ref_profile) without loading arrays.
    try:
        for f in predictor_files:
            with _open_as_raster(f) as src:
                # Optional: Check if CRS matches ref_profile, etc.
                pass
        return True # Success
    except Exception as e:
        st.error(f"Error validating predictor files: {e}")
        return False

def sample_training_data(target_files, predictor_files, ref_profile, total_samples=10000, window_size=512):
    """
    Efficiently samples training data using windowed reading to minimize memory usage.
    """
    X_samples = []
    y_samples = []

    try:
        # Open all predictor files but don't read them into memory yet.
        predictor_sources = [_open_as_raster(f) for f in predictor_files]
        
        # We only need the latest target file for sampling y values
        latest_target_file = target_files[-1]
        
        with _open_as_raster(latest_target_file) as lc_src:
            width, height = lc_src.width, lc_src.height
            nodata = lc_src.nodata

            for i in tqdm(range(0, height, window_size), desc="Sampling rows"):
                # THIS IS THE CORRECTED LINE:
                for j in tqdm(range(0, width, window_size), desc="Sampling columns", leave=False):
                    if len(y_samples) >= total_samples:
                        break
                    
                    # Define the window to read
                    window = Window(j, i, min(window_size, width - j), min(window_size, height - i))

                    # Read only the window from the target raster
                    lc_window = lc_src.read(1, window=window)
                    
                    # Create a mask for valid data within the window
                    mask_window = (lc_window != nodata) & (lc_window is not None)
                    valid_rows_win, valid_cols_win = np.where(mask_window)

                    n_valid = len(valid_rows_win)
                    if n_valid == 0:
                        continue

                    # Decide how many samples to take from this window
                    n_samples_from_window = min(100, n_valid) 
                    sample_indices = np.random.choice(n_valid, size=n_samples_from_window, replace=False)

                    # Read the corresponding window from all predictor rasters
                    predictor_windows = [p_src.read(1, window=window) for p_src in predictor_sources]
                    
                    for idx in sample_indices:
                        r = valid_rows_win[idx]
                        c = valid_cols_win[idx]

                        # Get the predictor values for the specific pixel (r, c) in the window
                        pixel_values = [p_win[r, c] for p_win in predictor_windows]
                        
                        X_samples.append(pixel_values)
                        y_samples.append(lc_window[r, c])
                
                # Break the outer loop as well if we have enough samples
                if len(y_samples) >= total_samples:
                    break

            if len(y_samples) == 0:
                st.warning("No valid data points could be sampled.")
                return None, None

        # Filter classes with too few samples (as before)
        class_counts = Counter(y_samples)
        valid_classes = {cls for cls, count in class_counts.items() if count >= 2}

        X = [x for x, y in zip(X_samples, y_samples) if y in valid_classes]
        y = [y for y in y_samples if y in valid_classes]

        # Close all the raster files
        for src in predictor_sources:
            src.close()

        return np.array(X), np.array(y)

    except Exception as e:
        st.error(f"Error during optimized training data sampling: {e}")
        # Ensure sources are closed on error
        if 'predictor_sources' in locals():
            for src in predictor_sources:
                if not src.closed:
                    src.close()
        return None, None