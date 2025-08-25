from io import BytesIO
from contextlib import contextmanager
from contextlib import ExitStack

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
    This version correctly handles in-memory files for rasterio.
    """
    if hasattr(file_object_or_path, "read"):
        # For an uploaded file, we must read its content into bytes for MemoryFile.
        file_object_or_path.seek(0)
        file_bytes = file_object_or_path.read()
        
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                yield src
    else:
        # For a local file path, open it directly.
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
    MODIFIED: This version is robust and memory-efficient for multiple files.
    It reads the profile from all files but only reads the full pixel data
    from the LAST file to generate the mask.
    """
    if not target_files or len(target_files) < 2:
        # This message is now handled in the function itself for clarity.
        st.warning("Please upload at least two target files.")
        return None, None

    try:
        # Lightweight check: Open each file to get its profile. This uses more
        # memory than just checking the header, but is necessary with MemoryFile.
        # We rely on Streamlit's resource limits to manage this.
        profiles = []
        for f in target_files:
            with _open_as_raster(f) as src:
                profiles.append(src.profile)
        
        # Now, do a final read ONLY on the last file to get the mask.
        # The bytes for this file are read again, which is inefficient but safe.
        with _open_as_raster(target_files[-1]) as src:
            ref_profile = src.profile
            arr = src.read(1)
            nodata = src.nodata
            mask = create_mask(arr, nodata=nodata)

        return ref_profile, mask

    except Exception as e:
        st.error(f"An error occurred while processing target files: {e}")
        return None, None

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

    # Use ExitStack to manage multiple file contexts
    with ExitStack() as stack:
        try:
            # Open all predictor files and add them to the stack
            predictor_sources = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            
            # Open the latest target file
            lc_src = stack.enter_context(_open_as_raster(target_files[-1]))
            
            width, height = lc_src.width, lc_src.height
            nodata = lc_src.nodata

            # ... (the rest of the function is the same, no changes needed inside the loops) ...
            for i in tqdm(range(0, height, window_size), desc="Sampling rows"):
                for j in tqdm(range(0, width, window_size), desc="Sampling columns", leave=False):
                    if len(y_samples) >= total_samples:
                        break
                    
                    window = Window(j, i, min(window_size, width - j), min(window_size, height - i))
                    lc_window = lc_src.read(1, window=window)
                    
                    mask_window = (lc_window != nodata) & (lc_window is not None)
                    valid_rows_win, valid_cols_win = np.where(mask_window)

                    n_valid = len(valid_rows_win)
                    if n_valid == 0:
                        continue

                    n_samples_from_window = min(100, n_valid) 
                    sample_indices = np.random.choice(n_valid, size=n_samples_from_window, replace=False)

                    predictor_windows = [p_src.read(1, window=window) for p_src in predictor_sources]
                    
                    for idx in sample_indices:
                        r = valid_rows_win[idx]
                        c = valid_cols_win[idx]
                        pixel_values = [p_win[r, c] for p_win in predictor_windows]
                        X_samples.append(pixel_values)
                        y_samples.append(lc_window[r, c])
                
                if len(y_samples) >= total_samples:
                    break

            if len(y_samples) == 0:
                st.warning("No valid data points could be sampled.")
                return None, None

            class_counts = Counter(y_samples)
            valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
            X = [x for x, y in zip(X_samples, y_samples) if y in valid_classes]
            y = [y for y in y_samples if y in valid_classes]

            return np.array(X), np.array(y)

        except Exception as e:
            st.error(f"Error during optimized training data sampling: {e}")
            return None, None