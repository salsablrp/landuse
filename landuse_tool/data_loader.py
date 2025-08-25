from contextlib import contextmanager, ExitStack
import rasterio
from rasterio.windows import Window
import numpy as np
from collections import Counter
from tqdm import tqdm
import streamlit as st
import tempfile
import os

from .utils import reproject_raster, align_rasters, create_mask

@contextmanager
def _open_as_raster(file_object_or_path):
    """
    A context manager to safely open a raster from an uploaded file or a path.
    This version writes uploaded files to a temporary location on disk to
    avoid holding the entire file in RAM, which is crucial for memory-constrained
    environments like Streamlit Cloud.
    """
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            # This is a Streamlit UploadedFile object.
            # Write its contents to a temporary file on disk.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                while True:
                    chunk = file_object_or_path.read(16 * 1024)  # Read 16KB at a time
                    if not chunk:
                        break
                    tmp.write(chunk)

            # Now, open the raster from the temporary file path.
            with rasterio.open(temp_filepath) as src:
                yield src
        else:
            # This is already a local file path.
            with rasterio.open(str(file_object_or_path)) as src:
                yield src
    finally:
        # Clean up the temporary file after we're done.
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def load_targets(target_files, align=True):
    """
    ROBUST & MEMORY-SAFE: This function validates all target files and
    creates the data mask in a memory-efficient way by processing the
    last target file in windows instead of reading it all at once.
    """
    if not target_files or len(target_files) < 2:
        st.warning("Please upload at least two target files.")
        return None, None

    try:
        # Lightweight validation: Open each file to check it's a valid raster.
        for f in target_files:
            with _open_as_raster(f) as src:
                pass  # Just opening it is enough validation for now.
        
        # --- MEMORY-SAFE MASK CREATION ---
        # Open the last target file to get its properties.
        with _open_as_raster(target_files[-1]) as src:
            ref_profile = src.profile
            nodata = src.nodata
            
            # Create an empty boolean array to hold the mask. This uses 8x less
            # memory than reading the original raster data.
            mask = np.empty((src.height, src.width), dtype=bool)

            # Iterate over the raster in windows to build the mask piece by piece.
            for _, window in src.block_windows(1):
                window_data = src.read(1, window=window)
                # Create the mask for the small window and place it in the larger array.
                mask[window.row_off:window.row_off+window.height, 
                     window.col_off:window.col_off+window.width] = (window_data != nodata)

        return ref_profile, mask

    except Exception as e:
        st.error(f"An error occurred while processing target files: {e}")
        return None, None

def load_predictors(predictor_files, ref_profile=None):
    """
    MODIFIED: This function now validates that each predictor has the
    same dimensions (width and height) as the reference profile. This
    prevents IndexError during sampling.
    """
    if not predictor_files:
        return False
    
    if not ref_profile:
        st.error("Cannot validate predictors without a reference profile from the target files.")
        return False

    ref_width = ref_profile['width']
    ref_height = ref_profile['height']
    
    try:
        for f in predictor_files:
            with _open_as_raster(f) as src:
                if src.width != ref_width or src.height != ref_height:
                    st.error(f"Dimension mismatch: Predictor '{f.name}' ({src.width}x{src.height}) does not match target dimensions ({ref_width}x{ref_height}). Please align your rasters.")
                    return False
        return True # Success, all dimensions match
    except Exception as e:
        st.error(f"Error validating predictor files: {e}")
        return False

def sample_training_data(target_files, predictor_files, ref_profile, total_samples=10000, window_size=512):
    """
    Efficiently samples training data using windowed reading to minimize memory usage.
    """
    X_samples = []
    y_samples = []

    with ExitStack() as stack:
        try:
            predictor_sources = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            lc_src = stack.enter_context(_open_as_raster(target_files[-1]))
            
            width, height = lc_src.width, lc_src.height
            nodata = lc_src.nodata

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
                        r, c = valid_rows_win[idx], valid_cols_win[idx]
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
