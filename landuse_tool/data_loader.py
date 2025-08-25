from contextlib import contextmanager, ExitStack
import rasterio
from rasterio.windows import Window
import numpy as np
from collections import Counter
import streamlit as st
import tempfile
import os

from .utils import create_mask

@contextmanager
def _open_as_raster(file_object_or_path):
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp") as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                while True:
                    chunk = file_object_or_path.read(16 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
            with rasterio.open(temp_filepath) as src:
                yield src
        else:
            with rasterio.open(str(file_object_or_path)) as src:
                yield src
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def load_targets(target_files):
    if not target_files or len(target_files) < 2:
        st.warning("Please upload at least two target files.")
        return None, None
    try:
        for f in target_files:
            with _open_as_raster(f):
                pass
        with _open_as_raster(target_files[-1]) as src:
            ref_profile = src.profile
            nodata = src.nodata
            mask = np.empty((src.height, src.width), dtype=bool)
            for _, window in src.block_windows(1):
                window_data = src.read(1, window=window)
                mask[window.row_off:window.row_off+window.height, 
                     window.col_off:window.col_off+window.width] = (window_data != nodata)
        return ref_profile, mask
    except Exception as e:
        st.error(f"An error occurred while processing target files: {e}")
        return None, None

def load_predictors(predictor_files, ref_profile):
    if not ref_profile:
        st.error("Cannot validate predictors without a reference profile.")
        return False
    ref_width, ref_height = ref_profile['width'], ref_profile['height']
    try:
        for f in predictor_files:
            with _open_as_raster(f) as src:
                if src.width != ref_width or src.height != ref_height:
                    st.error(f"Dimension mismatch: Predictor '{f.name}' ({src.width}x{src.height}) does not match target ({ref_width}x{ref_height}).")
                    return False
        return True
    except Exception as e:
        st.error(f"Error validating predictor files: {e}")
        return False

# MODIFIED FUNCTION SIGNATURE
def sample_training_data(target_files, predictor_files, ref_profile, total_samples=10000, window_size=512, progress_callback=None):
    X_samples, y_samples = [], []
    with ExitStack() as stack:
        try:
            predictor_sources = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            lc_src = stack.enter_context(_open_as_raster(target_files[-1]))
            width, height, nodata = lc_src.width, lc_src.height, lc_src.nodata

            # MODIFICATION FOR PROGRESS BAR
            for i in range(0, height, window_size):
                if progress_callback:
                    progress_fraction = i / height
                    progress_callback(progress_fraction, f"Sampling... {int(progress_fraction*100)}% complete")
                
                for j in range(0, width, window_size):
                    if len(y_samples) >= total_samples: break
                    window = Window(j, i, min(window_size, width - j), min(window_size, height - i))
                    lc_window = lc_src.read(1, window=window)
                    mask_window = (lc_window != nodata) & (lc_window is not None)
                    valid_rows_win, valid_cols_win = np.where(mask_window)
                    n_valid = len(valid_rows_win)
                    if n_valid == 0: continue
                    n_samples_from_window = min(100, n_valid)
                    sample_indices = np.random.choice(n_valid, size=n_samples_from_window, replace=False)
                    predictor_windows = [p_src.read(1, window=window) for p_src in predictor_sources]
                    for idx in sample_indices:
                        r, c = valid_rows_win[idx], valid_cols_win[idx]
                        pixel_values = [p_win[r, c] for p_win in predictor_windows]
                        X_samples.append(pixel_values)
                        y_samples.append(lc_window[r, c])
                if len(y_samples) >= total_samples: break
            
            if progress_callback: progress_callback(1.0, "Finalizing samples...")
            class_counts = Counter(y_samples)
            valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
            X = [x for x, y in zip(X_samples, y_samples) if y in valid_classes]
            y = [y for y in y_samples if y in valid_classes]
            return np.array(X), np.array(y)
        except Exception as e:
            st.error(f"Error during sampling: {e}")
            return None, None
