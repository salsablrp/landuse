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
    This version writes uploaded files to a temporary location on disk in chunks
    to avoid holding the entire file in RAM. This is the most memory-safe method.
    """
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            # This is a Streamlit UploadedFile object.
            # Create the temp file in the system's /tmp directory to avoid
            # triggering Streamlit's file watcher and hitting the inotify limit.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp") as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                # Read and write in chunks to avoid memory spikes
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

def sample_training_data_stratified(target_files, predictor_files, samples_per_class=2000, progress_callback=None):
    """
    Performs stratified sampling to create a balanced training dataset.
    This is crucial for preventing the model from only learning the majority class.
    """
    if progress_callback is None:
        def progress_callback(frac, text): pass

    progress_callback(0.0, "Starting stratified sampling...")

    # --- Step 1: Identify classes and their locations (memory-safe) ---
    class_locations = {}
    with _open_as_raster(target_files[-1]) as lc_src:
        nodata = lc_src.nodata
        total_blocks = sum(1 for _ in lc_src.block_windows(1))
        current_block = 0
        for _, window in lc_src.block_windows(1):
            current_block += 1
            progress_callback(0.1 + (0.4 * (current_block / total_blocks)), f"Scanning for classes... (block {current_block}/{total_blocks})")
            lc_window = lc_src.read(1, window=window)
            valid_mask = (lc_window != nodata) & (lc_window is not None)
            
            unique_classes = np.unique(lc_window[valid_mask])
            for cls in unique_classes:
                if cls not in class_locations:
                    class_locations[cls] = []
                
                # Get coordinates relative to the window, then convert to global coordinates
                rows, cols = np.where(lc_window == cls)
                global_rows, global_cols = rows + window.row_off, cols + window.col_off
                class_locations[cls].extend(zip(global_rows, global_cols))

    if not class_locations:
        st.error("Could not find any valid data classes in the target raster.")
        return None, None

    # --- Step 2: Create balanced samples for each class ---
    X_samples, y_samples = [], []
    with ExitStack() as stack:
        predictor_sources = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
        
        num_classes = len(class_locations)
        current_class_idx = 0
        for cls, locations in class_locations.items():
            current_class_idx += 1
            progress_callback(0.5 + (0.5 * (current_class_idx / num_classes)), f"Sampling class {cls}...")
            
            # If a class has fewer pixels than desired, take all of them
            num_to_sample = min(samples_per_class, len(locations))
            
            # Randomly choose indices for the locations to sample
            sample_indices = np.random.choice(len(locations), num_to_sample, replace=False)
            
            for idx in sample_indices:
                r, c = locations[idx]
                
                # Read the single pixel value from each predictor raster
                try:
                    pixel_values = [src.read(1, window=Window(c, r, 1, 1))[0, 0] for src in predictor_sources]
                    X_samples.append(pixel_values)
                    y_samples.append(cls)
                except Exception as e:
                    # This can happen if a pixel is on the edge, safely skip it
                    continue

    if not X_samples:
        st.error("Sampling resulted in an empty dataset. Check raster alignment and data values.")
        return None, None
        
    progress_callback(1.0, "Sampling complete.")
    return np.array(X_samples), np.array(y_samples)
