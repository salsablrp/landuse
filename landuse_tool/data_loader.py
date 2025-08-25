import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
import numpy as np
from collections import Counter
from tqdm import tqdm
import streamlit as st # Import streamlit for error messages

from .utils import reproject_raster, align_rasters, create_mask

def _open_as_raster(file_object_or_path):
    """
    Open a raster from either an uploaded file-like object or a local file path.
    Returns (array, profile).
    """
    if hasattr(file_object_or_path, "read"):
        # Case 1: file-like object (Streamlit upload)
        file_bytes = file_object_or_path.read()
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read(1)
                profile = src.profile
    else:
        # Case 2: local file path (string/Path) - this is for non-uploaded files
        with rasterio.open(str(file_object_or_path)) as src:
            arr = src.read(1)
            profile = src.profile
    return arr, profile

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
    Load multi-temporal land cover rasters from a list of files or paths.
    Args:
        target_files (list[Union[str, UploadedFile]]): List of file objects or paths.
    Returns:
        arrays, masks, profiles
    """
    raster_list = []
    for f in target_files:
        arr, profile = load_raster(f)
        if arr is not None:
            raster_list.append((arr, profile))

    if not raster_list:
        st.warning("No valid target rasters were loaded.")
        return [], [], []

    try:
        if align and len(raster_list) > 1:
            raster_list = align_rasters(raster_list)
    except Exception as e:
        st.error(f"Error aligning target rasters: {e}")
        return [], [], []

    try:
        arrays, profiles = zip(*raster_list)
        masks = [create_mask(arr, nodata=prof.get("nodata")) for arr, prof in raster_list]
        return arrays, masks, profiles
    except Exception as e:
        st.error(f"Error processing target raster data: {e}")
        return [], [], []


def load_predictors(predictor_files, ref_profile=None, align=True):
    """
    Load predictor rasters from a list of files, align them to a reference.
    Returns stacked predictors [bands, height, width].
    """
    raster_list = []
    for f in predictor_files:
        arr, profile = load_raster(f)
        if arr is not None:
            raster_list.append((arr, profile))

    if not raster_list:
        st.warning("No valid predictor rasters were loaded.")
        return None

    if ref_profile and align:
        aligned = []
        from .utils import resample_raster
        try:
            for arr, prof in raster_list:
                aligned_arr = resample_raster(arr, prof, ref_profile)
                aligned.append((aligned_arr, ref_profile))
            raster_list = aligned
        except Exception as e:
            st.error(f"Error aligning predictor rasters: {e}")
            return None

    try:
        arrays, _ = zip(*raster_list)
        stack = np.stack(arrays, axis=0)
        return stack
    except Exception as e:
        st.error(f"Error stacking predictor arrays: {e}")
        return None

def sample_training_data(targets, predictors, total_samples=10000, window_size=512):
    X_samples = []
    y_samples = []

    try:
        # Use the in-memory arrays directly from the arguments
        lc_full = targets[0][-1]  # The latest target raster array
        width, height = lc_full.shape[1], lc_full.shape[0]
        mask_full = targets[1][-1] # The latest target mask

        for i in tqdm(range(0, height, window_size), desc="Sampling rows"):
            for j in range(0, width, window_size):
                if len(X_samples) >= total_samples:
                    break

                lc_window = lc_full[i:i+window_size, j:j+window_size]
                mask_window = mask_full[i:i+window_size, j:j+window_size]
                valid_rows, valid_cols = np.where(mask_window)

                n_valid = len(valid_rows)
                if n_valid == 0:
                    continue

                n_samples = min(100, n_valid)
                sample_indices = np.random.choice(n_valid, size=n_samples, replace=False)

                for idx in sample_indices:
                    r_win = valid_rows[idx]
                    c_win = valid_cols[idx]

                    # Extract the pixel values directly from the in-memory predictors stack
                    pixel_values = predictors[:, i + r_win, j + c_win].tolist()
                    
                    X_samples.append(pixel_values)
                    y_samples.append(lc_window[r_win, c_win])

        # Filter classes with too few samples
        class_counts = Counter(y_samples)
        valid_classes = {cls for cls, count in class_counts.items() if count >= 2}

        X = [x for x, y in zip(X_samples, y_samples) if y in valid_classes]
        y = [y for y in y_samples if y in valid_classes]

        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"Error during training data sampling: {e}")
        return None, None
