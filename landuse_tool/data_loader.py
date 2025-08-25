import rasterio
from rasterio.windows import Window
import numpy as np
from collections import Counter
from tqdm import tqdm
import streamlit as st # Import streamlit for error messages

from .utils import reproject_raster, align_rasters, create_mask

def load_raster(path):
    """
    Load a single raster from a given path.
    """
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            profile = src.profile
        return arr, profile
    except Exception as e:
        st.error(f"Error loading raster file '{path}': {e}")
        return None, None

def load_targets(target_paths, align=True):
    """
    Load multi-temporal land cover rasters from a list of paths.
    Args:
        target_paths (list[str]): List of file paths.
    Returns:
        arrays, masks, profiles
    """
    raster_list = []
    for path in target_paths:
        arr, profile = load_raster(path)
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


def load_predictors(predictor_paths, ref_profile=None, align=True):
    """
    Load predictor rasters from a list of paths, align them to a reference.
    Returns stacked predictors [bands, height, width].
    """
    raster_list = []
    for path in predictor_paths:
        arr, profile = load_raster(path)
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

def sample_training_data(target_path, predictor_paths, total_samples=10000, window_size=512):
    X_samples = []
    y_samples = []

    src_target = None
    predictor_srcs = []

    try:
        src_target = rasterio.open(target_path)
        lc_full = src_target.read(1)
        src_profile = src_target.profile
        width, height = src_profile["width"], src_profile["height"]
        nodata = src_profile.get("nodata")
        mask_full = (lc_full != 255) & (lc_full != 254) & (lc_full != nodata)

        # We need to open the predictor rasters only once
        for path in predictor_paths:
            predictor_srcs.append(rasterio.open(path))

        for i in tqdm(range(0, height, window_size), desc="Sampling rows"):
            for j in range(0, width, window_size):
                if len(X_samples) >= total_samples:
                    break

                w = min(window_size, width - j)
                h = min(window_size, height - i)
                window = Window(j, i, w, h)

                lc_window = lc_full[i:i+h, j:j+w]
                mask_window = mask_full[i:i+h, j:j+w]
                valid_rows, valid_cols = np.where(mask_window)

                n_valid = len(valid_rows)
                if n_valid == 0:
                    continue

                n_samples = min(100, n_valid)
                sample_indices = np.random.choice(n_valid, size=n_samples, replace=False)

                for idx in sample_indices:
                    r_win = valid_rows[idx]
                    c_win = valid_cols[idx]

                    pixel_values = []
                    valid_pixel = True
                    for src_pred in predictor_srcs:
                        try:
                            val = src_pred.read(1, window=Window(j + c_win, i + r_win, 1, 1))[0, 0]
                            if np.isnan(val):
                                valid_pixel = False
                                break
                            pixel_values.append(val)
                        except Exception as e:
                            valid_pixel = False
                            break

                    if valid_pixel:
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
    finally:
        # Close all opened predictor files
        if src_target:
            src_target.close()
        for src_pred in predictor_srcs:
            src_pred.close()
