import rasterio
from rasterio.windows import Window
import numpy as np
from collections import Counter
from tqdm import tqdm

import tempfile
from pathlib import Path
from .utils import reproject_raster, align_rasters, create_mask

def _open_as_raster(path_or_file):
    """
    Helper to open a raster from a file path or an uploaded file-like object.
    Returns (array, profile).
    """
    if hasattr(path_or_file, "read"):  
        # It's a file-like object (e.g., from Streamlit uploader)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp.write(path_or_file.read())
        tmp.flush()
        tmp.close()
        path = tmp.name
    else:
        # It's already a file path
        path = str(path_or_file)

    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile

    return arr, profile


def load_raster(path_or_file):
    """
    Load a single raster and return (array, profile).
    Works with local paths or file-like objects (Streamlit upload).
    """
    return _open_as_raster(path_or_file)


def load_targets(target_paths, align=True):
    """
    Load multi-temporal land cover rasters.
    Args:
        target_paths (list[str or file-like]): Paths or uploaded files.
        align (bool): Whether to align rasters to the first one.
    Returns:
        arrays, masks, profiles
    """
    raster_list = [load_raster(p) for p in target_paths]

    # Align rasters
    if align and len(raster_list) > 1:
        raster_list = align_rasters(raster_list)

    arrays, profiles = zip(*raster_list)
    masks = [create_mask(arr, nodata=prof.get("nodata")) for arr, prof in raster_list]

    return arrays, masks, profiles


def load_predictors(predictor_paths, ref_profile=None, align=True):
    """
    Load predictor rasters, align them to a reference (if provided).
    Args:
        predictor_paths (list[str or file-like]): Paths or uploaded files.
        ref_profile (dict): Reference raster profile (from target).
        align (bool): Whether to align predictors to the reference profile.
    Returns:
        np.ndarray: stacked predictors [bands, height, width]
    """
    raster_list = [load_raster(p) for p in predictor_paths]

    if ref_profile and align:
        aligned = []
        from .utils import resample_raster
        for arr, prof in raster_list:
            aligned_arr = resample_raster(arr, prof, ref_profile)
            aligned.append((aligned_arr, ref_profile))
        raster_list = aligned

    arrays, _ = zip(*raster_list)
    stack = np.stack(arrays, axis=0)  # shape = [n_predictors, H, W]

    return stack


def sample_training_data(target_path, predictor_paths, total_samples=10000, window_size=512):
    X_samples = []
    y_samples = []

    with rasterio.open(target_path) as src_target:
        width, height = src_target.width, src_target.height
        nodata = src_target.nodata
        lc_full = src_target.read(1)
        mask_full = (lc_full != 255) & (lc_full != 254) & (lc_full != nodata)

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

                    for fname in predictor_paths:
                        with rasterio.open(fname) as src_pred:
                            try:
                                val = src_pred.read(1, window=Window(j + c_win, i + r_win, 1, 1))[0, 0]
                                if np.isnan(val):
                                    valid_pixel = False
                                    break
                                pixel_values.append(val)
                            except:
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


# import rasterio
# import numpy as np
# from .config import TARGET_RASTER, PREDICTOR_PATHS

# def load_target():
#     with rasterio.open(TARGET_RASTER) as src:
#         lc = src.read(1)
#         profile = src.profile
#         mask = (lc != 254) & (lc != 255) & (lc != src.nodata)
#     return lc, mask, profile

# def load_predictors(mask):
#     stack = []
#     for path in PREDICTOR_PATHS:
#         with rasterio.open(path) as src:
#             band = src.read(1)
#             band = np.where(mask, band, np.nan)
#             stack.append(band)
#     return np.stack(stack, axis=0)