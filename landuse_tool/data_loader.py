import rasterio
from rasterio.windows import Window
from rasterio.io import MemoryFile
import numpy as np
from collections import Counter
from tqdm import tqdm

from .utils import reproject_raster, align_rasters, create_mask

def _open_as_raster(path_or_file):
    """
    Open a raster either from:
      - an uploaded file-like object (Streamlit uploader)
      - a local file path
    Returns (array, profile).
    """
    if hasattr(path_or_file, "read"):  
        # Case 1: file-like object (Streamlit upload)
        file_bytes = path_or_file.read()
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read(1)
                profile = src.profile
    else:
        # Case 2: local file path (string/Path)
        with rasterio.open(str(path_or_file)) as src:
            arr = src.read(1)
            profile = src.profile

    return arr, profile


def load_raster(path_or_file):
    """
    Load a single raster and return (array, profile).
    """
    return _open_as_raster(path_or_file)


def load_targets(target_files, align=True):
    """
    Load multi-temporal land cover rasters.
    Args:
        target_files (list[file-like or str]): Uploaded files or file paths.
    Returns:
        arrays, masks, profiles
    """
    raster_list = [load_raster(f) for f in target_files]

    # Optional alignment
    if align and len(raster_list) > 1:
        raster_list = align_rasters(raster_list)  # assumes you have this util

    arrays, profiles = zip(*raster_list)
    masks = [create_mask(arr, nodata=prof.get("nodata")) for arr, prof in raster_list]

    return arrays, masks, profiles


def load_predictors(predictor_files, ref_profile=None, align=True):
    """
    Load predictor rasters, align them to a reference (if provided).
    Returns stacked predictors [bands, height, width].
    """
    raster_list = [load_raster(f) for f in predictor_files]

    if ref_profile and align:
        aligned = []
        from .utils import resample_raster
        for arr, prof in raster_list:
            aligned_arr = resample_raster(arr, prof, ref_profile)
            aligned.append((aligned_arr, ref_profile))
        raster_list = aligned

    arrays, _ = zip(*raster_list)
    stack = np.stack(arrays, axis=0)

    return stack


def sample_training_data(target_file, predictor_files, total_samples=10000, window_size=512):
    X_samples = []
    y_samples = []

    # Use _open_as_raster to handle the target file
    lc_full, src_profile = _open_as_raster(target_file)
    width, height = src_profile["width"], src_profile["height"]
    nodata = src_profile.get("nodata")
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

                # Use _open_as_raster for each predictor file
                for fobj in predictor_files:
                    try:
                        pred_arr, pred_profile = _open_as_raster(fobj)
                        # Ensure the correct window is read after opening
                        val = pred_arr[r_win, c_win]
                        if np.isnan(val):
                            valid_pixel = False
                            break
                        pixel_values.append(val)
                    except Exception as e:
                        print(f"Error reading pixel from predictor: {e}")
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
