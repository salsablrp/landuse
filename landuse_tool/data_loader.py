import rasterio
import numpy as np
from pathlib import Path
from .utils import reproject_raster, align_rasters, create_mask

def load_raster(path):
    """
    Load a single raster and return (array, profile).
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
    return arr, profile


def load_targets(target_paths, align=True):
    """
    Load multi-temporal land cover rasters.
    Args:
        target_paths (list[str]): Paths to land cover rasters (e.g., yearly).
        align (bool): Whether to align rasters to the first one.
    Returns:
        list of (array, profile), list of masks
    """
    raster_list = [load_raster(p) for p in target_paths]

    # Align rasters
    if align:
        raster_list = align_rasters(raster_list)

    arrays, profiles = zip(*raster_list)
    masks = [create_mask(arr, nodata=prof.get("nodata")) for arr, prof in raster_list]

    return arrays, masks, profiles


def load_predictors(predictor_paths, ref_profile=None, align=True):
    """
    Load predictor rasters, align them to a reference (if provided).
    Args:
        predictor_paths (list[str]): Paths to predictor rasters.
        ref_profile (dict): Reference raster profile (from target).
        align (bool): Whether to align predictors to the reference profile.
    Returns:
        np.ndarray: stacked predictors [bands, height, width]
    """
    raster_list = [load_raster(p) for p in predictor_paths]

    if ref_profile and align:
        aligned = []
        for arr, prof in raster_list:
            from .utils import resample_raster
            aligned_arr = resample_raster(arr, prof, ref_profile)
            aligned.append((aligned_arr, ref_profile))
        raster_list = aligned

    arrays, _ = zip(*raster_list)
    stack = np.stack(arrays, axis=0)  # shape = [n_predictors, H, W]

    return stack


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