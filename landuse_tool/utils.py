import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(src_path, dst_crs, resampling=Resampling.nearest):
    """
    Reproject a raster to a target CRS.
    Args:
        src_path (str): Path to source raster.
        dst_crs (dict or str): Target CRS (e.g., 'EPSG:4326').
        resampling: rasterio.warp.Resampling method.
    Returns:
        np.ndarray, dict: reprojected array and updated profile.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        dst = np.empty((height, width), dtype=src.meta["dtype"])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
    return dst, kwargs


def resample_raster(array, src_profile, ref_profile, resampling=Resampling.nearest):
    """
    Resample a raster array to match a reference profile.
    Args:
        array (np.ndarray): Input raster array.
        src_profile (dict): Source raster profile.
        ref_profile (dict): Reference raster profile.
        resampling: rasterio.warp.Resampling method.
    Returns:
        np.ndarray: Resampled array.
    """
    dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=array.dtype)

    reproject(
        source=array,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=resampling
    )
    return dst


def align_rasters(raster_list, resampling=Resampling.nearest):
    """
    Align multiple rasters to the same grid (CRS, resolution, extent).
    Args:
        raster_list: list of (array, profile) tuples
        resampling: resampling method
    Returns:
        list of (aligned_array, ref_profile)
    """
    # Use the first raster as reference
    ref_array, ref_profile = raster_list[0]
    aligned = [(ref_array, ref_profile)]

    for arr, prof in raster_list[1:]:
        aligned_arr = resample_raster(arr, prof, ref_profile, resampling=resampling)
        aligned.append((aligned_arr, ref_profile))
    return aligned


def create_mask(array, nodata=None, invalid_values=[254, 255]):
    """
    Create a boolean mask for valid pixels.
    Args:
        array: np.ndarray
        nodata: nodata value
        invalid_values: list of values to exclude
    Returns:
        np.ndarray (bool mask)
    """
    mask = np.ones_like(array, dtype=bool)
    if nodata is not None:
        mask &= array != nodata
    for val in invalid_values:
        mask &= array != val
    return mask
