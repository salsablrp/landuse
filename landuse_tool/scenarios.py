import rasterio
import numpy as np
import os
import tempfile
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

from .data_loader import _open_as_raster

def generate_neighborhood_predictors(base_raster_file, temp_dir, radius_pixels=5):
    """
    Generates neighborhood-based predictor rasters (e.g., density).

    Args:
        base_raster_file (UploadedFile): The raster to analyze (e.g., latest LC map).
        temp_dir (str): Directory to save output temporary files.
        radius_pixels (int): The radius in pixels for the neighborhood analysis.

    Returns:
        list: A list of file paths to the newly created predictor rasters.
    """
    new_predictor_paths = []
    
    with _open_as_raster(base_raster_file) as src:
        profile = src.profile
        profile.update(dtype='float32', count=1, nodata=-1.0)
        lc_arr = src.read(1)
        unique_classes = np.unique(lc_arr[lc_arr != src.nodata])

    # The size of the filter window is (2 * radius + 1)
    filter_size = 2 * radius_pixels + 1

    for cls in unique_classes:
        # Create a binary map for the current class
        binary_map = (lc_arr == cls).astype('float32')
        
        # Use a uniform filter to calculate the density (mean) in the neighborhood
        # This is a fast and efficient way to perform a moving window analysis
        density_map = uniform_filter(binary_map, size=filter_size, mode='constant', cval=0)
        
        # Save the new raster
        output_path = os.path.join(temp_dir, f'density_class_{cls}.tif')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(density_map, 1)
        
        new_predictor_paths.append(output_path)
        
    return new_predictor_paths

