import rasterio
import numpy as np
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

def create_neighborhood_predictors(predictor_files, lc_end_file, radius_pixels, temp_dir):
    """
    Generates new predictor rasters based on neighborhood statistics.
    
    Args:
        predictor_files (list): List of file-like objects for original predictors.
        lc_end_file (file-like): The most recent land cover map.
        radius_pixels (int): The radius of the neighborhood in pixels.
        temp_dir (str): Path to the temporary directory.

    Returns:
        list: A list of file paths to the new neighborhood predictor rasters.
    """
    new_predictor_paths = []
    
    # Use a set to avoid duplicating neighborhood calculations for the same base predictor
    # This might happen if user uploads same file twice by mistake
    processed_files = set()

    for predictor_file in predictor_files:
        if predictor_file.name in processed_files:
            continue
        
        # We need to read the file to a temporary path to process it
        # This uses the same memory-safe approach as the data_loader
        from .data_loader import _open_as_raster
        with _open_as_raster(predictor_file) as src:
            profile = src.profile
            profile.update(dtype='float32')
            arr = src.read(1, out_dtype='float32')

            # Calculate mean in neighborhood using a uniform filter (fast)
            # The size of the filter is (radius * 2 + 1)
            filter_size = int(radius_pixels * 2 + 1)
            neighborhood_mean = uniform_filter(arr, size=filter_size, mode='reflect')

            # Save the new raster to a temporary file
            output_filename = f"neighborhood_mean_{predictor_file.name}"
            output_path = f"{temp_dir}/{output_filename}"

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(neighborhood_mean, 1)

            new_predictor_paths.append(output_path)
            processed_files.add(predictor_file.name)
            
    return new_predictor_paths
