import rasterio
import numpy as np
import os
import tempfile
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import streamlit as st

from .data_loader import _open_as_raster

def modify_predictor_raster(input_raster_file, operator, value, temp_dir):
    """
    Applies a simple arithmetic operation to a raster and saves it as a new
    temporary file. This is done in a memory-safe, windowed manner.
    """
    try:
        # Create a unique name for the new temporary file
        output_name = f"scenario_{operator.lower()}_{value}_{input_raster_file.name}"
        output_path = os.path.join(temp_dir, output_name)

        with _open_as_raster(input_raster_file) as src:
            profile = src.profile
            profile.update(dtype='float32') # Update to float to handle decimals

            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process the raster in chunks (windows)
                for _, window in src.block_windows(1):
                    window_data = src.read(1, window=window).astype('float32')
                    
                    # Apply the selected operation
                    if operator == 'Multiply':
                        result_data = window_data * value
                    elif operator == 'Add':
                        result_data = window_data + value
                    elif operator == 'Subtract':
                        result_data = window_data - value
                    elif operator == 'Divide':
                        if value != 0:
                            result_data = window_data / value
                        else:
                            st.warning("Division by zero attempted. Skipping this operation.")
                            result_data = window_data # Return original data on error
                    else:
                        result_data = window_data
                    
                    dst.write(result_data, 1, window=window)
        
        return output_path
    except Exception as e:
        st.error(f"Failed to modify predictor '{input_raster_file.name}': {e}")
        return None

def generate_neighborhood_predictors(base_raster_file, temp_dir, radius_pixels=5):
    """
    Generates neighborhood-based predictor rasters (e.g., density) for the 
    Spatially-Aware AI feature.
    """
    new_predictor_paths = []
    
    try:
        with _open_as_raster(base_raster_file) as src:
            profile = src.profile
            profile.update(dtype='float32', count=1, nodata=-1.0)
            lc_arr = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                unique_classes = np.unique(lc_arr[lc_arr != nodata])
            else:
                unique_classes = np.unique(lc_arr)

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
    except Exception as e:
        st.error(f"Failed to generate neighborhood predictors: {e}")
        return []

