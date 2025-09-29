import rasterio
import numpy as np
import os
import tempfile
from rasterio.windows import Window
import streamlit as st

from .data_loader import _open_as_raster

def modify_predictor_raster(input_raster_file, operator, value, temp_dir):
    """
    Applies a simple arithmetic operation to a raster and saves it as a new
    temporary file. This is done in a memory-safe, windowed manner.

    Args:
        input_raster_file (UploadedFile): The base predictor to modify.
        operator (str): The operation to perform ('Multiply', 'Add', etc.).
        value (float): The value to use in the operation.
        temp_dir (str): The directory to save the temporary output file.

    Returns:
        str: The file path to the newly created, modified predictor raster.
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
                    
                    dst.write(result_data, 1, window=window)
        
        return output_path
    except Exception as e:
        st.error(f"Failed to modify predictor '{input_raster_file.name}': {e}")
        return None

