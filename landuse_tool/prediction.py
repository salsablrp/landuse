# In prediction.py
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
import tempfile # To create a temporary output file
from .data_loader import _open_as_raster # Use your optimized opener

def predict_map_windowed(model, predictor_files, mask, ref_profile, window_size=512):
    """
    Generates a prediction map using windowed processing to keep memory usage low.
    Writes the output directly to a temporary GeoTIFF file.
    """
    # Create a temporary file to store the output raster.
    # It will be automatically deleted when closed unless we persist it.
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    temp_filepath = temp_file.name
    temp_file.close() # Close it so rasterio can write to it.

    # Update the profile for the output file
    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw') # Or appropriate dtype

    # Open all predictor sources
    predictor_sources = [_open_as_raster(f) for f in predictor_files]
    height, width = ref_profile['height'], ref_profile['width']
    nodata_val = out_profile['nodata']

    # Open the output file in write mode and process window by window
    with rasterio.open(temp_filepath, 'w', **out_profile) as out_src:
        for i in tqdm(range(0, height, window_size), desc="Predicting Rows"):
            for j in range(0, width, window_size), desc="Predicting Cols", leave=False):
                
                window = Window(j, i, min(window_size, width - j), min(window_size, height - i))

                # Read the window from each predictor
                # The shape will be (bands, window_height, window_width)
                window_stack = np.array([p_src.read(1, window=window) for p_src in predictor_sources])
                
                # Get the mask for the current window
                mask_window = mask[window.row_off:window.row_off + window.height, 
                                   window.col_off:window.col_off + window.width]
                
                # Reshape data for prediction: (n_pixels, n_bands)
                # We only want to predict on valid pixels
                valid_pixels = window_stack[:, mask_window]
                valid_pixels_reshaped = valid_pixels.T # Transpose to (n_pixels, n_bands)

                # Initialize output window with nodata
                out_window = np.full(mask_window.shape, nodata_val, dtype=np.uint8)

                if valid_pixels_reshaped.size > 0:
                    # Run prediction
                    predictions = model.predict(valid_pixels_reshaped)
                    # Place predictions back into the window at the correct locations
                    out_window[mask_window] = predictions.astype(np.uint8)
                
                # Write the processed window to the output file
                out_src.write(out_window, 1, window=window)

    # Close all predictor files
    for src in predictor_sources:
        src.close()
        
    # Return the path to the completed GeoTIFF
    return temp_filepath