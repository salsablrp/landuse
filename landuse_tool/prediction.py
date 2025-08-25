# In prediction.py
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
import tempfile
from .data_loader import _open_as_raster

def predict_map_windowed(model, predictor_files, mask, ref_profile, window_size=512):
    """
    Generates a prediction map using windowed processing to keep memory usage low.
    Writes the output directly to a temporary GeoTIFF file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    temp_filepath = temp_file.name
    temp_file.close()

    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    predictor_sources = [_open_as_raster(f) for f in predictor_files]
    height, width = ref_profile['height'], ref_profile['width']
    nodata_val = out_profile.get('nodata') # Use .get() for safety

    with rasterio.open(temp_filepath, 'w', **out_profile) as out_src:
        for i in tqdm(range(0, height, window_size), desc="Predicting Rows"):
            # THIS IS THE CORRECTED LINE:
            for j in tqdm(range(0, width, window_size), desc="Predicting Cols", leave=False):
                
                window = Window(j, i, min(window_size, width - j), min(window_size, height - i))

                window_stack = np.array([p_src.read(1, window=window) for p_src in predictor_sources])
                
                mask_window = mask[window.row_off:window.row_off + window.height, 
                                   window.col_off:window.col_off + window.width]
                
                valid_pixels = window_stack[:, mask_window]
                valid_pixels_reshaped = valid_pixels.T

                out_window = np.full(mask_window.shape, nodata_val, dtype=np.uint8)

                if valid_pixels_reshaped.size > 0:
                    predictions = model.predict(valid_pixels_reshaped)
                    out_window[mask_window] = predictions.astype(np.uint8)
                
                out_src.write(out_window, 1, window=window)

    for src in predictor_sources:
        src.close()
        
    return temp_filepath