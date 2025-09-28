import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
import tempfile
from contextlib import ExitStack

from .data_loader import _open_as_raster

# --- FIX IS HERE: Added 'progress_callback=None' ---
def predict_map_windowed(model, predictor_files, mask, ref_profile, window_size=512, progress_callback=None):
    """
    Generates a prediction map using windowed processing to keep memory usage low.
    Writes the output directly to a temporary GeoTIFF file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir="/tmp")
    temp_filepath = temp_file.name
    temp_file.close()

    out_profile = ref_profile.copy()
    out_profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    height, width = ref_profile['height'], ref_profile['width']
    
    nodata_val = out_profile.get('nodata')
    
    if nodata_val is None or not (0 <= nodata_val <= 255):
        nodata_val = 255
    
    out_profile['nodata'] = nodata_val

    with ExitStack() as stack:
        predictor_sources = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
        out_src = stack.enter_context(rasterio.open(temp_filepath, 'w', **out_profile))

        # --- And here, we use the callback inside the loop ---
        for i in range(0, height, window_size):
            if progress_callback:
                progress_fraction = i / height
                progress_callback(progress_fraction, f"Predicting... {int(progress_fraction*100)}% complete")

            for j in range(0, width, window_size):
                
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

    if progress_callback:
        progress_callback(1.0, "Prediction finished!")

    return temp_filepath
