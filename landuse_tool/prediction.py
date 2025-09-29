import rasterio
import numpy as np
import tempfile
import os
import joblib
from contextlib import ExitStack
from rasterio.windows import Window
import gc
from scipy.ndimage import binary_dilation

from .data_loader import _open_as_raster

def generate_suitability_map(from_class, model_path, predictor_files, lc_end_file, temp_dir, model_type=''):
    """Generates a probability map for a specific transition."""
    model = joblib.load(model_path)
    to_cls = os.path.basename(model_path).split('_')[-1].split('.')[0]
    filename = f'suitability_{model_type}_{from_class}_to_{to_cls}.tif' if model_type else f'suitability_{from_class}_to_{to_cls}.tif'
    temp_filepath = os.path.join(temp_dir, filename)

    with _open_as_raster(lc_end_file) as ref_src:
        ref_arr = ref_src.read(1); profile = ref_src.profile
        profile.update(dtype='float32', count=1, nodata=-1.0)
        from_mask = (ref_arr == from_class)
        from_coords = np.argwhere(from_mask)
        if from_coords.size == 0:
            with rasterio.open(temp_filepath, 'w', **profile) as dst:
                dst.write(np.full(ref_arr.shape, -1.0, dtype='float32'), 1)
            return temp_filepath
        suitability_map = np.full(ref_arr.shape, -1.0, dtype='float32')

    batch_size = 50000
    with ExitStack() as stack:
        predictors = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
        for i in range(0, len(from_coords), batch_size):
            batch_coords = from_coords[i:i+batch_size]
            X_batch = []
            for r, c in batch_coords:
                pixel_values = [p_src.read(1, window=Window(c, r, 1, 1))[0, 0] for p_src in predictors]
                X_batch.append(pixel_values)
            if X_batch:
                probs = model.predict_proba(np.array(X_batch))[:, 1]
                rows, cols = batch_coords.T
                suitability_map[rows, cols] = probs
    with rasterio.open(temp_filepath, 'w', **profile) as dst:
        dst.write(suitability_map, 1)
    return temp_filepath

def run_simulation(lc_end_file, predictor_files, transition_counts, trained_model_paths, temp_dir, stochastic=False, progress_callback=None):
    """
    Runs the full simulation using a memory-safe, windowed approach for allocation.
    """
    if progress_callback is None:
        def progress_callback(p, t): pass

    try:
        # --- STAGE 2: Generate Suitability Atlas (This part is already memory-safe) ---
        suitability_paths = {}
        transitions_to_model = list(trained_model_paths.keys())
        total_suitability_maps = len(transitions_to_model)
        for i, key in enumerate(transitions_to_model):
            progress_callback(i / (total_suitability_maps + 1), f"Generating suitability map {i+1}/{total_suitability_maps}")
            model_path = trained_model_paths.get(key)
            from_cls = key[0]
            model_type = key[2] if len(key) == 3 else ''
            if model_path:
                suitability_paths[key] = generate_suitability_map(from_cls, model_path, predictor_files, lc_end_file, temp_dir, model_type)

        progress_callback(total_suitability_maps / (total_suitability_maps + 1), "Suitability atlas complete. Starting final allocation...")

        # --- STAGE 3: Windowed Cellular Automata Simulation ---
        with _open_as_raster(lc_end_file) as src:
            profile = src.profile
            output_path = os.path.join(temp_dir, "predicted_land_cover.tif")
            # Copy the latest LC map to the output file to serve as the base
            with rasterio.open(output_path, 'w', **profile) as dst:
                for _, window in src.block_windows(1):
                    dst.write(src.read(1, window=window), 1, window=window)

        sorted_transitions = transition_counts.stack().sort_values(ascending=False).index.tolist()
        
        with ExitStack() as stack:
            suitability_sources = {key: stack.enter_context(rasterio.open(path)) for key, path in suitability_paths.items() if path and os.path.exists(path)}
            
            with rasterio.open(output_path, 'r+') as future_lc_src:
                for from_cls, to_cls in sorted_transitions:
                    if from_cls == to_cls: continue
                    demand = int(transition_counts.loc[from_cls, to_cls])
                    if demand <= 0: continue
                    
                    growth_mode = (from_cls, to_cls, 'expander') in suitability_sources
                    
                    # Find all suitable pixels across the entire map by reading in windows
                    all_available_scores = []
                    all_available_coords = []
                    
                    for _, window in future_lc_src.block_windows(1):
                        future_lc_window = future_lc_src.read(1, window=window)
                        
                        if growth_mode:
                            exp_suit = suitability_sources[(from_cls, to_cls, 'expander')].read(1, window=window)
                            pat_suit = suitability_sources[(from_cls, to_cls, 'patcher')].read(1, window=window)
                            to_class_mask = (future_lc_window == to_cls)
                            dilated_mask = binary_dilation(to_class_mask, structure=np.ones((3,3)))
                            suitability_window = np.where(dilated_mask, exp_suit, pat_suit)
                        else:
                            suit_src = suitability_sources.get((from_cls, to_cls))
                            if not suit_src: continue
                            suitability_window = suit_src.read(1, window=window)

                        available_mask = (future_lc_window == from_cls)
                        scores = suitability_window[available_mask]
                        coords = np.argwhere(available_mask)
                        
                        global_coords = coords + np.array([window.row_off, window.col_off])
                        all_available_scores.append(scores)
                        all_available_coords.append(global_coords)
                    
                    all_available_scores = np.concatenate(all_available_scores)
                    all_available_coords = np.concatenate(all_available_coords)
                    
                    num_to_change = min(demand, len(all_available_scores))
                    if num_to_change <= 0: continue
                    
                    top_indices = np.argpartition(all_available_scores, -num_to_change)[-num_to_change:]
                    coords_to_change = all_available_coords[top_indices]
                    
                    # Update the output file pixel by pixel (slow but extremely memory-safe)
                    for r, c in coords_to_change:
                        future_lc_src.write(np.array([[to_cls]], dtype=profile['dtype']), 1, window=Window(c, r, 1, 1))

        progress_callback(1.0, "Simulation complete!")
        return output_path

    except Exception as e:
        raise Exception(f"An error occurred during simulation: {e}")

