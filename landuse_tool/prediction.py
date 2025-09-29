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
    model = joblib.load(model_path)
    to_cls = os.path.basename(model_path).split('_')[-1].split('.')[0]
    # Add model type to filename to distinguish expander/patcher maps
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
    # --- THIS IS THE CORRECTED SYNTAX ---
    # The 'def' statement is now on its own, properly indented line.
    if progress_callback is None:
        def progress_callback(p, t):
            pass
            
    try:
        suitability_paths = {}
        # Check if we are in growth mode
        growth_mode = any('_expander_' in v for v in trained_model_paths.values())

        transitions_to_model = set(k for k, v in trained_model_paths.items() if '_expander_' in v or '_patcher_' in v) if growth_mode else set(trained_model_paths.keys())

        total_models = len(trained_model_paths)
        i = 0
        for from_cls, to_cls in transitions_to_model:
            if growth_mode:
                for mode in ['expander', 'patcher']:
                    i += 1
                    progress_callback(i / total_models, f"Generating {mode} map for {from_cls}->{to_cls}")
                    model_path = trained_model_paths.get((from_cls, to_cls, mode))
                    if model_path:
                        suitability_paths[(from_cls, to_cls, mode)] = generate_suitability_map(from_cls, model_path, predictor_files, lc_end_file, temp_dir, mode)
            else:
                i += 1
                progress_callback(i / total_models, f"Generating suitability map for {from_cls}->{to_cls}")
                model_path = trained_model_paths.get((from_cls, to_cls))
                if model_path:
                    suitability_paths[(from_cls, to_cls)] = generate_suitability_map(from_cls, model_path, predictor_files, lc_end_file, temp_dir)

        progress_callback(1.0, "Suitability atlas complete. Starting simulation...")

        with _open_as_raster(lc_end_file) as src:
            future_lc = src.read(1); profile = src.profile
        
        sorted_transitions = transition_counts.stack().sort_values(ascending=False).index.tolist()
        
        for from_cls, to_cls in sorted_transitions:
            if from_cls == to_cls: continue
            demand = int(transition_counts.loc[from_cls, to_cls])
            if demand <= 0: continue

            if growth_mode:
                exp_path = suitability_paths.get((from_cls, to_cls, 'expander'))
                pat_path = suitability_paths.get((from_cls, to_cls, 'patcher'))
                if not exp_path or not pat_path: continue

                with rasterio.open(exp_path) as exp_src, rasterio.open(pat_path) as pat_src:
                    exp_suit = exp_src.read(1)
                    pat_suit = pat_src.read(1)

                # Create a combined suitability map based on proximity to existing patches
                to_class_mask = (future_lc == to_cls)
                dilated_mask = binary_dilation(to_class_mask, structure=np.ones((3,3)))
                
                # Use expander scores for adjacent pixels, patcher scores for others
                suitability_map = np.where(dilated_mask, exp_suit, pat_suit)

            else: # Standard mode
                suitability_path = suitability_paths.get((from_cls, to_cls))
                if not suitability_path: continue
                with rasterio.open(suitability_path) as src: suitability_map = src.read(1)

            available_mask = (future_lc == from_cls)
            available_scores = suitability_map[available_mask]
            available_coords = np.argwhere(available_mask)
            num_to_change = min(demand, len(available_scores))
            if num_to_change <= 0: del suitability_map; gc.collect(); continue
            
            # Stochastic or deterministic allocation
            if stochastic:
                scores_sum = np.sum(available_scores)
                if scores_sum > 0:
                    probabilities = available_scores / scores_sum
                    probabilities /= np.sum(probabilities)
                    chosen_indices = np.random.choice(len(available_coords), size=num_to_change, replace=False, p=probabilities)
                else:
                    chosen_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]
            else:
                chosen_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]

            coords_to_change = available_coords[chosen_indices]
            rows, cols = coords_to_change.T
            future_lc[rows, cols] = to_cls
            del suitability_map; gc.collect()
        
        output_path = os.path.join(temp_dir, "predicted_land_cover.tif")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(future_lc, 1)
        
        progress_callback(1.0, "Simulation complete!")
        return output_path
    except Exception as e:
        raise Exception(f"An error occurred during simulation: {e}")

