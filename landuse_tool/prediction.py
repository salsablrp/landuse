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

def _generate_single_suitability_map(from_class, model_path, predictor_files, lc_end_file, temp_dir, model_type=''):
    """Helper function to generate one suitability map."""
    model = joblib.load(model_path)
    to_cls_str = os.path.basename(model_path).split('_')[-1].split('.')[0]
    filename = f'suitability_{model_type}_{from_class}_to_{to_cls_str}.tif' if model_type else f'suitability_{from_class}_to_{to_cls_str}.tif'
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
        # --- MODIFIED: Ensure predictors are opened in the correct order based on the model's training schema ---
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

def generate_suitability_atlas(predictor_files, lc_end_file, trained_model_paths, temp_dir, progress_callback=None):
    if progress_callback is None: def progress_callback(p, t): pass
    
    suitability_paths = {}
    growth_mode = any(len(k) == 3 for k in trained_model_paths.keys())
    transitions_to_model = set((k[0], k[1]) for k in trained_model_paths.keys()) if growth_mode else set(trained_model_paths.keys())

    total_maps_to_generate = len(trained_model_paths)
    i = 0
    for from_cls, to_cls in transitions_to_model:
        if growth_mode:
            for mode in ['expander', 'patcher']:
                i += 1
                progress_callback(i / total_maps_to_generate, f"Generating {mode} map for {from_cls}->{to_cls}")
                model_path = trained_model_paths.get((from_cls, to_cls, mode))
                if model_path:
                    suitability_paths[(from_cls, to_cls, mode)] = _generate_single_suitability_map(from_cls, model_path, predictor_files, lc_end_file, temp_dir, mode)
        else:
            i += 1
            progress_callback(i / total_maps_to_generate, f"Generating suitability map for {from_cls}->{to_cls}")
            model_path = trained_model_paths.get((from_cls, to_cls))
            if model_path:
                suitability_paths[(from_cls, to_cls)] = _generate_single_suitability_map(from_cls, model_path, predictor_files, lc_end_file, temp_dir)
    
    progress_callback(1.0, "Suitability atlas complete!")
    return suitability_paths

def run_allocation_simulation(lc_end_file, transition_counts, suitability_paths, temp_dir, stochastic=False):
    growth_mode = any(len(k) == 3 for k in suitability_paths.keys())

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
                exp_suit, pat_suit = exp_src.read(1), pat_src.read(1)
            to_class_mask = (future_lc == to_cls)
            dilated_mask = binary_dilation(to_class_mask, structure=np.ones((3,3)))
            suitability_map = np.where(dilated_mask, exp_suit, pat_suit)
        else:
            suitability_path = suitability_paths.get((from_cls, to_cls))
            if not suitability_path: continue
            with rasterio.open(suitability_path) as src: suitability_map = src.read(1)

        available_mask = (future_lc == from_cls)
        available_scores = suitability_map[available_mask]
        available_coords = np.argwhere(available_mask)
        num_to_change = min(demand, len(available_scores))
        if num_to_change <= 0: del suitability_map; gc.collect(); continue
        
        if stochastic:
            non_zero_indices = np.where(available_scores > 0)[0]
            size_for_choice = min(num_to_change, len(non_zero_indices))
            if size_for_choice > 0:
                scores_for_choice = available_scores[non_zero_indices]
                probs_for_choice = scores_for_choice / np.sum(scores_for_choice)
                chosen_subset_indices = np.random.choice(len(non_zero_indices), size=size_for_choice, replace=False, p=probs_for_choice)
                top_indices = non_zero_indices[chosen_subset_indices]
            else:
                top_indices = []
        else:
            top_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]

        if len(top_indices) > 0:
            coords_to_change = available_coords[top_indices]
            rows, cols = coords_to_change.T
            future_lc[rows, cols] = to_cls
        del suitability_map; gc.collect()
    
    output_path = os.path.join(temp_dir, "predicted_land_cover.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(future_lc, 1)
    
    return output_path

