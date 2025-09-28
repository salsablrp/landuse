import rasterio
import numpy as np
import tempfile
import os
import joblib
from contextlib import ExitStack
from rasterio.windows import Window

from .data_loader import _open_as_raster

def generate_suitability_map(from_class, model, predictor_files, lc_end_file, temp_dir):
    """Generates a probability map for a specific transition."""
    temp_filepath = os.path.join(temp_dir, f'suitability_{from_class}.tif')

    with _open_as_raster(lc_end_file) as ref_src:
        ref_arr = ref_src.read(1)
        profile = ref_src.profile
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
    Runs the full simulation with optional stochastic allocation.
    """
    if progress_callback is None:
        def progress_callback(p, t): pass

    try:
        suitability_paths = {}
        significant_transitions = list(trained_model_paths.keys())
        
        total_models = len(significant_transitions)
        for i, (from_cls, to_cls) in enumerate(significant_transitions):
            progress_text = f"Generating suitability map for {from_cls} -> {to_cls} ({i+1}/{total_models})"
            progress_callback((i / total_models) * 0.5, progress_text) # Suitability is first 50%
            
            model_path = trained_model_paths.get((from_cls, to_cls))
            if not model_path: continue
            
            model = joblib.load(model_path)
            
            suitability_map_path = generate_suitability_map(
                from_class=from_cls, model=model, predictor_files=predictor_files,
                lc_end_file=lc_end_file, temp_dir=temp_dir
            )
            suitability_paths[(from_cls, to_cls)] = suitability_map_path

        progress_callback(0.5, "Suitability atlas complete. Starting allocation...")

        with _open_as_raster(lc_end_file) as src:
            future_lc = src.read(1)
            profile = src.profile
        
        sorted_transitions = transition_counts.stack().sort_values(ascending=False).index.tolist()
        
        total_transitions_to_allocate = len([t for t in sorted_transitions if t[0] != t[1]])
        current_transition_idx = 0

        for from_cls, to_cls in sorted_transitions:
            if from_cls == to_cls: continue
            
            current_transition_idx += 1
            progress_text = f"Allocating change for {from_cls} -> {to_cls} ({current_transition_idx}/{total_transitions_to_allocate})"
            progress_callback(0.5 + (current_transition_idx / total_transitions_to_allocate) * 0.5, progress_text)

            demand = int(transition_counts.loc[from_cls, to_cls])
            if demand <= 0: continue
            
            suitability_path = suitability_paths.get((from_cls, to_cls))
            if not suitability_path: continue
            
            with rasterio.open(suitability_path) as src:
                suitability_map = src.read(1)
            
            available_mask = (future_lc == from_cls)
            available_scores = suitability_map[available_mask]
            available_coords = np.argwhere(available_mask)
            
            num_to_change = min(demand, len(available_scores))
            if num_to_change <= 0: continue
            
            if stochastic:
                # Normalize scores to be probabilities for np.random.choice
                scores_sum = np.sum(available_scores)
                probabilities = available_scores / scores_sum if scores_sum > 0 else None
                chosen_indices = np.random.choice(len(available_coords), size=num_to_change, replace=False, p=probabilities)
            else:
                # Deterministic: choose the top N highest scores
                chosen_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]
            
            coords_to_change = available_coords[chosen_indices]
            rows, cols = coords_to_change.T
            future_lc[rows, cols] = to_cls
        
        output_path = os.path.join(temp_dir, "predicted_land_cover.tif")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(future_lc, 1)
        
        progress_callback(1.0, "Simulation complete!")
        return output_path

    except Exception as e:
        raise Exception(f"An error occurred during simulation: {e}")

