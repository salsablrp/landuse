import rasterio
import numpy as np
import streamlit as st
from contextlib import ExitStack
from rasterio.windows import Window
import tempfile
import os
from .data_loader import _open_as_raster

def generate_suitability_map(from_class, model, predictor_files, ref_lc_file, progress_callback=None):
    """
    Generates a probability map (0.0 to 1.0) for a specific transition.
    """
    temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=temp_dir)
    output_path = temp_file.name
    temp_file.close()

    try:
        with _open_as_raster(ref_lc_file) as ref_src:
            profile = ref_src.profile
            profile.update(dtype='float32', count=1, compress='lzw')
            
            from_mask = (ref_src.read(1) == from_class)
            from_coords = np.argwhere(from_mask)
            
            if len(from_coords) == 0:
                return None # No source pixels left to analyze

            suitability_map = np.full(from_mask.shape, -1.0, dtype='float32')

        batch_size = 50000
        total_batches = (len(from_coords) + batch_size - 1) // batch_size

        with ExitStack() as stack:
            predictors = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_coords = from_coords[start_idx:end_idx]
                
                X_batch = [ [src.read(1, window=Window(c, r, 1, 1))[0, 0] for src in predictors] for r, c in batch_coords ]
                
                if X_batch:
                    probs = model.predict_proba(X_batch)[:, 1]
                    rows, cols = batch_coords.T
                    suitability_map[rows, cols] = probs
                
                if progress_callback:
                    progress_callback(float(i + 1) / total_batches)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(suitability_map, 1)

        return output_path

    except Exception as e:
        st.error(f"Failed to generate suitability map for class {from_class}: {e}")
        return None

def run_simulation(lc_end_file, transition_counts, suitability_paths, progress_callback=None):
    """
    Runs the Cellular Automata simulation to generate a future land cover map.
    """
    temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False, dir=temp_dir)
    output_path = temp_file.name
    temp_file.close()
    
    try:
        with _open_as_raster(lc_end_file) as src:
            future_lc = src.read(1)
            profile = src.profile

        sorted_transitions = transition_counts.stack().sort_values(ascending=False).index.tolist()
        
        total_transitions = len(sorted_transitions)
        for i, (from_cls, to_cls) in enumerate(sorted_transitions):
            if from_cls == to_cls: continue
                
            demand = int(transition_counts.loc[from_cls, to_cls])
            if demand == 0: continue
            
            if progress_callback:
                progress_callback(float(i) / total_transitions, f"Simulating: {from_cls} -> {to_cls}...")

            suitability_path = suitability_paths.get((from_cls, to_cls))
            if not suitability_path: continue
            
            with _open_as_raster(suitability_path) as src:
                suitability_map = src.read(1)
                
            available_mask = (future_lc == from_cls)
            available_scores = suitability_map[available_mask]
            available_coords = np.argwhere(available_mask)
            
            num_to_change = min(demand, len(available_scores))
            if num_to_change == 0: continue
            
            top_indices = np.argpartition(available_scores, -num_to_change)[-num_to_change:]
            coords_to_change = available_coords[top_indices]
            rows, cols = coords_to_change.T
            future_lc[rows, cols] = to_cls
            
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(future_lc, 1)

        if progress_callback:
            progress_callback(1.0, "Simulation complete!")

        return output_path
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None

