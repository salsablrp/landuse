# training.py

import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from contextlib import ExitStack
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
import joblib  # <-- NEW: Import joblib
import json    # <-- NEW: Import json
import os      # <-- NEW: Import os

from .data_loader import _open_as_raster

def create_transition_dataset(from_cls, to_cls, lc_start_file, lc_end_file, predictor_files, mode='all'):
    """
    Creates a targeted, balanced training dataset for a specific transition.
    Includes logic for 'expander' and 'patcher' growth modes.
    """
    try:
        with _open_as_raster(lc_start_file) as src_start, _open_as_raster(lc_end_file) as src_end:
            lc_start = src_start.read(1)
            lc_end = src_end.read(1)
        
        positive_mask = (lc_start == from_cls) & (lc_end == to_cls)
        negative_mask = (lc_start == from_cls) & (lc_end == from_cls)
        
        positive_coords = np.argwhere(positive_mask)
        
        if mode != 'all':
            initial_to_class_mask = (lc_start == to_cls)
            dilated_mask = binary_dilation(initial_to_class_mask, structure=np.ones((3,3)))
            expander_mask = dilated_mask & positive_mask
            
            if mode == 'expander':
                positive_coords = np.argwhere(expander_mask)
            elif mode == 'patcher':
                patcher_mask = ~dilated_mask & positive_mask
                positive_coords = np.argwhere(patcher_mask)

        negative_coords = np.argwhere(negative_mask)
        
        n_samples = min(len(positive_coords), len(negative_coords))
        if n_samples < 10: return None, None
            
        max_samples = 50000 
        n_samples = min(n_samples, max_samples)

        positive_sample_coords = positive_coords[np.random.choice(len(positive_coords), n_samples, replace=False)]
        negative_sample_coords = negative_coords[np.random.choice(len(negative_coords), n_samples, replace=False)]
        
        all_coords = np.vstack([positive_sample_coords, negative_sample_coords])
        y = np.array([1] * n_samples + [0] * n_samples)
        
        X = []
        with ExitStack() as stack:
            predictors = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            for r, c in all_coords:
                pixel_values = [p.read(1, window=Window(c, r, 1, 1))[0, 0] for p in predictors]
                X.append(pixel_values)
                
        return np.array(X), y
    except Exception:
        return None, None

def train_rf_model(X, y, model_path, predictor_schema): # <-- MODIFIED: Added new arguments
    """
    Trains a Random Forest model, saves it, saves its feature schema, and returns accuracy.
    """
    if X is None or y is None or len(X) == 0: return None, 0
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        # --- NEW: Save the model and its feature schema ---
        # 1. Save the trained model object
        joblib.dump(model, model_path)
        
        # 2. Save the list of predictor names (the schema) to a corresponding JSON file
        features_path = model_path.replace(".joblib", "_features.json")
        with open(features_path, 'w') as f:
            json.dump(predictor_schema, f, indent=4)
        # --- End of new code ---

        return accuracy # <-- MODIFIED: Only return accuracy now
    except Exception:
        return 0 # <-- MODIFIED: Simplified return on error