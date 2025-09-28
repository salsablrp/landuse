import rasterio
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from contextlib import ExitStack
from rasterio.windows import Window
from .data_loader import _open_as_raster

def create_transition_dataset(from_class, to_class, lc_start_file, lc_end_file, predictor_files):
    """
    Creates a balanced training dataset (X, y) for a specific transition.
    """
    try:
        # Read full maps once; this is memory-intensive but necessary for this logic.
        # Assumes the rasters can fit in memory.
        with _open_as_raster(lc_start_file) as lc_start_src:
            lc_start = lc_start_src.read(1)
        with _open_as_raster(lc_end_file) as lc_end_src:
            lc_end = lc_end_src.read(1)

        positive_mask = (lc_start == from_class) & (lc_end == to_class)
        negative_mask = (lc_start == from_class) & (lc_end == from_class)
        
        positive_coords = np.argwhere(positive_mask)
        negative_coords = np.argwhere(negative_mask)
        
        # Balance the dataset
        n_samples = min(len(positive_coords), len(negative_coords))
        if n_samples == 0:
            return None, None
            
        positive_sample_coords = positive_coords[np.random.choice(len(positive_coords), n_samples, replace=False)]
        negative_sample_coords = negative_coords[np.random.choice(len(negative_coords), n_samples, replace=False)]
        
        all_coords = np.vstack([positive_sample_coords, negative_sample_coords])
        y = np.array([1] * n_samples + [0] * n_samples)
        
        # Extract predictor values for the sampled coordinates
        X = []
        with ExitStack() as stack:
            predictors = [stack.enter_context(_open_as_raster(f)) for f in predictor_files]
            for r, c in all_coords:
                # Read a single pixel value from each predictor raster
                pixel_values = [src.read(1, window=Window(c, r, 1, 1))[0, 0] for src in predictors]
                X.append(pixel_values)
                
        return np.array(X), y
    except Exception as e:
        st.error(f"Failed to create dataset for transition {from_class}->{to_class}: {e}")
        return None, None

def train_rf_model(X, y):
    """
    Trains a Random Forest classifier and returns the model and its accuracy.
    """
    if X is None or len(X) == 0:
        return None, 0
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return model, accuracy
    except Exception as e:
        st.error(f"Model training failed: {e}")
        return None, 0

