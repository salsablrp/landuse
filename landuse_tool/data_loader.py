import rasterio
import numpy as np
import streamlit as st
import tempfile
import os
from contextlib import contextmanager

from .utils import create_mask

@contextmanager
def _open_as_raster(file_object_or_path):
    """
    A context manager to safely open a raster from an uploaded file or a path.
    Writes uploaded files to a temporary location to keep memory usage low.
    """
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=temp_dir) as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                while chunk := file_object_or_path.read(16 * 1024):
                    tmp.write(chunk)
            
            with rasterio.open(temp_filepath) as src:
                yield src
        else:
            with rasterio.open(str(file_object_or_path)) as src:
                yield src
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

def load_targets(target_files):
    """
    Validates all target files and creates the data mask from the last target file.
    """
    if not target_files or len(target_files) < 2:
        st.warning("Please upload at least two historical land cover maps.")
        return None, None
    try:
        # Lightweight validation
        for f in target_files:
            with _open_as_raster(f):
                pass
        
        # Create mask from the latest file for prediction step
        with _open_as_raster(target_files[-1]) as src:
            ref_profile = src.profile
            mask = create_mask(src.read(1), nodata=src.nodata)
        return ref_profile, mask
    except Exception as e:
        st.error(f"An error occurred while processing target files: {e}")
        return None, None

def load_predictors(predictor_files, ref_profile=None):
    """
    Validates that each predictor has the same dimensions as the reference profile.
    """
    if not predictor_files: return False
    if not ref_profile:
        st.error("Cannot validate predictors without a reference profile from target files.")
        return False
    ref_width, ref_height = ref_profile['width'], ref_profile['height']
    try:
        for f in predictor_files:
            with _open_as_raster(f) as src:
                if src.width != ref_width or src.height != ref_height:
                    st.error(f"Dimension mismatch: Predictor '{f.name}' ({src.width}x{src.height}) does not match target dimensions ({ref_width}x{ref_height}).")
                    return False
        return True
    except Exception as e:
        st.error(f"Error validating predictor files: {e}")
        return False

