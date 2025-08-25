import operator
import rasterio
from rasterio.windows import Window
import tempfile
import streamlit as st
from tqdm import tqdm

# --- Import the memory-safe file opener from data_loader ---
from .data_loader import _open_as_raster

# Define allowed operators
OPS = {
    "multiply": operator.mul,
    "add": operator.add,
    "subtract": operator.sub,
    "divide": operator.truediv,
}

def apply_scenario_windowed(predictor_files, scenario_def, progress_callback=None):
    """
    Apply a user-defined scenario to predictor files in a memory-safe,
    windowed manner, creating a new set of temporary raster files.

    Parameters
    ----------
    predictor_files : list of UploadedFile
        List of original predictor file objects from Streamlit.
    scenario_def : dict
        Scenario definition with a list of changes.
    progress_callback : function, optional
        A function to report progress to the Streamlit frontend.

    Returns
    -------
    list of str
        A list of file paths to the new temporary scenario rasters.
    """
    
    scenario_filepaths = []
    
    # Create a lookup dictionary for quick access to changes
    changes_map = {change["layer"]: change for change in scenario_def.get("changes", [])}

    total_files = len(predictor_files)
    for i, p_file in enumerate(predictor_files):
        
        if progress_callback:
            progress_fraction = (i + 1) / total_files
            progress_callback(progress_fraction, f"Processing {p_file.name}...")

        # Open the original predictor to get its profile
        with _open_as_raster(p_file) as src:
            profile = src.profile
            
            # Create a new temporary file for the scenario output
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir="/tmp")
            scenario_filepaths.append(temp_file.name)
            temp_file.close()

            # Open the new temp file in write mode
            with rasterio.open(temp_file.name, 'w', **profile) as dst:
                # Process the raster window by window
                for _, window in src.block_windows(1):
                    original_data = src.read(1, window=window)
                    
                    # Check if a change needs to be applied to this layer
                    if p_file.name in changes_map:
                        change = changes_map[p_file.name]
                        op_name = change["op"]
                        value = change["value"]
                        
                        if op_name in OPS:
                            func = OPS[op_name]
                            # Apply the operation
                            modified_data = func(original_data, value)
                        else:
                            # If operator is invalid, just use original data
                            modified_data = original_data
                    else:
                        # No changes for this layer, just copy the data
                        modified_data = original_data
                    
                    # Write the (potentially modified) data to the new file
                    dst.write(modified_data, 1, window=window)

    if progress_callback:
        progress_callback(1.0, "Scenario files created successfully!")
        
    return scenario_filepaths
