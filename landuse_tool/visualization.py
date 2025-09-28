import streamlit as st
import rasterio
import numpy as np
import folium
from folium.plugins import MiniMap, LayerControl
from streamlit_folium import st_folium
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .data_loader import _open_as_raster

def get_raster_bounds(raster_file):
    """Gets the geographic bounds of a raster file."""
    with _open_as_raster(raster_file) as src:
        bounds = src.bounds
        return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

def create_downscaled_png(raster_file, downscale_factor=10):
    """
    Creates a temporary, downscaled PNG from a raster for visualization.
    Returns the path to the temporary PNG and the raster's bounds.
    """
    with _open_as_raster(raster_file) as src:
        # Calculate new shape
        new_height = src.height // downscale_factor
        new_width = src.width // downscale_factor
        
        # Read the data, downsampling by slicing
        data = src.read(1, out_shape=(new_height, new_width), resampling=rasterio.enums.Resampling.nearest)
        
        bounds = src.bounds
        image_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        
        # Create a colormap (can be customized)
        unique_vals = np.unique(data)
        cmap = plt.get_cmap('terrain', len(unique_vals))
        norm = colors.BoundaryNorm(np.arange(len(unique_vals) + 1) - 0.5, len(unique_vals))
        
        # Turn off axes and save the image
        fig, ax = plt.subplots(1, 1, figsize=(new_width/100, new_height/100))
        ax.imshow(data, cmap=cmap, norm=norm)
        ax.axis('off')
        
        temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
        temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=temp_dir)
        fig.savefig(temp_png.name, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        
        return temp_png.name, image_bounds

def create_interactive_map(target_files=[], prediction_filepath=None):
    """
    Creates a folium map with raster layers for targets and prediction.
    """
    if not target_files:
        return folium.Map(location=[0, 0], zoom_start=2)

    # Use the first target to center the map
    with _open_as_raster(target_files[0]) as src:
        center_y = (src.bounds.bottom + src.bounds.top) / 2
        center_x = (src.bounds.left + src.bounds.right) / 2

    m = folium.Map(location=[center_y, center_x], zoom_start=8, tiles="CartoDB positron")

    # Add historical target layers
    for i, file in enumerate(target_files):
        png_path, bounds = create_downscaled_png(file)
        folium.raster_layers.ImageOverlay(
            image=png_path,
            bounds=bounds,
            opacity=0.7,
            name=f"Historical LC: {file.name}",
            show=(i == len(target_files) - 1) # Show the latest by default
        ).add_to(m)

    # Add prediction layer if available
    if prediction_filepath:
        png_path, bounds = create_downscaled_png(prediction_filepath)
        folium.raster_layers.ImageOverlay(
            image=png_path,
            bounds=bounds,
            opacity=0.8,
            name="Simulated Future Land Cover",
            show=True
        ).add_to(m)

    folium.LayerControl().add_to(m)
    MiniMap().add_to(m)
    
    return m
