import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import streamlit as st
import seaborn as sns
import pandas as pd
import folium
from io import BytesIO
import base64

# We need to import the memory-safe file opener from data_loader
from .data_loader import _open_as_raster

def plot_prediction(arr, cmap_list, title):
    """
    Generate a static plot of a predicted raster.
    """
    cmap = ListedColormap(cmap_list)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(arr, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Generates a heatmap figure for a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    return fig

def create_interactive_map(target_files, predictor_files, prediction_filepath):
    """
    Creates an interactive folium map with toggleable layers.
    This version is memory-safe and avoids crashes by downscaling large rasters
    before displaying them as overlays.
    """
    st.info("Generating interactive map... This may take a moment.")
    
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB positron")

    def add_raster_as_overlay(file_or_path, layer_name, palette=None):
        try:
            with _open_as_raster(file_or_path) as src:
                bounds = ((src.bounds.bottom, src.bounds.left), (src.bounds.top, src.bounds.right))
                
                # --- FIX IS HERE: DOWNSCALING LOGIC ---
                # Define a maximum dimension for the visualization thumbnail
                MAX_DIM = 512
                
                # Calculate the new shape, preserving aspect ratio
                if src.height > MAX_DIM or src.width > MAX_DIM:
                    if src.height > src.width:
                        new_height = MAX_DIM
                        new_width = int(src.width * (MAX_DIM / src.height))
                    else:
                        new_width = MAX_DIM
                        new_height = int(src.height * (MAX_DIM / src.width))
                    
                    out_shape = (new_height, new_width)
                else:
                    out_shape = (src.height, src.width)

                # Read the data, downscaling it on the fly to the new shape
                data = src.read(1, out_shape=out_shape)
                # --- END OF FIX ---

                buffer = BytesIO()
                cmap = 'viridis' if not palette else ListedColormap(palette)
                
                # Normalize for better color mapping if it's a predictor
                vmin, vmax = (np.nanmin(data), np.nanmax(data)) if not palette else (None, None)

                plt.imsave(buffer, data, cmap=cmap, format='png', vmin=vmin, vmax=vmax)
                buffer.seek(0)
                
                encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
                image_url = f'data:image/png;base64,{encoded_image}'
                
                img_overlay = folium.raster_layers.ImageOverlay(
                    image=image_url,
                    bounds=bounds,
                    opacity=0.7,
                    name=layer_name,
                    interactive=True
                )
                img_overlay.add_to(m)

        except Exception as e:
            st.warning(f"Could not display layer '{layer_name}': {e}")

    # Add Prediction Result Layer
    if prediction_filepath:
        palette = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        add_raster_as_overlay(prediction_filepath, "Prediction Result", palette=palette)

    # Add Target Layers
    if target_files:
        for file in target_files:
            add_raster_as_overlay(file, f"Target: {file.name}")

    # Add Predictor Layers
    if predictor_files:
        for file in predictor_files:
            add_raster_as_overlay(file, f"Predictor: {file.name}")
    
    folium.LayerControl().add_to(m)
            
    return m
