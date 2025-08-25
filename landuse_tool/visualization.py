import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import leafmap.foliumap as leafmap
import streamlit as st
import seaborn as sns
import pandas as pd

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
    Creates an interactive leafmap with toggleable layers for all relevant files.
    """
    st.info("Generating interactive map... This may take a moment for the first load.")
    
    m = leafmap.Map(center=[0, 0], zoom=2, locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)

    # Add Prediction Result first so it's on top
    if prediction_filepath:
        palette = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        m.add_raster(prediction_filepath, palette=palette, layer_name="Prediction Result")

    # Add Target Layers
    if target_files:
        for file in target_files:
            m.add_raster(file, layer_name=f"Target: {file.name}")

    # Add Predictor Layers
    if predictor_files:
        for file in predictor_files:
            m.add_raster(file, layer_name=f"Predictor: {file.name}")
            
    return m
