import rasterio
import numpy as np
import folium
from rasterio.enums import Resampling
from PIL import Image
import io
import base64
from streamlit_folium import st_folium

def plot_prediction_from_path(filepath):
    """
    DEPRECATED but kept for reference.
    Creates a simple static plot from a raster file path.
    """
    import matplotlib.pyplot as plt
    with rasterio.open(filepath) as src:
        arr = src.read(1)
        fig, ax = plt.subplots()
        im = ax.imshow(arr, cmap='terrain')
        fig.colorbar(im, ax=ax)
        return fig

def create_interactive_map(prediction_filepath):
    """
    Creates an interactive folium map with the predicted raster overlayed.
    Uses a memory-safe downscaling approach for visualization.
    """
    with rasterio.open(prediction_filepath) as src:
        # Get raster bounds in geographic coordinates
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        
        # Calculate a downscaling factor to keep the visualization lightweight
        max_dim = 1024 # Max width or height for the preview
        scale_factor = max(src.width / max_dim, src.height / max_dim)
        
        # Define the output shape for the downscaled array
        out_shape = (
            1,
            int(src.height / scale_factor),
            int(src.width / scale_factor)
        )

        # Read the data, downscaling it to the new shape
        data = src.read(
            out_shape=out_shape,
            resampling=Resampling.nearest
        )[0]

        # Normalize and colorize the data to create a PNG image in memory
        unique_vals = np.unique(data)
        # Create a colormap (can be customized)
        colormap = plt.get_cmap('terrain')
        
        # Normalize data to 0-1 range for the colormap
        normalized_data = (data - np.min(unique_vals)) / (np.max(unique_vals) - np.min(unique_vals) + 1e-6)
        colored_data = (colormap(normalized_data) * 255).astype(np.uint8)
        
        # Convert to a PIL image and save as PNG bytes
        img = Image.fromarray(colored_data, 'RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_str = base64.b64encode(buf.getvalue()).decode()
        
    # Create the Folium map centered on the raster
    center_lat = (src.bounds.bottom + src.bounds.top) / 2
    center_lon = (src.bounds.left + src.bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add the image overlay
    img_overlay = folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_str}',
        bounds=bounds,
        opacity=0.8,
        name='Predicted Land Cover'
    )
    img_overlay.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m

