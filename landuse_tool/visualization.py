import rasterio
import numpy as np
import folium
from rasterio.enums import Resampling
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import branca.colormap as cm

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

def create_interactive_map(prediction_filepath, class_legends=None):
    """
    Creates an interactive folium map with the predicted raster overlayed,
    including a proper legend if class names are provided.
    """
    with rasterio.open(prediction_filepath) as src:
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        
        max_dim = 1024
        scale_factor = max(src.width / max_dim, src.height / max_dim, 1)
        
        out_shape = (1, int(src.height / scale_factor), int(src.width / scale_factor))

        data = src.read(out_shape=out_shape, resampling=Resampling.nearest)[0]
        
        unique_vals = np.unique(data)
        unique_vals = unique_vals[unique_vals != src.nodata] # Exclude nodata from legend

        # Use a qualitative colormap if we have few classes, otherwise use a continuous one
        if len(unique_vals) <= 12:
            colormap_func = plt.get_cmap('Paired', len(unique_vals))
        else:
            colormap_func = plt.get_cmap('terrain', len(unique_vals))

        # Create a mapping from class value to color
        colors = {val: colormap_func(i) for i, val in enumerate(unique_vals)}
        
        # Create the colored image array
        colored_data = np.zeros((*data.shape, 4), dtype=np.uint8)
        for val, color in colors.items():
            mask = data == val
            colored_data[mask] = (np.array(color) * 255).astype(np.uint8)

        # Handle nodata transparency
        if src.nodata is not None:
             colored_data[data == src.nodata] = [0, 0, 0, 0] # Make nodata transparent

        img = Image.fromarray(colored_data, 'RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_str = base64.b64encode(buf.getvalue()).decode()
        
    center_lat = (src.bounds.bottom + src.bounds.top) / 2
    center_lon = (src.bounds.left + src.bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    img_overlay = folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_str}',
        bounds=bounds,
        opacity=0.8,
        name='Predicted Land Cover'
    )
    img_overlay.add_to(m)
    
    # --- Add Legend ---
    if class_legends:
        legend_html = '''
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 180px; height: auto; 
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white;
         ">&nbsp;<b>Legend</b><br>
         '''
        for val in unique_vals:
            class_name = class_legends.get(val, f"Class {val}")
            color_hex = '#%02x%02x%02x' % tuple((np.array(colors[val][:3])*255).astype(int))
            legend_html += f'&nbsp;<i class="fa fa-square" style="color:{color_hex}"></i>&nbsp;{class_name}<br>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    
    return m