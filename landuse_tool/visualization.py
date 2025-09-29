import rasterio
import numpy as np
import folium
from rasterio.enums import Resampling
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _raster_to_png_overlay(raster_file, colormap_func, nodata_val):
    """Helper function to convert a raster file to a PNG image overlay for folium."""
    with rasterio.open(raster_file) as src:
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
        
        max_dim = 1024
        scale_factor = max(src.width / max_dim, src.height / max_dim, 1)
        out_shape = (1, int(src.height / scale_factor), int(src.width / scale_factor))

        data = src.read(out_shape=out_shape, resampling=Resampling.nearest)[0]
        
        unique_vals = np.unique(data[data != nodata_val])
        colors = {val: colormap_func(i) for i, val in enumerate(range(len(unique_vals)))}
        val_map = {val: i for i, val in enumerate(unique_vals)}

        colored_data = np.zeros((*data.shape, 4), dtype=np.uint8)
        for val in unique_vals:
            mask = data == val
            color_index = val_map[val]
            colored_data[mask] = (np.array(colors[color_index]) * 255).astype(np.uint8)

        if nodata_val is not None:
             colored_data[data == nodata_val] = [0, 0, 0, 0]

        img = Image.fromarray(colored_data, 'RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        return f'data:image/png;base64,{img_str}', bounds, unique_vals, colors, val_map

def create_interactive_map(target_files_with_years=None, prediction_filepath=None, class_legends=None):
    """Creates an interactive folium map with historical and predicted layers."""
    if not target_files_with_years and not prediction_filepath:
        return folium.Map(location=[0, 0], zoom_start=2)

    ref_file = prediction_filepath or target_files_with_years[0]['file']
    with rasterio.open(ref_file) as src:
        center_lat = (src.bounds.bottom + src.bounds.top) / 2
        center_lon = (src.bounds.left + src.bounds.right) / 2
        zoom = 10
        nodata_val = src.nodata
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")

    all_unique_vals = set()
    all_colors = {}
    all_val_maps = {}
    
    layers = []
    if prediction_filepath: layers.append({'file': prediction_filepath, 'name': 'Simulated Future'})
    if target_files_with_years:
        for item in reversed(target_files_with_years):
            layers.append({'file': item['file'], 'name': f"LC {item['year']}"})

    colormap_func = plt.get_cmap('Paired', 20)

    for layer in layers:
        img_str, bounds, unique_vals, colors, val_map = _raster_to_png_overlay(layer['file'], colormap_func, nodata_val)
        all_unique_vals.update(unique_vals)
        all_colors.update(colors)
        all_val_maps.update(val_map)

        folium.raster_layers.ImageOverlay(
            image=img_str, bounds=bounds, opacity=0.8, name=layer['name'], show=(layer['name'] == 'Simulated Future')
        ).add_to(m)

    if class_legends:
        # --- IMPROVEMENT: Added color:black; to the style ---
        legend_html = '''
         <div style="position: fixed; bottom: 50px; left: 50px; width: auto; max-width: 250px;
         max-height: 400px; overflow-y: auto; padding: 10px;
         border:2px solid grey; z-index:9999; font-size:14px;
         background-color:white; color:black;
         ">&nbsp;<b>Legend</b><br>'''
        
        sorted_vals = sorted(list(all_unique_vals))
        for val in sorted_vals:
            class_name = class_legends.get(val, f"Class {val}")
            color_index = all_val_maps.get(val, 0)
            color = all_colors.get(color_index, (0,0,0,0))
            color_hex = '#%02x%02x%02x' % tuple((np.array(color[:3])*255).astype(int))
            legend_html += f'&nbsp;<i class="fa fa-square" style="color:{color_hex}"></i>&nbsp;{class_name}<br>'
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m

def create_downloadable_map_image(prediction_filepath, class_legends=None, title="Simulated Future Land Cover"):
    """
    Creates a high-quality, static map plot for download with a custom title.
    """
    with rasterio.open(prediction_filepath) as src:
        data = src.read(1)
        nodata = src.nodata
        
    unique_vals = np.unique(data[data != nodata])
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Use a colormap that matches the interactive map
    colormap_func = plt.get_cmap('Paired', 20)
    colors = {val: colormap_func(i) for i, val in enumerate(range(len(unique_vals)))}
    val_map = {val: i for i, val in enumerate(unique_vals)}
    
    # Create a discrete colormap for imshow
    cmap = plt.cm.get_cmap('Paired', len(unique_vals))
    
    im = ax.imshow(data, cmap=cmap, interpolation='nearest')
    
    # --- THIS IS THE KEY CHANGE ---
    # Use the 'title' variable passed into the function
    ax.set_title(title, fontsize=16)
    
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Create legend
    if class_legends:
        patches = [mpatches.Patch(color=cmap(val_map[val]), label=class_legends.get(val, f'Class {val}')) for val in unique_vals]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    fig.tight_layout()
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    return buf.getvalue()
