import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio

def plot_prediction(map_array, cmap_list, title="Prediction Map", show=True):
    """
    Plot the prediction map with a custom colormap.
    Works in both Jupyter and Streamlit (when passed to st.pyplot).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(map_array, cmap=ListedColormap(cmap_list))
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    if show:
        plt.show()
    return fig


def save_prediction_as_tif(map_array, profile, out_path="prediction.tif"):
    """
    Save prediction array as GeoTIFF using original profile.
    """
    new_profile = profile.copy()
    new_profile.update(dtype=rasterio.uint8, count=1)
    
    with rasterio.open(out_path, "w", **new_profile) as dst:
        dst.write(map_array.astype(rasterio.uint8), 1)
    
    return out_path


def save_prediction_as_png(map_array, cmap_list, out_path="prediction.png"):
    """
    Save prediction as PNG (non-georeferenced).
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(map_array, cmap=ListedColormap(cmap_list))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
