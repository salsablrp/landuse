import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio

import tempfile

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


def save_prediction_as_tif(array, ref_profile, temp=True, out_path=None):
    """
    Save predicted map as GeoTIFF.

    Args:
        array (np.ndarray): Predicted map [H, W]
        ref_profile (dict): Raster profile to match
        temp (bool): If True, saves to temp file
        out_path (str): Optional path to save to if not temp

    Returns:
        str: path to saved file
    """
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=255)

    if temp:
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    elif not out_path:
        out_path = "prediction_output.tif"

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(rasterio.uint8), 1)

    return out_path


def save_prediction_as_png(array, out_path="prediction_output.png"):
    """
    Save predicted map as a PNG.

    Args:
        array (np.ndarray): Predicted map [H, W]
        out_path (str): PNG output file

    Returns:
        str: path to saved file
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(array, cmap="tab20", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return out_path