import numpy as np
import rasterio

def predict_map(model, X_stack, mask, ref_profile, nodata=255, save_path=None):
    """
    Apply trained model to predictor stack and return classified raster.

    Args:
        model: trained sklearn model
        X_stack (np.ndarray): predictors [n_bands, H, W]
        mask (np.ndarray): boolean mask of valid pixels
        ref_profile (dict): rasterio profile from reference raster
        nodata (int, optional): value for nodata pixels
        save_path (str, optional): if given, saves GeoTIFF to this path

    Returns:
        np.ndarray: predicted raster [H, W]
    """
    # Flatten predictors
    X_flat = X_stack.reshape(X_stack.shape[0], -1).T  # shape = [n_pixels, n_bands]

    # Apply mask
    valid_mask = mask.flatten()
    X_valid = X_flat[valid_mask]

    # Predict only on valid pixels
    y_pred = model.predict(X_valid)

    # Rebuild full raster
    y_full = np.full(mask.size, nodata, dtype=np.uint8)
    y_full[valid_mask] = y_pred
    y_map = y_full.reshape(mask.shape)

    # Save to GeoTIFF if requested
    if save_path:
        profile = ref_profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=nodata)
        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(y_map, 1)

    return y_map
