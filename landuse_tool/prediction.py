import numpy as np
import rasterio

def predict_map(model, X_stack, mask, ref_profile, nodata=255, save_path=None, batch_size=50_000, progress_callback=None):
    """
    Apply trained model to predictor stack and return classified raster.
    """
    X_flat = X_stack.reshape(X_stack.shape[0], -1).T
    valid_mask = mask.flatten()
    X_valid = X_flat[valid_mask]

    # Predict in batches
    y_pred = []
    total = X_valid.shape[0]

    for i in range(0, total, batch_size):
        batch = X_valid[i:i+batch_size]
        y_pred.append(model.predict(batch))

        # Call progress callback (0.0 to 1.0)
        if progress_callback:
            progress_callback(min((i + batch_size) / total, 1.0))

    y_pred = np.concatenate(y_pred)

    # Fill output
    y_full = np.full(mask.size, nodata, dtype=np.uint8)
    y_full[valid_mask] = y_pred
    y_map = y_full.reshape(mask.shape)

    # Optional save
    if save_path:
        profile = ref_profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=nodata)
        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(y_map, 1)

    return y_map
