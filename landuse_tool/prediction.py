import numpy as np

def predict_map(model, X_stack, mask, nodata=255):
    X_flat = X_stack.reshape(X_stack.shape[0], -1).T
    valid_mask = mask.flatten()
    X_valid = X_flat[valid_mask]

    y_pred = model.predict(X_valid)

    y_full = np.full(mask.size, nodata, dtype=np.uint8)
    y_full[valid_mask] = y_pred
    return y_full.reshape(mask.shape)
