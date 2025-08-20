# landuse_tool/training.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_rf(
    X,
    y,
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    save_path=None,
    verbose=True,
):
    """
    Train a Random Forest classifier on predictor/target data.

    Args:
        X (np.ndarray): Predictor features (n_samples, n_features).
        y (np.ndarray): Target labels (n_samples,).
        test_size (float): Proportion for test set.
        random_state (int): Seed for reproducibility.
        n_estimators (int): Number of RF trees.
        save_path (str): Optional path to save trained model (.joblib).
        verbose (bool): Whether to print validation results.

    Returns:
        model: Trained RandomForestClassifier.
        dict: Validation metrics (classification_report + confusion_matrix).
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train RF
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Validation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {"report": report, "confusion_matrix": cm}

    if verbose:
        print("Validation Results:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", cm)

    # Feature importance
    feature_importance = model.feature_importances_
    metrics["feature_importance"] = feature_importance

    # Optionally save
    if save_path:
        joblib.dump(model, save_path)
        if verbose:
            print(f"✅ Model saved to {save_path}")

    return model, metrics
