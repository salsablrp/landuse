import streamlit as st
import leafmap
import numpy as np
import rasterio

import tempfile
import os
from landuse_tool import data_loader, utils, training, prediction, scenarios, visualization

st.set_page_config(layout="wide")
st.title("🌍 Land Use Monitoring & Prediction Tool")

# --- SESSION STATE RESET ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "targets" not in st.session_state:
    st.session_state.targets = None
if "predictors" not in st.session_state:
    st.session_state.predictors = None
if "show_maps" not in st.session_state:
    st.session_state.show_maps = False
if "model" not in st.session_state:
    st.session_state.model = None
if "predicted" not in st.session_state:
    st.session_state.predicted = None
if "scenario_stack" not in st.session_state:
    st.session_state.scenario_stack = None

def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

st.sidebar.button("🔄 Reset Workflow", on_click=reset_all)


# --- STEP 1: Upload Data ---
st.header("Step 1: Upload Data")

st.subheader("1a. Upload Land Cover Targets (≥2 years)")
uploaded_targets = st.file_uploader(
    "Upload land cover rasters (e.g., landcover_2020.tif, landcover_2021.tif)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key="targets_uploader"
)

if uploaded_targets:
    if len(uploaded_targets) < 2:
        st.warning("⚠️ Please upload at least 2 land cover rasters from different years.")
    else:
        st.session_state.targets = data_loader.load_targets(uploaded_targets)
        st.success(f"Loaded {len(uploaded_targets)} land cover rasters.")

st.subheader("1b. Upload Predictor Rasters")
uploaded_predictors = st.file_uploader(
    "Upload predictor rasters (e.g., elevation.tif, distance_to_roads.tif)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key="predictors_uploader"
)

if uploaded_predictors:
    if st.session_state.targets is None:
        st.error("⚠️ Please upload target raster(s) before predictors.")
    else:
        _, _, target_profiles = st.session_state.targets
        ref_profile = target_profiles[0]
        st.session_state.predictors = data_loader.load_predictors(
            uploaded_predictors, ref_profile=ref_profile
        )
        st.success(f"Loaded {len(uploaded_predictors)} predictor rasters.")

# --- Save uploaded files to temp and return paths ---
def save_uploaded_files(uploaded_files):
    tmp_dir = tempfile.mkdtemp()
    file_paths = []
    for f in uploaded_files:
        out_path = os.path.join(tmp_dir, f.name)
        with open(out_path, "wb") as fp:
            # f.seek(0)
            fp.write(f.read())
        file_paths.append(out_path)
    return file_paths


# --- Main Upload Handling ---
if uploaded_targets and uploaded_predictors:

    # Show action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Proceed to Training"):
            st.session_state.step = 1
    with col2:
        if st.button("🗺️ Visualize Targets & Predictors Maps"):
            st.session_state.show_maps = True

    # Save files once
    target_paths = save_uploaded_files(uploaded_targets)
    predictor_paths = save_uploaded_files(uploaded_predictors)

    # Load data for consistency
    arrays, masks, profiles = data_loader.load_targets(target_paths)
    predictors = data_loader.load_predictors(predictor_paths, ref_profile=profiles[0])

    # --- Conditional Map Preview ---
    if st.session_state.get("show_maps", False):
        st.subheader("🗺️ Rasters Preview")

        m = leafmap.Map(center=[0, 0], zoom=2)

        # Add target rasters
        for i, path in enumerate(target_paths):
            try:
                m.add_raster(path, layer_name=f"Target {i+1}")
            except Exception as e:
                st.error(f"Error loading target raster {i+1}: {e}")

        # Add predictor rasters
        for i, path in enumerate(predictor_paths):
            try:
                m.add_raster(path, layer_name=f"Predictor {i+1}")
            except Exception as e:
                st.error(f"Error loading predictor raster {i+1}: {e}")

        m.to_streamlit(height=500)

# --- STEP 2: Training ---
if st.session_state.step >= 1:
    st.header("Step 2: Train Model")
    if st.button("⚡ Train Random Forest"):
        # Use latest target for training
        arrays, masks, profiles = st.session_state.targets
        latest_target = arrays[-1]
        latest_mask = masks[-1]
        predictors = st.session_state.predictors

        X, y = data_loader.prepare_training_data(predictors, latest_target, latest_mask)
        model = training.train_rf(X, y)
        st.session_state.model = model
        st.success("✅ Model trained!")

    if st.button("➡️ Proceed to Prediction"):
        st.session_state.step = 2


# --- STEP 3: Prediction ---
if st.session_state.step >= 2:
    st.header("Step 3: Predict Land Cover")
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
    else:
        if st.button("🛰️ Run Prediction"):
            arrays, masks, profiles = st.session_state.targets
            _, _, ref_profile = st.session_state.targets
            mask = masks[-1]

            predictors = st.session_state.predictors
            predicted = prediction.predict_map(st.session_state.model, predictors, mask)
            st.session_state.predicted = predicted
            vis_path = visualization.save_prediction_as_tif(predicted, ref_profile[-1])
            m = leafmap.Map(center=[48.0, 67.0], zoom=4)
            m.add_raster(vis_path, layer_name="Predicted Land Cover")
            m.to_streamlit(height=500)
            st.success("Prediction complete!")

    if st.button("➡️ Proceed to Scenarios"):
        st.session_state.step = 3


# --- STEP 4: Scenarios ---
if st.session_state.step >= 3:
    st.header("Step 4: Run Scenarios")

    scenario_name = st.text_input("Scenario name", "Afforestation")
    layer_ops = st.text_area(
        "Define layer adjustments (JSON-style dict)",
        """{
    "distance_to_roads": { "operator": "increase", "value": 1000 },
    "biomass": { "operator": "increase", "value": 0.2 }
}"""
    )

    if st.button("🌱 Apply Scenario"):
        predictors = st.session_state.predictors
        import json
        adjustments = json.loads(layer_ops)
        scenario_stack = scenarios.apply_scenario(predictors, adjustments)
        st.session_state.scenario_stack = scenario_stack
        st.success(f"Scenario '{scenario_name}' applied!")

    if st.button("➡️ Proceed to Visualization"):
        st.session_state.step = 4


# --- STEP 5: Visualization ---
if st.session_state.step >= 4:
    st.header("Step 5: Visualization")

    if st.session_state.predicted is not None:
        st.success("✅ Prediction ready for visualization.")
        _, _, profiles = st.session_state.targets

        # Buttons for visualization
        cmap_list = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        title = "Predicted Land Cover Map"

        if st.button("📊 Show Inline Map"):
            fig = visualization.plot_prediction(st.session_state.predicted, cmap_list, title)
            st.pyplot(fig)

        if st.button("💾 Save as GeoTIFF"):
            out_tif = visualization.save_prediction_as_tif(
                st.session_state.predicted, profiles[-1], "prediction.tif"
            )
            st.success(f"Saved as {out_tif}")
            m = leafmap.Map(center=[48.0, 67.0], zoom=4)
            m.add_raster(out_tif, layer_name="Predicted Map")
            m.to_streamlit()

        if st.button("🖼️ Save as PNG"):
            out_png = visualization.save_prediction_as_png(
                st.session_state.predicted, cmap_list, "prediction.png"
            )
            st.success(f"Saved as {out_png}")
            st.image(out_png, caption="Prediction PNG Preview")

        if st.session_state.scenario_stack is not None:
            if st.button("🛰️ Run Prediction on Scenario"):
                scenario_pred = prediction.predict_map(
                    st.session_state.model, st.session_state.scenario_stack, mask=None
                )
                fig = visualization.plot_prediction(scenario_pred, cmap_list, f"{title} (Scenario)")
                st.pyplot(fig)