import streamlit as st
import os

from landuse_tool import data_loader, prediction, utils, training, scenarios, visualization

st.set_page_config(layout="wide")
st.title("🌍 Land Use Monitoring & Prediction Tool")

# --- Initialize Session State ---
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.targets_loaded = False
    st.session_state.predictors_loaded = False

defaults = {
    "targets": None,
    "predictors": None,
    "show_maps": False,
    "model": None,
    "predicted": None,
    "scenario_stack": None,
    "sample_success": False,
    "train_success": False,
    "prediction_success": False,
    "geotiff_saved": None,
    "png_saved": None,
    "scenario_changes": [],
    "uploaded_targets": [],
    "uploaded_predictors": [],
    "ref_profile": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Reset Workflow ---
def reset_all():
    st.session_state.clear()
    st.session_state.step = 0
    st.info("Workflow has been reset.")

st.sidebar.button("🔄 Reset Workflow", on_click=reset_all)

# --- Sidebar Navigation ---
steps = [
    "🏡 Home",
    "1️⃣ Upload Data",
    "2️⃣ Training",
    "3️⃣ Prediction",
    "4️⃣ Scenarios",
    "5️⃣ Visualization"
]

st.sidebar.markdown("### 📋 Navigation")
for i, label in enumerate(steps):
    if st.sidebar.button(label):
        st.session_state.active_step = i

if "active_step" not in st.session_state:
    st.session_state.active_step = 0

# --- Step 0: Landing Page ---
if st.session_state.active_step == 0:
    st.header("👋 Welcome to the Land Use Monitoring & Prediction Tool")
    st.markdown("""
    This tool helps you analyze and forecast land cover change using remote sensing data.
    
    **Workflow Overview:**
    1. Upload historical land cover and predictor rasters.
    2. Train a model on these datasets.
    3. Predict future land cover based on current conditions.
    4. Simulate different scenarios by modifying predictors.
    5. Visualize and export the results.

    👈 Use the sidebar to begin with **Step 1**.
    """)

# --- Step 1: Upload Data ---
elif st.session_state.active_step == 1:
    st.header("Step 1: Upload Data")

    st.markdown("""
    In this step, you will upload the required raster datasets:

    - **Land cover targets** from at least two different years.
    - **Predictor variables**, such as elevation, slope, distance to roads, etc.

    These datasets will be aligned and prepared for training in the next step.
    """)

    st.subheader("1a. Upload Land Cover Targets (≥2 years)")
    uploaded_targets = st.file_uploader(
        "Upload land cover rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="targets_uploader"
    )

    if uploaded_targets and uploaded_targets != st.session_state.uploaded_targets:
        st.session_state.uploaded_targets = uploaded_targets
        st.session_state.targets_loaded = False
        st.session_state.predictors_loaded = False

    if st.session_state.uploaded_targets and not st.session_state.targets_loaded:
        with st.spinner("Processing targets..."):
            try:
                profiles, mask = data_loader.load_targets(st.session_state.uploaded_targets, align=True)
                if profiles:
                    st.session_state.ref_profile = profiles[-1]
                    st.session_state.mask = mask
                    st.session_state.targets_loaded = True
                    st.success("Successfully processed targets.")
                else:
                    st.session_state.targets_loaded = False
                    st.error("Target processing failed. Check the files for errors.")
            except Exception as e:
                st.session_state.targets_loaded = False
                st.error(f"An unexpected error occurred during target processing: {e}")
                
    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader(
        "Upload predictor rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="predictors_uploader"
    )

    if uploaded_predictors and uploaded_predictors != st.session_state.uploaded_predictors:
        st.session_state.uploaded_predictors = uploaded_predictors
        st.session_state.predictors_loaded = False

    if st.session_state.uploaded_predictors and st.session_state.ref_profile and not st.session_state.predictors_loaded:
            with st.spinner("Processing predictors..."):
                try:
                    # The function now returns True or False
                    is_valid = data_loader.load_predictors(
                        st.session_state.uploaded_predictors,
                        st.session_state.ref_profile
                    )
                    if is_valid:
                        # We no longer store the giant array, just mark as loaded/validated
                        st.session_state.predictors_loaded = True
                        st.success(f"Validated {len(st.session_state.uploaded_predictors)} predictor files successfully.")
                    else:
                        st.session_state.predictors_loaded = False
                        st.error("Predictor validation failed. Check the files for errors.")
                except Exception as e:
                    st.session_state.predictors_loaded = False
                    st.error(f"An unexpected error occurred during predictor processing: {e}")
                    
    if st.session_state.targets_loaded and st.session_state.predictors_loaded:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Proceed to Training"):
                st.session_state.step = 1
                st.session_state.active_step = 2
        with col2:
            if st.button("🗺️ Visualize Maps"):
                st.session_state.show_maps = True

    if st.session_state.get("show_maps"):
        st.subheader("🗺️ Map Preview")
        # Note: Visualizing uploaded rasters on leafmap requires more logic
        # For now, we'll just show the map without the data.
        import leafmap
        m = leafmap.Map(center=[0, 0], zoom=2)
        m.to_streamlit(height=500)

# --- Step 2: Training ---
elif st.session_state.active_step == 2:
    st.header("Step 2: Sample and Train Model")
    # ... (markdown text)

    if not st.session_state.targets_loaded or not st.session_state.predictors_loaded:
        st.warning("⚠️ No files found. Go back to Step 1 and upload them.")
    else:
        if st.button("📥 Sample Training Data"):
            with st.spinner("Sampling training data... This may take a while for large files."):
                try:
                    # CORRECTED FUNCTION NAME
                    X, y = data_loader.sample_training_data(
                        target_files=st.session_state.uploaded_targets,
                        predictor_files=st.session_state.uploaded_predictors,
                        ref_profile=st.session_state.ref_profile
                    )
                    if X is not None and y is not None:
                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.sample_success = True
                        st.success(f"Sampled {len(X)} training points.")
                    else:
                        st.session_state.sample_success = False
                        st.error("Sampling failed. Check error messages above.")
                except Exception as e:
                    st.error(f"Sampling failed: {e}")

    if st.session_state.sample_success:
        if st.button("⚡ Train Random Forest"):
            with st.spinner("Training model..."):
                model, metrics = training.train_rf(st.session_state.X, st.session_state.y)
                st.session_state.model = model
                st.session_state.train_success = True
                st.success("Model trained successfully.")

    if st.session_state.train_success:
        if st.button("➡️ Proceed to Prediction"):
            st.session_state.step = 2
            st.session_state.active_step = 3

# --- Step 3: Prediction ---
elif st.session_state.active_step == 3:
    st.header("Step 3: Predict Land Cover")

    st.markdown("""
    Use the trained model to:

    - Predict land cover for a future or unseen time period.
    - Visualize and export the results in raster formats (GeoTIFF, PNG).

    Optionally, run predictions using modified scenarios from Step 4.
    """)

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
    else:
        if st.button("🛰️ Run Prediction"):
            with st.spinner("Running prediction..."):
                # CORRECTED: Use the new function and session state variables
                predicted_filepath = prediction.predict_map_windowed(
                    model=st.session_state.model,
                    predictor_files=st.session_state.uploaded_predictors, # Use the file list
                    mask=st.session_state.mask, # Use the mask from session state
                    ref_profile=st.session_state.ref_profile
                )
                # Store the FILEPATH of the result, not the array
                st.session_state.predicted_filepath = predicted_filepath
                st.session_state.prediction_success = True
                st.success("Prediction complete.")

    if st.session_state.predicted is not None:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🖼️ Save as PNG"):
                out = visualization.save_prediction_as_png(st.session_state.predicted)
                st.success(f"Saved as {out}")
        with col2:
            if st.button("💾 Save as GeoTIFF"):
                out = visualization.save_prediction_as_tif(
                    st.session_state.predicted,
                    st.session_state.ref_profile
                )
                st.success(f"Saved as {out}")

        if st.button("➡️ Proceed to Scenarios"):
            st.session_state.step = 3
            st.session_state.active_step = 4

# --- Step 4: Scenarios ---
elif st.session_state.active_step == 4:
    st.header("Step 4: Run Scenarios")

    st.markdown("""
    Simulate alternative futures by:

    - Modifying predictor layers (e.g., increase population, deforestation).
    - Applying arithmetic operations to specific rasters.

    The modified predictor stack can be used for scenario-based predictions in Step 3.
    """)

    if not st.session_state.predictors_loaded:
        st.warning("⚠️ No predictor files found. Go back to Step 1 and upload them.")
    else:
        predictor_filenames = [f.name for f in st.session_state.uploaded_predictors]

        scenario_name = st.text_input("Scenario name", "My Scenario")

        st.markdown("### ➕ Add a Change")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            selected_layer = st.selectbox("Layer", options=predictor_filenames)

        with col2:
            selected_op = st.selectbox("Operator", options=["multiply", "add", "subtract", "divide"])

        with col3:
            value = st.number_input("Value", value=1.0)

        if st.button("➕ Add Change"):
            st.session_state.scenario_changes.append({
                "layer": selected_layer,
                "op": selected_op,
                "value": value
            })
            st.success(f"Added: `{selected_layer}` {selected_op} {value}")

        if st.session_state.scenario_changes:
            st.markdown("### ✅ Current Scenario Changes")
            for i, change in enumerate(st.session_state.scenario_changes):
                st.write(f"{i+1}. `{change['layer']}` **{change['op']}** {change['value']}")

            if st.button("🧹 Clear All Changes"):
                st.session_state.scenario_changes = []
                st.success("Cleared all scenario changes.")

        if st.button("🌱 Apply Scenario"):
            scenario_stack = scenarios.apply_scenario(
                stack=st.session_state.predictors,
                uploaded_predictors=st.session_state.uploaded_predictors,
                scenario_def={
                    "name": scenario_name,
                    "changes": st.session_state.scenario_changes
                }
            )
            st.session_state.scenario_stack = scenario_stack
            st.success(f"✅ Scenario '{scenario_name}' applied.")
            st.info("Return to Step 3 to run prediction with the scenario.")

        if st.button("➡️ Proceed to Visualization"):
            st.session_state.step = 4
            st.session_state.active_step = 5

# --- Step 5: Visualization ---
elif st.session_state.active_step == 5:
    st.header("Step 5: Visualization")

    st.markdown("""
    Here you can:

    - View predictions as inline maps or images.
    - Export the predicted land cover as PNG or GeoTIFF.
    - Compare baseline predictions with scenario outcomes (if available).
    """)

    if st.session_state.predicted is None:
        st.warning("⚠️ No prediction available. Please go back to Step 3 to generate one.")
    else:
        st.success("✅ Prediction is ready for visualization.")
        cmap_list = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        title = "Predicted Land Cover Map"

        if st.button("📊 Show Inline Map"):
            fig = visualization.plot_prediction(st.session_state.predicted, cmap_list, title)
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save as GeoTIFF"):
                out_tif = visualization.save_prediction_as_tif(
                    st.session_state.predicted,
                    st.session_state.ref_profile,
                    "prediction.tif"
                )
                st.success(f"Saved as `{out_tif}`")

        with col2:
            if st.button("🖼️ Save as PNG"):
                out_png = visualization.save_prediction_as_png(
                    st.session_state.predicted,
                    cmap_list,
                    "prediction.png"
                )
                st.success(f"Saved as `{out_png}`")
                st.image(out_png, caption="Prediction PNG Preview")

        if st.session_state.scenario_stack is not None:
            if st.button("🛰️ Run Prediction on Scenario"):
                st.info("Running scenario-based prediction...")
                scenario_pred = prediction_ori.predict_map(
                    model=st.session_state.model,
                    X_stack=st.session_state.scenario_stack,
                    mask=None,
                    ref_profile=st.session_state.ref_profile
                )
                fig = visualization.plot_prediction(
                    scenario_pred,
                    cmap_list,
                    f"{title} (Scenario)"
                )
                st.pyplot(fig)
