import streamlit as st
import leafmap
import numpy as np
import rasterio

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
    arrays, masks, profiles = data_loader.load_targets(uploaded_targets, align=True)
    st.session_state.targets = (arrays, masks, profiles)
    st.session_state.profiles = profiles
    st.success(f"Loaded {len(arrays)} target rasters.")


st.subheader("1b. Upload Predictor Rasters")
uploaded_predictors = st.file_uploader(
    "Upload predictor rasters (e.g., elevation.tif, distance_to_roads.tif)",
    type=["tif", "tiff"],
    accept_multiple_files=True,
    key="predictors_uploader"
)

if uploaded_predictors and st.session_state.targets:
    ref_profile = st.session_state.profiles[0]
    predictors = data_loader.load_predictors(uploaded_predictors, ref_profile)
    st.session_state.predictors = predictors
    st.success(f"Loaded predictors stack with shape {predictors.shape}")
    
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
    target_paths = st.session_state.get("target_paths", [])
    predictor_paths = st.session_state.get("predictor_paths", [])

    # Load data for consistency
    arrays, masks, profiles = st.session_state.targets
    predictors = st.session_state.predictors

    # --- Conditional Map Preview ---
    if st.session_state.get("show_maps", False):
        st.subheader("🗺️ Rasters Preview")

        m = leafmap.Map(center=[0, 0], zoom=2)

        # Add target rasters
        for i, path in enumerate(target_paths):
            name = os.path.basename(path)
            try:
                m.add_raster(path, layer_name=f"Target {i+1}")
            except Exception as e:
                st.error(f"Error loading target raster {i+1}: {e}")

        # Add predictor rasters
        for i, path in enumerate(predictor_paths):
            name = os.path.basename(path)
            try:
                m.add_raster(path, layer_name=f"Predictor {i+1}")
            except Exception as e:
                st.error(f"Error loading predictor raster {i+1}: {e}")

        m.to_streamlit(height=500)


# --- STEP 2: Sampling & Training ---
if st.session_state.step >= 1:
    st.header("Step 2: Sample and Train Model")

    # --- Sampling ---
    if st.button("📥 Sample Training Data"):
        with st.spinner("Sampling training data..."):

            target_paths = st.session_state.get("target_paths", [])
            predictor_paths = st.session_state.get("predictor_paths", [])

            if not target_paths or not predictor_paths:
                st.error("❌ Missing target or predictor raster paths.")
            else:
                latest_target_path = target_paths[-1]  # use most recent year

                try:
                    X, y = data_loader.sample_training_data(
                        target_path=latest_target_path,
                        predictor_paths=predictor_paths,
                        total_samples=5000,
                        window_size=512
                    )
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.sample_success = True  # ✅ Save success flag
                except Exception as e:
                    st.error(f"Sampling failed: {e}")
                    st.session_state.sample_success = False

    # --- Show sampling success ---
    if st.session_state.get("sample_success", False):
        st.success(f"✅ Sampled {len(st.session_state.X)} training points from raster.")

    # --- Training ---
    if "X" in st.session_state and "y" in st.session_state:
        if st.button("⚡ Train Random Forest"):
            with st.spinner("Training model..."):
                model, metrics = training.train_rf(
                    st.session_state.X,
                    st.session_state.y
                )
                st.session_state.model = model
                st.session_state.train_success = True  # ✅ Save success flag

    # --- Show training success ---
    if st.session_state.get("train_success", False):
        st.success("✅ Model trained!")

    # --- Proceed ---
    if st.button("➡️ Proceed to Prediction"):
        st.session_state.step = 2


# --- STEP 3: Prediction ---
if st.session_state.step >= 2:
    st.header("Step 3: Predict Land Cover")

    # Show success messages
    if st.session_state.get("prediction_success"):
        st.success("✅ Prediction complete!")

    if st.session_state.get("geotiff_saved"):
        st.success(f"✅ Saved GeoTIFF: `{st.session_state.geotiff_saved}`")

    if st.session_state.get("png_saved"):
        st.success(f"✅ Saved PNG: `{st.session_state.png_saved}`")

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
    else:
        if st.button("🛰️ Run Prediction"):
            arrays, masks, profiles = st.session_state.targets
            mask = masks[-1]
            ref_profile = profiles[-1]

            if "scenario_stack" in st.session_state:
                predictors = st.session_state.scenario_stack
                st.info("📊 Using scenario-adjusted predictors for prediction.")
            else:
                predictors = st.session_state.predictors
                st.info("📊 Using original predictors for prediction.")

            # Reset all success states on re-run
            st.session_state.prediction_success = False
            st.session_state.png_saved = None
            st.session_state.geotiff_saved = None

            with st.spinner("Running prediction..."):
                progress_bar = st.progress(0.0)

                def update_progress(p):
                    progress_bar.progress(p)

                predicted = prediction.predict_map(
                    model=st.session_state.model,
                    X_stack=predictors,
                    mask=mask,
                    ref_profile=ref_profile,
                    progress_callback=update_progress
                )

                st.session_state.predicted = predicted
                st.session_state.ref_profile = ref_profile
            
            st.success("✅ Prediction complete!")
            st.session_state.prediction_success = True

            # Optional: clear scenario after use
            st.session_state.pop("scenario_stack", None)

    if "predicted" in st.session_state:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            if st.button("🖼️ Save as PNG"):
                path = visualization.save_prediction_as_png(
                    st.session_state.predicted
                )
                st.session_state.png_saved = os.path.basename(path)

                st.success(f"✅ Saved PNG: `{st.session_state.png_saved}`")

        with col2:
            if st.button("💾 Save as GeoTIFF"):
                path = visualization.save_prediction_as_tif(
                    st.session_state.predicted,
                    st.session_state.ref_profile,
                    temp=False
                )
                st.session_state.geotiff_saved = os.path.basename(path)

                st.success(f"✅ Saved GeoTIFF: `{st.session_state.geotiff_saved}`")

        with col3:
            if st.button("➡️ Proceed to Scenarios"):
                st.session_state.step = 3

        with col4:
            if st.button("🗺️ Visualize Predicted Map"):
                m = leafmap.Map(center=[48.0, 67.0], zoom=4)
                vis_path = visualization.save_prediction_as_tif(
                    st.session_state.predicted,
                    st.session_state.ref_profile,
                    temp=True
                )
                m.add_raster(vis_path, layer_name="Predicted Land Cover")
                m.to_streamlit(height=500)


# --- STEP 4: Scenarios ---
if st.session_state.step >= 3:
    st.header("Step 4: Run Scenarios")

    predictor_files = st.session_state.get("predictor_paths", [])
    predictor_filenames = [os.path.basename(p) for p in predictor_files]

    if "scenario_changes" not in st.session_state:
        st.session_state.scenario_changes = []

    scenario_name = st.text_input("Scenario name", "My Scenario")

    st.markdown("### ➕ Add a Change")
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_layer = st.selectbox("Layer", options=predictor_filenames, key="selected_layer")

    with col2:
        selected_op = st.selectbox("Operator", options=["multiply", "add", "subtract", "divide"], key="selected_op")

    with col3:
        value = st.number_input("Value", value=1.0, key="selected_value")

    if st.button("➕ Add Change"):
        st.session_state.scenario_changes.append({
            "layer": selected_layer,
            "op": selected_op,
            "value": value
        })

    if st.session_state.scenario_changes:
        st.markdown("### ✅ Current Changes:")
        for i, change in enumerate(st.session_state.scenario_changes):
            st.write(f"{i+1}. `{change['layer']}` **{change['op']}** {change['value']}")

        if st.button("🧹 Clear All Changes"):
            st.session_state.scenario_changes = []

    if st.button("🌱 Apply Scenario"):
        from landuse_tool.scenarios import apply_scenario
        scenario_stack = scenarios.apply_scenario(
            stack=st.session_state.predictors,
            predictor_files=predictor_filenames,
            scenario_def={
                "name": scenario_name,
                "changes": st.session_state.scenario_changes
            }
        )
        st.session_state.scenario_stack = scenario_stack
        st.session_state.prediction_success = False  # Reset old success flags
        st.success(f"✅ Scenario '{scenario_name}' applied!")
        st.info("➡️ Now return to Step 3 and run prediction.")

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