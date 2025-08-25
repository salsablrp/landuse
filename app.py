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
    "ref_profile": None,
    "predicted_filepath": None
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
        st.session_state.predictors_loaded = False # Reset predictors if targets change

    if st.session_state.uploaded_targets and not st.session_state.targets_loaded:
        with st.spinner("Processing targets..."):
            try:
                ref_profile, mask = data_loader.load_targets(st.session_state.uploaded_targets, align=True)
                if ref_profile and mask is not None:
                    st.session_state.ref_profile = ref_profile
                    st.session_state.mask = mask
                    st.session_state.targets_loaded = True
                else:
                    st.session_state.targets_loaded = False
            except Exception as e:
                st.session_state.targets_loaded = False
                st.error(f"An unexpected error occurred during target processing: {e}")

    # Display success message based on session state
    if st.session_state.targets_loaded:
        st.success("✅ Successfully processed targets.")
                
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

    if st.session_state.uploaded_predictors and not st.session_state.targets_loaded:
        st.info("Please upload the target files first. Predictors will be validated after the targets are loaded.")

    if st.session_state.uploaded_predictors and st.session_state.ref_profile and not st.session_state.predictors_loaded:
        with st.spinner("Validating predictors..."):
            try:
                is_valid = data_loader.load_predictors(
                    st.session_state.uploaded_predictors,
                    st.session_state.ref_profile
                )
                if is_valid:
                    st.session_state.predictors_loaded = True
                else:
                    st.session_state.predictors_loaded = False
                    st.error("Predictor validation failed. Check the files for errors.")
            except Exception as e:
                st.session_state.predictors_loaded = False
                st.error(f"An unexpected error occurred during predictor processing: {e}")
    
    # Display success message based on session state
    if st.session_state.predictors_loaded:
        st.success(f"✅ Validated {len(st.session_state.uploaded_predictors)} predictor files successfully.")
                    
    if st.session_state.targets_loaded and st.session_state.predictors_loaded:
        if st.button("➡️ Proceed to Training"):
            st.session_state.active_step = 2
            st.rerun()

# --- Step 2: Training ---
elif st.session_state.active_step == 2:
    st.header("Step 2: Sample and Train Model")
    
    if not st.session_state.targets_loaded or not st.session_state.predictors_loaded:
        st.warning("⚠️ No files found. Go back to Step 1 and upload them.")
    else:
        if st.button("📥 Sample Training Data"):
            with st.spinner("Sampling training data... This may take a while for large files."):
                try:
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
            st.session_state.active_step = 3
            st.rerun()

# --- Step 3: Prediction ---
elif st.session_state.active_step == 3:
    st.header("Step 3: Predict Land Cover")

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first.")
    else:
        if st.button("🛰️ Run Prediction"):
            with st.spinner("Running prediction..."):
                predicted_filepath = prediction.predict_map_windowed(
                    model=st.session_state.model,
                    predictor_files=st.session_state.uploaded_predictors,
                    mask=st.session_state.mask,
                    ref_profile=st.session_state.ref_profile
                )
                st.session_state.predicted_filepath = predicted_filepath
                st.session_state.prediction_success = True
                st.success("Prediction complete.")

    if st.session_state.prediction_success:
        st.info(f"Prediction saved to temporary file: {st.session_state.predicted_filepath}")
        if st.button("➡️ Proceed to Visualization"):
            st.session_state.active_step = 5
            st.rerun()


# --- Step 4: Scenarios (Placeholder) ---
elif st.session_state.active_step == 4:
    st.header("Step 4: Run Scenarios")
    st.info("Scenario modeling is under development.")


# --- Step 5: Visualization ---
elif st.session_state.active_step == 5:
    st.header("Step 5: Visualization")

    if not st.session_state.prediction_success or not st.session_state.predicted_filepath:
        st.warning("⚠️ No prediction available. Please go back to Step 3 to generate one.")
    else:
        st.success("✅ Prediction is ready for visualization.")
        cmap_list = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        title = "Predicted Land Cover Map"

        if st.button("📊 Show Inline Map"):
            with rasterio.open(st.session_state.predicted_filepath) as src:
                predicted_array = src.read(1)
            fig = visualization.plot_prediction(predicted_array, cmap_list, title)
            st.pyplot(fig)

