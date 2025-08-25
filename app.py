import streamlit as st
import os
import leafmap

# Make sure to import your corrected data_loader
from landuse_tool import data_loader, training, prediction, scenarios, visualization

st.set_page_config(layout="wide")
st.title("🌍 Land Use Monitoring & Prediction Tool")

# --- Initialize Session State ---
# Using a function to initialize is cleaner
def initialize_session():
    defaults = {
        "step": 0,
        "active_step": 0,
        "targets": None,
        "predictors": None,
        "profiles": None, # Storing profiles is important
        "show_maps": False,
        "model": None,
        "predicted": None,
        "scenario_stack": None,
        "sample_success": False,
        "train_success": False,
        "prediction_success": False,
        "scenario_changes": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session()

# --- Reset Workflow ---
def reset_all():
    # Keep track of the current active step before clearing
    current_step = st.session_state.get('active_step', 0)
    st.session_state.clear()
    initialize_session() # Re-initialize with defaults
    st.session_state.active_step = current_step # Optionally, stay on the same page

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
    if st.sidebar.button(label, key=f"nav_{i}"):
        st.session_state.active_step = i

# --- Page Content based on Active Step ---

# --- Step 0: Landing Page ---
if st.session_state.active_step == 0:
    st.header("👋 Welcome to the Land Use Monitoring & Prediction Tool")
    st.markdown("""
    This tool helps you analyze and forecast land cover change using remote sensing data.
    **Workflow Overview:**
    1.  **Upload Data:** Provide historical land cover and predictor rasters.
    2.  **Training:** Train a machine learning model on the uploaded data.
    3.  **Prediction:** Predict future land cover.
    4.  **Scenarios:** Simulate changes by modifying predictors.
    5.  **Visualization:** View and export your results.

    👈 Use the sidebar to begin with **Step 1: Upload Data**.
    """)

# --- Step 1: Upload Data ---
elif st.session_state.active_step == 1:
    st.header("Step 1: Upload Data")
    st.info("Upload your target (land cover) and predictor (e.g., elevation, slope) raster files.")

    st.subheader("1a. Upload Land Cover Targets (≥2 years)")
    uploaded_targets = st.file_uploader(
        "Upload land cover rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="targets_uploader"
    )

    if uploaded_targets:
        with st.spinner("Processing target rasters..."):
            try:
                arrays, masks, profiles = data_loader.load_targets(uploaded_targets, align=True)
                st.session_state.targets = (arrays, masks, profiles)
                st.session_state.profiles = profiles # Save profiles for later use
                st.success(f"✅ Loaded and processed {len(arrays)} target raster(s).")
            except Exception as e:
                st.error(f"Failed to process target files: {e}")


    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader(
        "Upload predictor rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="predictors_uploader"
    )

    if uploaded_predictors:
        if st.session_state.profiles:
            with st.spinner("Processing and aligning predictor rasters..."):
                try:
                    ref_profile = st.session_state.profiles[0]
                    predictors = data_loader.load_predictors(uploaded_predictors, ref_profile)
                    st.session_state.predictors = predictors
                    st.success(f"✅ Loaded and stacked {len(uploaded_predictors)} predictors. Stack shape: {predictors.shape}")
                except Exception as e:
                    st.error(f"Failed to process predictor files: {e}")
        else:
            st.warning("Please upload target files first to set the reference grid.")

    if st.session_state.targets and st.session_state.predictors is not None:
        st.markdown("---")
        if st.button("➡️ Proceed to Training"):
            st.session_state.active_step = 2
            st.rerun() # Use rerun for cleaner page transitions

# Note: The rest of the steps (2-5) would remain largely the same,
# as they operate on the data already loaded into st.session_state.
# The key was fixing the initial data loading step.
# Make sure the rest of your app.py file is included below.

# --- Placeholder for other steps to make the app runnable ---
elif st.session_state.active_step > 1:
    st.header(f"Step {st.session_state.active_step}: {steps[st.session_state.active_step]}")
    st.info("This section is a placeholder. Integrate the rest of your app logic here.")
    if not st.session_state.targets or st.session_state.predictors is None:
        st.warning("⚠️ No data found. Please go back to Step 1 and upload your files.")

