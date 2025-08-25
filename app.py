import streamlit as st
import os
import rasterio
import time

# --- Import your custom modules ---
from landuse_tool import data_loader, prediction, utils, training, scenarios, visualization

# --- Page Configuration ---
st.set_page_config(
    page_title="Land Use Prediction Tool",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Land Use Monitoring & Prediction Tool")

# --- Initialize Session State ---
# Use a function for cleaner initialization
def initialize_session_state():
    defaults = {
        "active_step": "Home",
        "targets_loaded": False,
        "predictors_loaded": False,
        "sample_success": False,
        "train_success": False,
        "prediction_success": False,
        "uploaded_targets": [],
        "uploaded_predictors": [],
        "ref_profile": None,
        "mask": None,
        "X": None,
        "y": None,
        "model": None,
        "predicted_filepath": None,
        "log": [] # To store persistent success messages
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()

# --- Helper Functions ---
def reset_workflow():
    """Clears the session state to start over."""
    for key in st.session_state.keys():
        del st.session_state[key]
    initialize_session_state()
    st.success("Workflow has been reset.")
    time.sleep(1) # Give user time to see the message
    st.rerun()

def get_file_size_mb(file_list):
    """Calculates the total size of uploaded files in MB."""
    total_size = 0
    for file in file_list:
        total_size += file.size
    return total_size / (1024 * 1024)

def remove_file(file_list_key, file_to_remove):
    """Removes a specific file from the session state list."""
    st.session_state[file_list_key] = [f for f in st.session_state[file_list_key] if f.name != file_to_remove.name]
    # Reset subsequent steps if files change
    st.session_state.targets_loaded = False
    st.session_state.predictors_loaded = False
    st.session_state.sample_success = False
    st.session_state.train_success = False
    st.session_state.prediction_success = False
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    
    # 3. Highlighted Sidebar Navigation
    steps = ["Home", "Upload Data", "Training", "Prediction", "Visualization"]
    icons = ["house", "upload", "bar-chart-steps", "map", "image"]
    
    # Using a radio button for selection naturally highlights the active choice
    st.session_state.active_step = st.radio(
        "Steps", 
        steps, 
        index=steps.index(st.session_state.active_step),
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # 4. Storage Meter
    st.header("Session Storage")
    total_mb = get_file_size_mb(st.session_state.uploaded_targets) + get_file_size_mb(st.session_state.uploaded_predictors)
    # Assuming Streamlit Cloud's 1GB RAM limit as a reference
    STORAGE_LIMIT_MB = 1000 
    st.progress(min(total_mb / STORAGE_LIMIT_MB, 1.0))
    st.caption(f"{total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")

    st.divider()
    st.button("🔄 Reset Workflow", on_click=reset_workflow, use_container_width=True)

# --- Main Page Content ---

# 1. Persistent Success Messages
# Display all logged success messages at the top of the page
for msg in st.session_state.log:
    st.success(msg)

# --- Page 0: Home ---
if st.session_state.active_step == "Home":
    st.header("👋 Welcome!")
    st.markdown("""
    This tool helps you analyze and forecast land cover change using remote sensing data.
    
    **Workflow Overview:**
    1. **Upload Data:** Provide historical land cover and predictor rasters.
    2. **Training:** Sample data and train a machine learning model.
    3. **Prediction:** Generate a future land cover map.
    4. **Visualization:** View and analyze your results.

    👈 Use the sidebar to begin.
    """)

# --- Page 1: Upload Data ---
elif st.session_state.active_step == "Upload Data":
    st.header("Step 1: Upload Data")

    st.subheader("1a. Upload Land Cover Targets (≥2 years)")
    uploaded_targets = st.file_uploader(
        "Upload land cover rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="targets_uploader"
    )

    if uploaded_targets:
        st.session_state.uploaded_targets = uploaded_targets
        with st.spinner("Processing targets..."):
            ref_profile, mask = data_loader.load_targets(st.session_state.uploaded_targets)
            if ref_profile and mask is not None:
                st.session_state.ref_profile = ref_profile
                st.session_state.mask = mask
                if not st.session_state.targets_loaded:
                    st.session_state.log.append("✅ Targets processed successfully.")
                    st.session_state.targets_loaded = True
                    st.rerun()

    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader(
        "Upload predictor rasters",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        key="predictors_uploader"
    )

    if uploaded_predictors:
        st.session_state.uploaded_predictors = uploaded_predictors
        if st.session_state.targets_loaded:
            with st.spinner("Validating predictors..."):
                is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, st.session_state.ref_profile)
                if is_valid and not st.session_state.predictors_loaded:
                    st.session_state.log.append(f"✅ {len(st.session_state.uploaded_predictors)} predictors validated.")
                    st.session_state.predictors_loaded = True
                    st.rerun()
        else:
            st.warning("Please upload and process target files before predictors.")

    st.divider()

    # 5. Layer Management System
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Uploaded Target Layers", expanded=True):
            if not st.session_state.uploaded_targets:
                st.caption("No target files uploaded.")
            for f in st.session_state.uploaded_targets:
                c1, c2, c3 = st.columns([4, 2, 2])
                c1.text(f.name)
                c2.caption(f"{(f.size / (1024*1024)):.2f} MB")
                c3.button("Remove", key=f"rem_t_{f.name}", on_click=remove_file, args=("uploaded_targets", f))
    
    with col2:
        with st.expander("Uploaded Predictor Layers", expanded=True):
            if not st.session_state.uploaded_predictors:
                st.caption("No predictor files uploaded.")
            for f in st.session_state.uploaded_predictors:
                c1, c2, c3 = st.columns([4, 2, 2])
                c1.text(f.name)
                c2.caption(f"{(f.size / (1024*1024)):.2f} MB")
                c3.button("Remove", key=f"rem_p_{f.name}", on_click=remove_file, args=("uploaded_predictors", f))

# --- Page 2: Training ---
elif st.session_state.active_step == "Training":
    st.header("Step 2: Sample and Train Model")
    if not st.session_state.targets_loaded or not st.session_state.predictors_loaded:
        st.warning("⚠️ Please upload and process both target and predictor files in Step 1.")
    else:
        if st.button("📥 Sample Training Data", disabled=st.session_state.sample_success):
            # 2. Percentage Progress Bar
            progress_bar = st.progress(0, text="Starting sampling...")
            
            def progress_callback(fraction, text):
                progress_bar.progress(fraction, text=text)

            X, y = data_loader.sample_training_data(
                target_files=st.session_state.uploaded_targets,
                predictor_files=st.session_state.uploaded_predictors,
                ref_profile=st.session_state.ref_profile,
                progress_callback=progress_callback # Pass the callback
            )
            if X is not None and y is not None:
                st.session_state.X, st.session_state.y = X, y
                st.session_state.log.append(f"✅ Sampled {len(X)} training points.")
                st.session_state.sample_success = True
                progress_bar.empty() # Remove the progress bar on completion
                st.rerun()

        if st.session_state.sample_success:
            if st.button("⚡ Train Random Forest", disabled=st.session_state.train_success):
                with st.spinner("Training model..."):
                    model, metrics = training.train_rf(st.session_state.X, st.session_state.y)
                    st.session_state.model = model
                    st.session_state.log.append("✅ Model trained successfully.")
                    st.session_state.train_success = True
                    st.rerun()

# --- Page 3: Prediction ---
elif st.session_state.active_step == "Prediction":
    st.header("Step 3: Predict Land Cover")
    if not st.session_state.train_success:
        st.warning("⚠️ Please train a model in Step 2 first.")
    else:
        if st.button("🛰️ Run Prediction", disabled=st.session_state.prediction_success):
            progress_bar = st.progress(0, text="Starting prediction...")
            
            def progress_callback(fraction, text):
                progress_bar.progress(fraction, text=text)

            predicted_filepath = prediction.predict_map_windowed(
                model=st.session_state.model,
                predictor_files=st.session_state.uploaded_predictors,
                mask=st.session_state.mask,
                ref_profile=st.session_state.ref_profile,
                progress_callback=progress_callback # Pass the callback
            )
            st.session_state.predicted_filepath = predicted_filepath
            st.session_state.log.append("✅ Prediction complete.")
            st.session_state.prediction_success = True
            progress_bar.empty()
            st.rerun()

# --- Page 5: Visualization ---
elif st.session_state.active_step == "Visualization":
    st.header("Step 5: Visualization")
    if not st.session_state.prediction_success:
        st.warning("⚠️ No prediction available. Please generate one in Step 3.")
    else:
        st.info(f"Displaying result from: `{st.session_state.predicted_filepath}`")
        cmap_list = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        title = "Predicted Land Cover Map"

        if st.button("📊 Show Inline Map"):
            with st.spinner("Loading map..."):
                with rasterio.open(st.session_state.predicted_filepath) as src:
                    predicted_array = src.read(1)
                fig = visualization.plot_prediction(predicted_array, cmap_list, title)
                st.pyplot(fig)
