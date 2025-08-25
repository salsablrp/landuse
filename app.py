import streamlit as st
import os
import rasterio
import time
import pandas as pd

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
def initialize_session_state():
    defaults = {
        "active_step": "Home",
        "targets_loaded": False,
        "predictors_loaded": False,
        "sample_success": False,
        "train_success": False,
        "prediction_success": False,
        "scenario_applied": False,
        "uploaded_targets": [],
        "uploaded_predictors": [],
        "scenario_changes": [],
        "scenario_predictor_paths": [],
        "ref_profile": None,
        "mask": None,
        "X": None,
        "y": None,
        "model": None,
        "predicted_filepath": None,
        "log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()

# --- Helper Functions ---
def reset_workflow():
    keys_to_keep = ['active_step']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    initialize_session_state()
    st.success("Workflow has been reset.")
    time.sleep(1)
    st.rerun()

def get_file_size_mb(file_list):
    total_size = sum(f.size for f in file_list)
    return total_size / (1024 * 1024)

def remove_file(file_list_key, file_to_remove):
    st.session_state[file_list_key] = [f for f in st.session_state[file_list_key] if f.name != file_to_remove.name]
    # Reset subsequent steps
    st.session_state.targets_loaded = False
    st.session_state.predictors_loaded = False
    st.session_state.sample_success = False
    st.session_state.train_success = False
    st.session_state.prediction_success = False
    st.session_state.scenario_applied = False
    st.session_state.log = []
    st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.header("Navigation")
    
    steps = ["Home", "Upload Data", "Training", "Scenario", "Prediction", "Visualization"]
    
    st.session_state.active_step = st.radio(
        "Steps", 
        steps, 
        index=steps.index(st.session_state.active_step),
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.header("Session Storage")
    total_mb = get_file_size_mb(st.session_state.uploaded_targets) + get_file_size_mb(st.session_state.uploaded_predictors)
    STORAGE_LIMIT_MB = 1000 
    st.progress(min(total_mb / STORAGE_LIMIT_MB, 1.0))
    st.caption(f"{total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")

    st.divider()
    st.button("🔄 Reset Workflow", on_click=reset_workflow, use_container_width=True)

# --- Main Page Content ---

for msg in st.session_state.log:
    st.success(msg)

# --- Page: Home ---
if st.session_state.active_step == "Home":
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

# --- Page: Upload Data ---
elif st.session_state.active_step == "Upload Data":
    st.header("Step 1: Upload Data")
    st.markdown("""
    In this step, you will upload the required raster datasets:

    - **Land cover targets** from at least two different years.
    - **Predictor variables**, such as elevation, slope, distance to roads, etc.

    These datasets will be aligned and prepared for training in the next step.
    """)

    st.subheader("1a. Upload Land Cover Targets (≥2 years)")
    uploaded_targets = st.file_uploader("Upload land cover rasters", type=["tif", "tiff"], accept_multiple_files=True, key="targets_uploader")

    if uploaded_targets and not st.session_state.targets_loaded:
        st.session_state.uploaded_targets = uploaded_targets
        with st.spinner("Processing targets..."):
            ref_profile, mask = data_loader.load_targets(st.session_state.uploaded_targets)
            if ref_profile and mask is not None:
                st.session_state.ref_profile, st.session_state.mask = ref_profile, mask
                st.session_state.targets_loaded = True
                if "✅ Targets processed successfully." not in st.session_state.log:
                    st.session_state.log.append("✅ Targets processed successfully.")

    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader("Upload predictor rasters", type=["tif", "tiff"], accept_multiple_files=True, key="predictors_uploader")

    if uploaded_predictors and not st.session_state.predictors_loaded:
        st.session_state.uploaded_predictors = uploaded_predictors
        if st.session_state.targets_loaded:
            with st.spinner("Validating predictors..."):
                is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, st.session_state.ref_profile)
                if is_valid:
                    st.session_state.predictors_loaded = True
                    log_msg = f"✅ {len(st.session_state.uploaded_predictors)} predictors validated."
                    if log_msg not in st.session_state.log:
                         st.session_state.log.append(log_msg)
        else:
            st.warning("Please upload and process target files before predictors.")

    st.divider()
    # Layer Management UI
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Uploaded Target Layers", expanded=True):
            if not st.session_state.uploaded_targets: st.caption("No target files.")
            for f in st.session_state.uploaded_targets:
                c1, c2, c3 = st.columns([4, 2, 2]); c1.text(f.name); c2.caption(f"{(f.size/(1024*1024)):.2f} MB"); c3.button("Remove", key=f"rem_t_{f.name}", on_click=remove_file, args=("uploaded_targets", f))
    with col2:
        with st.expander("Uploaded Predictor Layers", expanded=True):
            if not st.session_state.uploaded_predictors: st.caption("No predictor files.")
            for f in st.session_state.uploaded_predictors:
                c1, c2, c3 = st.columns([4, 2, 2]); c1.text(f.name); c2.caption(f"{(f.size/(1024*1024)):.2f} MB"); c3.button("Remove", key=f"rem_p_{f.name}", on_click=remove_file, args=("uploaded_predictors", f))

# --- Page: Training ---
elif st.session_state.active_step == "Training":
    st.header("Step 2: Sample and Train Model")
    st.markdown("""
    Here you will:

    - Sample training data points from the latest land cover raster.
    - Train a **Random Forest** model using the sampled data and predictors.

    This model will then be used to predict future land cover changes.
    """)
    if not st.session_state.predictors_loaded:
        st.warning("⚠️ Please complete Step 1 first.")
    else:
        if st.button("📥 Sample Training Data", disabled=st.session_state.sample_success):
            progress_bar = st.progress(0, text="Starting sampling...")
            def cb(f, t): progress_bar.progress(f, text=t)
            X, y = data_loader.sample_training_data(st.session_state.uploaded_targets, st.session_state.uploaded_predictors, st.session_state.ref_profile, progress_callback=cb)
            if X is not None and y is not None:
                st.session_state.X, st.session_state.y = X, y
                st.session_state.log.append(f"✅ Sampled {len(X)} training points.")
                st.session_state.sample_success = True
                progress_bar.empty(); st.rerun()
        if st.session_state.sample_success:
            if st.button("⚡ Train Random Forest", disabled=st.session_state.train_success):
                with st.spinner("Training model..."):
                    model, metrics = training.train_rf(st.session_state.X, st.session_state.y)
                    st.session_state.model = model
                    st.session_state.metrics = metrics # Store metrics
                    st.session_state.log.append("✅ Model trained successfully.")
                    st.session_state.train_success = True
                    st.rerun()
        
        if st.session_state.train_success:
            st.subheader("📊 Model Performance")
            report_df = pd.DataFrame(st.session_state.metrics['report']).transpose()
            st.dataframe(report_df)
            
            cm_fig = visualization.plot_confusion_matrix(
                st.session_state.metrics['confusion_matrix'], 
                st.session_state.metrics['class_names']
            )
            st.pyplot(cm_fig)

# --- Page: Scenario & Prediction ---
elif st.session_state.active_step == "Scenario":
    st.header("Step 3: Define a Scenario (Optional)")
    st.markdown("""
    Simulate alternative futures by:

    - Modifying predictor layers (e.g., increase population, deforestation).
    - Applying arithmetic operations to specific rasters.

    The modified predictor stack can be used for scenario-based predictions in the next step.
    """)
    if not st.session_state.predictors_loaded:
        st.warning("⚠️ Please upload predictors in Step 1 first.")
    else:
        predictor_filenames = [f.name for f in st.session_state.uploaded_predictors]
        st.markdown("### ➕ Add a Change")
        col1, col2, col3 = st.columns([2, 2, 1])
        selected_layer = col1.selectbox("Layer", options=predictor_filenames)
        selected_op = col2.selectbox("Operator", options=["multiply", "add", "subtract", "divide"])
        value = col3.number_input("Value", value=1.0, format="%.2f")
        if st.button("➕ Add Change"):
            st.session_state.scenario_changes.append({"layer": selected_layer, "op": selected_op, "value": value})
        if st.session_state.scenario_changes:
            st.markdown("### 📝 Current Scenario Changes")
            for i, change in enumerate(st.session_state.scenario_changes):
                st.write(f"{i+1}. `{change['layer']}` **{change['op']}** {change['value']}")
            if st.button("🧹 Clear All Changes"):
                st.session_state.scenario_changes = []
                st.rerun()
        if st.button("🌱 Apply Scenario", disabled=not st.session_state.scenario_changes):
            progress_bar = st.progress(0, text="Applying scenario...")
            def cb(f, t): progress_bar.progress(f, text=t)
            scenario_paths = scenarios.apply_scenario_windowed(st.session_state.uploaded_predictors, {"changes": st.session_state.scenario_changes}, progress_callback=cb)
            st.session_state.scenario_predictor_paths = scenario_paths
            st.session_state.log.append("✅ Scenario applied successfully.")
            st.session_state.scenario_applied = True
            progress_bar.empty(); st.rerun()
elif st.session_state.active_step == "Prediction":
    st.header("Step 4: Run Prediction")
    st.markdown("""
    Use the trained model to:

    - Predict land cover for a future or unseen time period.
    - Visualize and export the results in raster formats (GeoTIFF, PNG).

    Optionally, run predictions using modified scenarios from the previous step.
    """)
    if not st.session_state.train_success:
        st.warning("⚠️ Please train a model in Step 2 first.")
    else:
        prediction_mode = st.radio("Select Prediction Mode", ["Baseline", "Scenario"], horizontal=True)
        if st.button("🛰️ Run Prediction"):
            predictor_source = st.session_state.uploaded_predictors
            if prediction_mode == "Scenario":
                if not st.session_state.scenario_applied:
                    st.error("You must apply a scenario in Step 3 first."); st.stop()
                predictor_source = st.session_state.scenario_predictor_paths
            progress_bar = st.progress(0, text="Starting prediction...")
            def cb(f, t): progress_bar.progress(f, text=t)
            predicted_filepath = prediction.predict_map_windowed(st.session_state.model, predictor_source, st.session_state.mask, st.session_state.ref_profile, progress_callback=cb)
            st.session_state.predicted_filepath = predicted_filepath
            st.session_state.log.append(f"✅ {prediction_mode} prediction complete.")
            st.session_state.prediction_success = True
            progress_bar.empty(); st.rerun()

# --- Page: Visualization ---
elif st.session_state.active_step == "Visualization":
    st.header("Step 5: Visualization")
    st.markdown("""
    Here you can:

    - View predictions as inline maps or images.
    - Export the predicted land cover as PNG or GeoTIFF.
    - Compare baseline predictions with scenario outcomes (if available).
    """)
    if not st.session_state.uploaded_targets and not st.session_state.uploaded_predictors:
        st.warning("⚠️ No data available to visualize. Please upload files in Step 1.")
    else:
        # The map will be generated with whichever layers are available.
        st.info("Use the layer control icon in the top right of the map to toggle layers on and off.")
        
        m = visualization.create_interactive_map(
            target_files=st.session_state.uploaded_targets,
            predictor_files=st.session_state.uploaded_predictors,
            prediction_filepath=st.session_state.predicted_filepath # This will be None if prediction hasn't run
        )
        m.to_streamlit(height=700)
