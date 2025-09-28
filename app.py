import streamlit as st
import os
import time

# --- Import your custom modules ---
from landuse_tool import data_loader, change_analysis, training, prediction, visualization

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
        "analysis_complete": False,
        "train_success": False,
        "simulation_success": False,
        "uploaded_targets": [],
        "uploaded_predictors": [],
        "transition_matrix": None,
        "transition_counts": None,
        "trained_models": {},
        "suitability_paths": {},
        "predicted_filepath": None,
        "log": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

initialize_session_state()

# --- Helper Functions ---
def reset_workflow():
    """Clears the session state to reset the application."""
    keys_to_keep = [] # No keys to keep, reset everything
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]
    initialize_session_state()
    st.success("Workflow has been reset.")
    time.sleep(1)


def get_file_size_mb(file_list):
    """Calculates the total size of a list of uploaded files in MB."""
    total_size = sum(f.size for f in file_list if f is not None)
    return total_size / (1024 * 1024)

def remove_file(list_key, file_to_remove):
    """Removes a specific file from the session state and resets progress."""
    st.session_state[list_key] = [f for f in st.session_state[list_key] if f.name != file_to_remove.name]
    # Reset all subsequent steps to force reprocessing
    st.session_state.targets_loaded = False
    st.session_state.predictors_loaded = False
    st.session_state.analysis_complete = False
    st.session_state.train_success = False
    st.session_state.simulation_success = False
    st.session_state.log = []
    # No st.rerun() needed here, Streamlit handles it

# --- Sidebar ---
with st.sidebar:
    st.header("Workflow")

    steps = ["Home", "1. Upload Data", "2. Analyze Change", "3. Train AI Models", "4. Simulate Future", "5. Visualization"]

    # Use st.radio for a clean, single-selection navigation
    st.session_state.active_step = st.radio(
        "Steps",
        steps,
        index=steps.index(st.session_state.active_step),
        label_visibility="collapsed"
    )

    st.divider()

    # Session Storage Meter
    st.header("Session Storage")
    total_mb = get_file_size_mb(st.session_state.uploaded_targets) + get_file_size_mb(st.session_state.uploaded_predictors)
    STORAGE_LIMIT_MB = 1000
    st.progress(min(total_mb / STORAGE_LIMIT_MB, 1.0))
    st.caption(f"{total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")

    st.divider()
    st.button("🔄 Reset Workflow", on_click=reset_workflow, use_container_width=True)

# --- Main Page Content ---

# Display persistent success messages at the top
for msg in st.session_state.log:
    st.success(msg)

# --- Page: Home ---
if st.session_state.active_step == "Home":
    st.header("👋 Welcome to the Hybrid AI Land Use Change Modeler")
    st.markdown("""
    This tool provides a state-of-the-art workflow for analyzing, predicting, and simulating land use change, inspired by leading models like Dinamica EGO and MOLUSCE, but delivered in an accessible, AI-native web application.

    **Our Methodology:**
    1.  **Analyze Past Change:** We use a **Markov Chain** to quantify the historical rate of change, determining *how much* land is likely to transition.
    2.  **Map Future Suitability:** We train specialized **Random Forest AI models** for each significant transition to learn *where* change is most likely to occur.
    3.  **Simulate the Future:** A **Cellular Automata** algorithm allocates the quantified change to the most suitable locations, creating a plausible future map.

    👈 Use the sidebar to begin.
    """)

# --- Page: Upload Data ---
elif st.session_state.active_step == "1. Upload Data":
    st.header("Step 1: Upload Your Data")
    st.markdown("Upload your historical land cover maps and the predictor variables that influence change.")

    st.subheader("1a. Upload Historical Land Cover Maps (≥2)")
    uploaded_targets = st.file_uploader("The first file should be the oldest map, and the last file the most recent.", type=["tif", "tiff"], accept_multiple_files=True, key="targets_uploader")
    if uploaded_targets:
        st.session_state.uploaded_targets = uploaded_targets

    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader("These are the static variables like elevation, slope, distance to roads, etc.", type=["tif", "tiff"], accept_multiple_files=True, key="predictors_uploader")
    if uploaded_predictors:
        st.session_state.uploaded_predictors = uploaded_predictors

    # Validation logic runs after files are uploaded
    if st.session_state.uploaded_targets and st.session_state.uploaded_predictors and not (st.session_state.targets_loaded and st.session_state.predictors_loaded):
        with st.spinner("Validating and processing uploaded files..."):
            st.session_state.ref_profile, st.session_state.mask = data_loader.load_targets(st.session_state.uploaded_targets)
            if st.session_state.ref_profile:
                st.session_state.targets_loaded = True
                is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, st.session_state.ref_profile)
                if is_valid:
                    st.session_state.predictors_loaded = True
                    st.session_state.log.append("✅ All uploaded files have been validated successfully.")

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

# --- Page: Analyze Change ---
elif st.session_state.active_step == "2. Analyze Change":
    st.header("Step 2: Analyze Historical Change")
    st.markdown("This step calculates the rate of change between your oldest and most recent land cover maps to understand historical trends.")
    if not (st.session_state.targets_loaded and st.session_state.predictors_loaded):
        st.warning("⚠️ Please upload and validate all data in Step 1 first.")
    else:
        if st.button("📈 Analyze Transition Matrix", disabled=st.session_state.analysis_complete):
            with st.spinner("Calculating transition probabilities..."):
                matrix, counts = change_analysis.calculate_transition_matrix(st.session_state.uploaded_targets[0], st.session_state.uploaded_targets[-1])
                st.session_state.transition_matrix = matrix
                st.session_state.transition_counts = counts
                st.session_state.analysis_complete = True
                st.session_state.log.append("✅ Historical change analysis complete.")
        
        if st.session_state.analysis_complete:
            st.subheader("Transition Probability Matrix (%)")
            st.dataframe(st.session_state.transition_matrix.style.format("{:.2%}"))
            st.subheader("Transition Pixel Counts")
            st.dataframe(st.session_state.transition_counts)

# --- Page: Train AI Models ---
elif st.session_state.active_step == "3. Train AI Models":
    st.header("Step 3: Train AI for Suitability Mapping")
    st.markdown("Here, we will train specialized AI models for each significant historical land use transition. These models learn *why* and *where* change happens.")
    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please complete the change analysis in Step 2 first.")
    else:
        # User-defined threshold for what constitutes a "significant" transition
        min_pixels_for_transition = st.number_input("Minimum number of pixels to model a transition:", min_value=10, value=1000, help="Higher values speed up training but ignore rarer transitions.")
        
        if st.button("🤖 Train AI Models", disabled=st.session_state.train_success):
            with st.spinner("Training models... This may take a while."):
                models = training.train_transition_models(
                    lc_start=st.session_state.uploaded_targets[0],
                    lc_end=st.session_state.uploaded_targets[-1],
                    predictor_files=st.session_state.uploaded_predictors,
                    transition_counts=st.session_state.transition_counts,
                    min_pixel_threshold=min_pixels_for_transition
                )
                st.session_state.trained_models = models
                st.session_state.train_success = True
                st.session_state.log.append(f"✅ Trained {len(models)} AI models successfully.")

        if st.session_state.train_success:
            st.subheader("Trained Model Accuracies")
            accuracies = {f"{k[0]} -> {k[1]}": v['accuracy'] for k, v in st.session_state.trained_models.items()}
            st.json(accuracies)

# --- Page: Simulate Future ---
elif st.session_state.active_step == "4. Simulate Future":
    st.header("Step 4: Simulate Future Land Cover")
    st.markdown("This final step uses a Cellular Automata to allocate the projected change across the landscape based on the AI-generated suitability maps.")
    if not st.session_state.train_success:
        st.warning("⚠️ Please train the AI models in Step 3 first.")
    else:
        if st.button("🛰️ Run Simulation", disabled=st.session_state.simulation_success):
            progress_bar = st.progress(0, text="Generating suitability atlas...")
            def cb(p, t): progress_bar.progress(p, text=t)

            future_lc_path = prediction.run_simulation(
                lc_end_file=st.session_state.uploaded_targets[-1],
                predictor_files=st.session_state.uploaded_predictors,
                transition_counts=st.session_state.transition_counts,
                trained_models=st.session_state.trained_models,
                progress_callback=cb
            )
            st.session_state.predicted_filepath = future_lc_path
            st.session_state.simulation_success = True
            st.session_state.log.append("✅ Simulation complete.")
            progress_bar.empty()

# --- Page: Visualization ---
elif st.session_state.active_step == "5. Visualization":
    st.header("Step 5: Visualize Results")
    st.markdown("Explore the input data and the final simulated map.")

    # Check for available data to visualize
    if not st.session_state.targets_loaded:
        st.warning("⚠️ No data available to visualize. Please upload data in Step 1.")
    else:
        with st.spinner("Generating and loading map..."):
            m = visualization.create_interactive_map(
                target_files=st.session_state.uploaded_targets,
                predictor_files=st.session_state.uploaded_predictors,
                prediction_filepath=st.session_state.predicted_filepath
            )
            visualization.st_folium(m, width=None, height=700)

        if st.session_state.predicted_filepath:
            st.divider()
            st.subheader("Download Results")
            with open(st.session_state.predicted_filepath, "rb") as fp:
                st.download_button(
                    label="📥 Download Predicted Map (GeoTIFF)",
                    data=fp,
                    file_name=os.path.basename(st.session_state.predicted_filepath),
                    mime="image/tiff",
                )

