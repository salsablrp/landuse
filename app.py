import streamlit as st
import os
import pandas as pd

from landuse_tool import data_loader, change_analysis, training, prediction, visualization

# --- Page Configuration and Initialization ---
st.set_page_config(page_title="Hybrid AI LUC Modeler", page_icon="🌍", layout="wide")
st.title("🌍 Hybrid AI Land Use Change Modeler")

# --- Initialize Session State ---
if "active_step" not in st.session_state:
    st.session_state.clear() # Clear state on first run or refresh
    st.session_state.active_step = "Home"
    st.session_state.targets_loaded = False
    st.session_state.predictors_loaded = False
    st.session_state.analysis_complete = False
    st.session_state.training_complete = False
    st.session_state.simulation_complete = False
    st.session_state.uploaded_targets = []
    st.session_state.uploaded_predictors = []
    st.session_state.transition_models = {}
    st.session_state.suitability_paths = {}

# --- Sidebar ---
with st.sidebar:
    st.header("Workflow")
    steps = ["Home", "1. Data Input", "2. Analyze Change", "3. Train AI", "4. Simulate Future", "5. Visualization"]
    st.session_state.active_step = st.radio("Steps", steps, index=steps.index(st.session_state.active_step), label_visibility="collapsed")
    
    if st.button("🔄 Reset Workflow", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- Main Page Content ---

if st.session_state.active_step == "Home":
    st.header("👋 Welcome to the Next-Generation Land Use Modeler")
    st.markdown("""
    This tool implements a state-of-the-art, three-stage simulation to forecast land use change. It combines statistical trend analysis with a powerful, AI-native suitability engine to create plausible future scenarios.

    **The Workflow:**
    1.  **Data Input:** Upload historical land cover maps and predictor variables.
    2.  **Analyze Change:** The tool uses a **Markov Chain** to quantify *how much* land has changed historically.
    3.  **Train AI:** Specialized **Random Forest models** are trained to learn the drivers of each change and map *where* future changes are most suitable.
    4.  **Simulate Future:** A **Cellular Automata** algorithm allocates the projected change onto the most suitable locations.
    5.  **Visualize:** Interact with the results and download maps and reports.
    """)

elif st.session_state.active_step == "1. Data Input":
    st.header("Step 1: Upload Your Geospatial Data")
    st.info("Please upload your data. All rasters must have the same dimensions, projection, and pixel size.")
    
    st.subheader("1a. Upload Historical Land Cover Maps (≥2)")
    uploaded_targets = st.file_uploader("Upload GeoTIFF files", type=["tif", "tiff"], accept_multiple_files=True, key="targets_uploader")
    if uploaded_targets: st.session_state.uploaded_targets = uploaded_targets

    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader("Upload predictor GeoTIFF files", type=["tif", "tiff"], accept_multiple_files=True, key="predictors_uploader")
    if uploaded_predictors: st.session_state.uploaded_predictors = uploaded_predictors
    
    if st.button("Process & Validate Inputs"):
        if len(st.session_state.uploaded_targets) < 2:
            st.error("Please upload at least two land cover maps.")
        elif not st.session_state.uploaded_predictors:
            st.error("Please upload at least one predictor map.")
        else:
            with st.spinner("Validating data..."):
                ref_profile, mask = data_loader.load_targets(st.session_state.uploaded_targets)
                if ref_profile and mask is not None:
                    st.session_state.ref_profile, st.session_state.mask = ref_profile, mask
                    st.session_state.targets_loaded = True
                    is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, ref_profile)
                    if is_valid:
                        st.session_state.predictors_loaded = True
                        st.success("All data processed and validated successfully! Proceed to the next step.")
                    else:
                        st.session_state.targets_loaded = False # Reset if predictors fail
                else:
                    st.session_state.targets_loaded = False

elif st.session_state.active_step == "2. Analyze Change":
    st.header("Step 2: Quantify Historical Change (Markov Chain)")
    if not st.session_state.predictors_loaded:
        st.warning("Please complete Step 1 first.")
    else:
        st.markdown("This step analyzes your first and last land cover maps to calculate the rate of change. This determines *how much* land is projected to change in the future.")
        if st.button("Run Change Analysis", disabled=st.session_state.analysis_complete):
            with st.spinner("Calculating transition matrix..."):
                matrix, counts = change_analysis.calculate_transition_matrix(st.session_state.uploaded_targets[0], st.session_state.uploaded_targets[-1])
                if matrix is not None and counts is not None:
                    st.session_state.transition_matrix = matrix
                    st.session_state.transition_counts = counts
                    st.session_state.analysis_complete = True
                    st.success("Change analysis complete!")
        
        if st.session_state.analysis_complete:
            st.subheader("Transition Pixel Counts")
            st.dataframe(st.session_state.transition_counts.style.background_gradient(cmap='viridis'))
            st.subheader("Transition Probability Matrix")
            st.dataframe(st.session_state.transition_matrix.style.format("{:.2%}").background_gradient(cmap='viridis'))

elif st.session_state.active_step == "3. Train AI":
    st.header("Step 3: Train AI Suitability Models")
    if not st.session_state.analysis_complete:
        st.warning("Please complete Step 2 first.")
    else:
        st.markdown("The tool will now train a specialized AI model for each significant historical transition. Each model learns the unique spatial drivers of that change.")
        threshold = st.number_input("Minimum pixels for a transition to be considered 'significant'", min_value=1, value=100)
        
        if st.button("Train All Models", disabled=st.session_state.training_complete):
            counts = st.session_state.transition_counts
            significant_transitions = counts[counts > threshold].stack().index.tolist()
            
            with st.expander("Training Log", expanded=True):
                for from_cls, to_cls in significant_transitions:
                    if from_cls == to_cls: continue
                    st.write(f"--- Training model for transition: {from_cls} -> {to_cls} ---")
                    with st.spinner(f"Creating dataset..."):
                        X, y = training.create_transition_dataset(from_cls, to_cls, st.session_state.uploaded_targets[0], st.session_state.uploaded_targets[-1], st.session_state.uploaded_predictors)
                    if X is not None and y is not None:
                        with st.spinner(f"Training model..."):
                            model, acc = training.train_rf_model(X, y)
                        if model:
                            st.session_state.transition_models[(from_cls, to_cls)] = model
                            st.write(f"✅ Model trained. Accuracy: {acc:.2%}")
                    else:
                        st.write("Skipped: Not enough data for this transition.")
            st.session_state.training_complete = True
            st.success("All AI models trained!")

elif st.session_state.active_step == "4. Simulate Future":
    st.header("Step 4: Run the Future Simulation")
    if not st.session_state.training_complete:
        st.warning("Please complete Step 3 first.")
    else:
        st.markdown("This final step will generate the suitability maps and run the Cellular Automata simulation to create the future land cover map.")
        if st.button("Run Simulation", disabled=st.session_state.simulation_complete):
            with st.spinner("Generating Suitability Atlas..."):
                models = st.session_state.transition_models
                total_models = len(models)
                progress_bar = st.progress(0, text="Starting Suitability Atlas generation...")
                for i, ((from_cls, to_cls), model) in enumerate(models.items()):
                    progress_text = f"Generating suitability map for {from_cls}->{to_cls}..."
                    st.session_state.suitability_paths[(from_cls, to_cls)] = prediction.generate_suitability_map(from_cls, model, st.session_state.uploaded_predictors, st.session_state.uploaded_targets[-1], lambda p: progress_bar.progress((i + p) / total_models, text=progress_text))
                st.success("Suitability Atlas generated.")
            
            with st.spinner("Running Cellular Automata Simulation..."):
                progress_bar = st.progress(0, text="Starting simulation...")
                def cb(p, t=""): progress_bar.progress(p, text=t)
                filepath = prediction.run_simulation(st.session_state.uploaded_targets[-1], st.session_state.transition_counts, st.session_state.suitability_paths, progress_callback=cb)
                if filepath:
                    st.session_state.predicted_filepath = filepath
                    st.session_state.simulation_complete = True
                    st.success("Future simulation complete!")
        
        if st.session_state.simulation_complete:
            st.balloons()
            st.info("Proceed to the 'Visualization' step to see your results.")

elif st.session_state.active_step == "5. Visualization":
    st.header("Step 5: Visualize and Export Results")
    if not st.session_state.simulation_complete:
        st.warning("Please complete all previous steps to generate a result.")
    else:
        # NOTE: Keeping the old visualization logic as requested.
        # This part can be upgraded later.
        st.info(f"Displaying result from the latest prediction.")
        if st.button("📊 Show Predicted Map as Image"):
            with st.spinner("Loading map..."):
                with rasterio.open(st.session_state.predicted_filepath) as src:
                    predicted_array = src.read(1)
                fig = visualization.plot_prediction(predicted_array) # Assumes a simple plot function
                st.pyplot(fig)

        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.predicted_filepath:
                with open(st.session_state.predicted_filepath, "rb") as fp:
                    st.download_button(label="Download Predicted Map (GeoTIFF)", data=fp, file_name="predicted_land_cover.tif")
        with col2:
            csv = st.session_state.transition_counts.to_csv().encode('utf-8')
            st.download_button(label="Download Transition Counts (CSV)", data=csv, file_name="transition_counts.csv")

