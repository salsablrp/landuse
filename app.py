import streamlit as st
import os
import pandas as pd
import joblib
import tempfile
from streamlit_folium import st_folium

from landuse_tool import data_loader, change_analysis, training, prediction, visualization, scenarios

# --- Page Configuration and Initialization ---
st.set_page_config(page_title="Hybrid AI LUC Modeler", page_icon="🌍", layout="wide")
st.title("🌍 Hybrid AI Land Use Change Modeler")

# --- Initialize Session State ---
def init_state():
    defaults = {
        "active_step": "Home", "targets_loaded": False, "predictors_loaded": False,
        "analysis_complete": False, "training_complete": False, "simulation_complete": False,
        "uploaded_targets_with_years": [], "uploaded_predictors": [],
        "model_paths": {}, "predicted_filepath": None,
        "transition_matrix": None, "transition_counts": None, "class_legends": pd.DataFrame(),
        "use_spatially_aware_ai": False, "neighborhood_radius": 5
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if 'temp_dir' not in st.session_state: st.session_state.temp_dir = tempfile.mkdtemp()

init_state()

# --- Sidebar ---
with st.sidebar:
    st.header("Workflow")
    steps = ["Home", "1. Data Input", "2. Analyze Change", "3. Train AI", "4. Simulate Future", "5. Visualization"]
    st.session_state.active_step = st.radio("Steps", steps, index=steps.index(st.session_state.active_step), label_visibility="collapsed")
    
    if st.button("🔄 Reset Workflow", use_container_width=True):
        st.session_state.clear(); st.rerun()

# --- Main Page Content ---

if st.session_state.active_step == "Home":
    st.header("👋 Welcome to the Next-Generation Land Use Modeler")
    st.markdown("This tool implements a state-of-the-art, three-stage simulation to forecast land use change. It combines statistical trend analysis with a powerful, AI-native suitability engine to create plausible future scenarios.")

elif st.session_state.active_step == "1. Data Input":
    st.header("Step 1: Upload Your Geospatial Data")
    st.info("Please upload your data. All rasters must have the same dimensions, projection, and pixel size.")
    
    st.subheader("1a. Upload Historical Land Cover Maps (≥2)")
    uploaded_targets = st.file_uploader("Upload GeoTIFF files for land cover", type=["tif", "tiff"], accept_multiple_files=True, key="targets_uploader")
    
    if uploaded_targets:
        temp_targets = []
        for i, f in enumerate(uploaded_targets):
            default_year = 2000 + i
            year = st.number_input(f"Enter year for `{f.name}`:", min_value=1900, max_value=2100, value=default_year, key=f"year_{f.name}")
            temp_targets.append({'file': f, 'year': year})
        st.session_state.uploaded_targets_with_years = temp_targets

    st.subheader("1b. Upload Predictor Rasters")
    uploaded_predictors = st.file_uploader("Upload predictor GeoTIFF files", type=["tif", "tiff"], accept_multiple_files=True, key="predictors_uploader")
    if uploaded_predictors: st.session_state.uploaded_predictors = uploaded_predictors
    
    if st.button("Process & Validate Inputs"):
        targets = st.session_state.uploaded_targets_with_years
        if len(targets) < 2: st.error("Please upload at least two land cover maps.")
        elif not st.session_state.uploaded_predictors: st.error("Please upload at least one predictor map.")
        else:
            with st.spinner("Validating data..."):
                target_files = [t['file'] for t in targets]
                ref_profile, mask = data_loader.load_targets(target_files)
                if ref_profile and mask is not None:
                    st.session_state.ref_profile, st.session_state.mask = ref_profile, mask
                    st.session_state.targets_loaded = True
                    is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, ref_profile)
                    if is_valid: st.session_state.predictors_loaded = True; st.success("All data processed successfully!")
                    else: st.session_state.targets_loaded = False
                else: st.session_state.targets_loaded = False

elif st.session_state.active_step == "2. Analyze Change":
    st.header("Step 2: Quantify Historical Change")
    if not st.session_state.predictors_loaded: st.warning("Please complete Step 1 first.")
    else:
        num_targets = len(st.session_state.uploaded_targets_with_years)
        analysis_mode = "Non-Linear (Most Recent Trend)" if num_targets > 2 else "Linear (Overall Trend)"
        st.subheader(f"Analysis Mode: {analysis_mode}")

        if st.button("Run Change Analysis", disabled=st.session_state.analysis_complete):
            with st.spinner("Calculating transition matrix..."):
                targets_with_years = st.session_state.uploaded_targets_with_years
                if analysis_mode.startswith("Non-Linear"):
                    matrix, counts = change_analysis.analyze_non_linear_trends(targets_with_years)
                else:
                    matrix, counts = change_analysis.calculate_transition_matrix(targets_with_years[0]['file'], targets_with_years[-1]['file'])
                if matrix is not None:
                    st.session_state.transition_matrix, st.session_state.transition_counts = matrix, counts
                    st.session_state.analysis_complete = True
                    class_ids = counts.index.tolist()
                    st.session_state.class_legends = pd.DataFrame({'Class Name': [f'Class {i}' for i in class_ids]}, index=class_ids)
                    st.success("Change analysis complete!")
        
        if st.session_state.analysis_complete:
            st.subheader("Define Land Cover Class Names")
            edited_legends = st.data_editor(st.session_state.class_legends, use_container_width=True)
            st.session_state.class_legends = edited_legends
            st.subheader("Transition Pixel Counts")
            st.dataframe(st.session_state.transition_counts.style.background_gradient(cmap='viridis'))


elif st.session_state.active_step == "3. Train AI":
    st.header("Step 3: Train AI Suitability Models")
    if not st.session_state.analysis_complete: st.warning("Please complete Step 2 first.")
    else:
        st.markdown("Train AI models for each significant transition. Enable advanced options for more accurate spatial patterns.")
        
        with st.expander("Advanced AI Options"):
            st.session_state.use_spatially_aware_ai = st.checkbox("Enable Spatially-Aware AI (Neighborhood Effects)", value=st.session_state.use_spatially_aware_ai)
            if st.session_state.use_spatially_aware_ai:
                st.session_state.neighborhood_radius = st.number_input("Neighborhood Radius (in pixels)", min_value=1, value=st.session_state.neighborhood_radius, help="A larger radius considers more spatial context but is slower.")

        threshold = st.number_input("Minimum pixels for a transition to be 'significant'", min_value=1, value=100)
        
        if st.button("Train All Models", disabled=st.session_state.training_complete):
            predictor_files_for_training = st.session_state.uploaded_predictors
            if st.session_state.use_spatially_aware_ai:
                with st.spinner("Generating neighborhood predictors... This may take time."):
                    neighborhood_predictors = scenarios.create_neighborhood_predictors(st.session_state.uploaded_predictors, st.session_state.uploaded_targets_with_years[-1]['file'], st.session_state.neighborhood_radius, st.session_state.temp_dir)
                    predictor_files_for_training += neighborhood_predictors
            
            counts = st.session_state.transition_counts
            transitions = counts[counts > threshold].stack().index.tolist()
            
            status = st.status("Starting model training...", expanded=True)
            for from_cls, to_cls in transitions:
                if from_cls == to_cls: continue
                status.update(label=f"Training model for transition: {from_cls} -> {to_cls}")
                X, y = training.create_transition_dataset(from_cls, to_cls, st.session_state.uploaded_targets_with_years[0]['file'], st.session_state.uploaded_targets_with_years[-1]['file'], predictor_files_for_training)
                if X is not None:
                    model, acc = training.train_rf_model(X, y)
                    if model:
                        model_path = os.path.join(st.session_state.temp_dir, f"model_{from_cls}_{to_cls}.joblib")
                        joblib.dump(model, model_path)
                        st.session_state.model_paths[(from_cls, to_cls)] = model_path
                        status.write(f"✅ Model for {from_cls} -> {to_cls} trained. Accuracy: {acc:.2%}")
            status.update(label="All AI models trained!", state="complete")
            st.session_state.training_complete = True

elif st.session_state.active_step == "4. Simulate Future":
    st.header("Step 4: Simulate Future Land Cover")
    if not st.session_state.training_complete: st.warning("⚠️ Please train AI models in Step 3 first.")
    else:
        st.subheader("Scenario & Simulation Options")
        
        # Policy Demand Override
        with st.expander("Override Historical Demand (Policy Simulation)"):
            st.info("Here you can manually edit the number of pixels projected to change for each transition, simulating policy targets.")
            edited_counts = st.data_editor(st.session_state.transition_counts, use_container_width=True)
        
        # Stochastic Allocation
        use_stochastic = st.checkbox("Enable Stochastic Allocation", help="Introduces randomness for sensitivity analysis. Each run will be slightly different.")

        if st.button("🛰️ Run Simulation", disabled=st.session_state.simulation_complete):
            status = st.status("Starting simulation...", expanded=True)
            def cb(p, t): status.update(label=t)
            
            targets = st.session_state.uploaded_targets_with_years
            
            # Use the edited counts from the policy editor for the simulation
            final_transition_counts = edited_counts
            
            future_lc_path = prediction.run_simulation(
                lc_end_file=targets[-1]['file'],
                predictor_files=st.session_state.uploaded_predictors, # Use original predictors for simulation
                transition_counts=final_transition_counts,
                trained_model_paths=st.session_state.model_paths,
                temp_dir=st.session_state.temp_dir,
                stochastic=use_stochastic,
                progress_callback=cb
            )
            st.session_state.predicted_filepath = future_lc_path
            st.session_state.simulation_complete = True
            status.update(label="Simulation Complete!", state="complete")

elif st.session_state.active_step == "5. Visualization":
    st.header("Step 5: Visualize and Export Results")
    if not st.session_state.simulation_complete: st.warning("Please generate a simulation result in Step 4.")
    else:
        st.info("Displaying interactive map of the simulated future land cover.")
        with st.spinner("Generating and loading map..."):
            class_legends_dict = st.session_state.class_legends['Class Name'].to_dict()
            m = visualization.create_interactive_map(st.session_state.predicted_filepath, class_legends_dict)
            st_folium(m, width=None, height=700)

        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.predicted_filepath:
                with open(st.session_state.predicted_filepath, "rb") as fp:
                    st.download_button(label="Download Predicted Map (GeoTIFF)", data=fp, file_name="predicted_land_cover.tif")
        with col2:
            if st.session_state.transition_counts is not None:
                csv = st.session_state.transition_counts.to_csv().encode('utf-8')
                st.download_button(label="Download Transition Counts (CSV)", data=csv, file_name="transition_counts.csv")

