import streamlit as st
import os
import pandas as pd
import joblib
import tempfile
from streamlit_folium import st_folium

from landuse_tool import data_loader, change_analysis, training, prediction, visualization, scenarios

# --- Page Configuration and Initialization ---
st.set_page_config(page_title="Land Use Prediction Tool", page_icon="🌍", layout="wide")
st.title("🌍 Land Use Monitoring and Prediction Tool")

# --- Helper Functions ---
def get_file_size_mb(file_list):
    """Calculates the total size of a list of uploaded files in MB."""
    if not file_list:
        return 0.0
    # The file list for targets is a list of dicts, so we extract the file object
    if isinstance(file_list[0], dict) and 'file' in file_list[0]:
        files = [item['file'] for item in file_list if item.get('file')]
    else:
        files = file_list
    
    total_size_bytes = sum(f.size for f in files if f is not None)
    return total_size_bytes / (1024 * 1024)

def remove_file(list_key, file_to_remove):
    """
    Removes a specific file from a list in the session state and resets
    the workflow progress, forcing re-validation.
    """
    # The target list is a list of dicts; predictors is a list of file objects.
    if list_key == "uploaded_targets_with_years":
        # Find the specific dictionary to remove based on the file object's name
        st.session_state[list_key] = [
            item for item in st.session_state[list_key] 
            if item['file'].name != file_to_remove['file'].name
        ]
    else: # For predictors
        st.session_state[list_key] = [
            f for f in st.session_state[list_key] 
            if f.name != file_to_remove.name
        ]
    
    # Invalidate all subsequent steps because the data has changed
    st.session_state.predictors_loaded = False
    st.session_state.analysis_complete = False
    st.session_state.training_complete = False
    st.session_state.simulation_complete = False
    
    # Rerun the app to immediately reflect the changes in the UI
    st.rerun()

# --- Initialize Session State ---
def init_state():
    defaults = {
        "active_step": "Home", "targets_loaded": False, "predictors_loaded": False,
        "analysis_complete": False, "training_complete": False, "simulation_complete": False,
        "uploaded_targets_with_years": [], "uploaded_predictors": [], "generated_predictors": [],
        "model_paths": {}, "predicted_filepath": None,
        "transition_matrix": None, "transition_counts": None, "class_legends": pd.DataFrame(), "scenario_predictors_modified": [], "scenario_predictors_uploaded": [], "use_neighborhood_choice": False,
        "radius_choice": 5,
        "use_growth_modes_choice": False,
        "threshold_choice": 100,
        "use_policy_demand_choice": False,
        "use_stochastic_choice": False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
    if 'temp_dir' not in st.session_state: st.session_state.temp_dir = tempfile.mkdtemp()

init_state()

# --- Sidebar ---
with st.sidebar:
    st.header("Workflow")
    steps = ["Home", "Data Input", "Analyze Change", "Training Model", "Simulate Future", "Visualization"]
    st.session_state.active_step = st.radio("Steps", steps, index=steps.index(st.session_state.active_step), label_visibility="collapsed")

    st.divider()

    st.header("Session Storage")
    total_mb = get_file_size_mb(st.session_state.get("uploaded_targets_with_years", [])) + get_file_size_mb(st.session_state.get("uploaded_predictors", []))
    STORAGE_LIMIT_MB = 1000 
    st.progress(min(total_mb / STORAGE_LIMIT_MB, 1.0))
    st.caption(f"{total_mb:.2f} MB / {STORAGE_LIMIT_MB} MB")

    st.divider()
    
    if st.button("🔄 Reset Workflow", use_container_width=True):
        st.session_state.clear(); st.rerun()

# --- Main Page Content ---

if st.session_state.active_step == "Home":
    st.header("👋 Welcome to the Land Use Monitoring and Prediction Tool")
    st.markdown("""
                This tool helps helps you analyze and forecast land cover change using remote sensing data and machine learning.
                
                This tool implements a state-of-the-art, three-stage simulation to forecast land use change. It combines statistical trend analysis with a powerful, AI-native suitability engine to create plausible future scenarios.

                **Workflow Overview:**
                1. **Data Input:** Upload historical land cover maps and predictor variables (e.g., elevation, distance to roads).
                2. **Analyze Change:** Quantify historical land cover transitions and trends.
                3. **Train AI Model:** Train machine learning models to learn the suitability of each land cover transition.
                4. **Simulate Future:** Use a Cellular Automata to allocate projected changes across the landscape.
                5. **Visualization:** Explore your results with an interactive map and export it into TIFF, CSV or JPG format.

                👈 Use the sidebar to begin with **Step 1**.
                """)
    st.info("To explore advanced features like non-linear trends, policy scenarios, and dynamic neighborhood predictors, please see the options within each step.")

elif st.session_state.active_step == "Data Input":
    st.header("Step 1: Upload Your Geospatial Data")
    st.markdown("""
    In this step, you will upload the required raster datasets:

    - **Land cover targets** from at least two different years.
    - **Predictor variables**, such as elevation, slope, distance to roads, etc.

    These datasets will be aligned and prepared for training in the next step.
    """)
    st.info("All rasters must have the same dimensions, projection, and pixel size. Check the alignment after uploading.")
    
    # --- File Uploaders are always visible ---
    st.subheader("1a. Upload or Add Historical Land Cover Maps (≥2 years)")
    uploaded_targets = st.file_uploader("Upload GeoTIFF files for land cover", type=["tif", "tiff"], accept_multiple_files=True, key="targets_uploader")

    st.subheader("1b. Upload or Add Predictor Rasters")
    uploaded_predictors = st.file_uploader("Upload predictor GeoTIFF files", type=["tif", "tiff"], accept_multiple_files=True, key="predictors_uploader")
    
    # --- Logic to add new files to the session state without losing existing ones ---
    if uploaded_targets:
        # Create a set of existing filenames for quick lookup
        existing_target_names = {item['file'].name for item in st.session_state.uploaded_targets_with_years}
        for f in uploaded_targets:
            if f.name not in existing_target_names:
                # Add new file to the list
                st.session_state.uploaded_targets_with_years.append({'file': f, 'year': 2000}) # Default year
                st.session_state.predictors_loaded = False # New files require re-validation

    if uploaded_predictors:
        existing_predictor_names = {f.name for f in st.session_state.uploaded_predictors}
        for f in uploaded_predictors:
            if f.name not in existing_predictor_names:
                st.session_state.uploaded_predictors.append(f)
                st.session_state.predictors_loaded = False # New files require re-validation

    st.divider()

    # --- Display Uploaded Files with Remove Buttons ---
    st.subheader("Uploaded Files for Analysis")
    if not st.session_state.uploaded_targets_with_years and not st.session_state.uploaded_predictors:
        st.caption("No files have been uploaded yet.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Target Layers", expanded=True):
            for i, item in enumerate(st.session_state.uploaded_targets_with_years):
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.text(item['file'].name)
                # Allow editing the year
                new_year = c2.number_input("Year", value=item['year'], key=f"year_edit_{item['file'].name}", label_visibility="collapsed")
                st.session_state.uploaded_targets_with_years[i]['year'] = new_year
                c3.button("Remove", key=f"rem_t_{item['file'].name}", on_click=remove_file, args=("uploaded_targets_with_years", item))

    with col2:
        with st.expander("Predictor Layers", expanded=True):
            for f in st.session_state.uploaded_predictors:
                c1, c2 = st.columns([5, 1])
                c1.text(f.name)
                c2.button("Remove", key=f"rem_p_{f.name}", on_click=remove_file, args=("uploaded_predictors", f))

    st.divider()

    # --- Validation Logic ---
    if st.session_state.predictors_loaded:
        st.success("✅ Current set of files has been validated.")
    else:
        # Show the validation button only if there are files to validate
        if st.session_state.uploaded_targets_with_years and st.session_state.uploaded_predictors:
            if st.button("Process & Validate Inputs"):
                targets = st.session_state.uploaded_targets_with_years
                if len(targets) < 2: st.error("⚠️ Please upload at least two land cover maps.")
                else:
                    with st.spinner("Validating data..."):
                        target_files = [t['file'] for t in targets]
                        ref_profile, mask = data_loader.load_targets(target_files)
                        if ref_profile and mask is not None:
                            st.session_state.ref_profile, st.session_state.mask = ref_profile, mask
                            st.session_state.targets_loaded = True
                            is_valid = data_loader.load_predictors(st.session_state.uploaded_predictors, ref_profile)
                            if is_valid: 
                                st.session_state.predictors_loaded = True
                                st.rerun() 
                            else: st.session_state.targets_loaded = False
                        else: st.session_state.targets_loaded = False

elif st.session_state.active_step == "Analyze Change":
    st.header("Step 2: Quantify Historical Change")
    st.markdown("""
    In this step, the model will:

    - Analyze entire historical sequence to understand the rate of change.
    - Incorporate Markov Chain Analysis to create Transition Probability Matrix.
    - Calculate historical probability of each Land Use class transitioning to every other class.
    - Project total amount of land that is expected to change from one class to another by a future target year.
    - This model will then be used to predict future land cover changes.
    """)
    if not st.session_state.predictors_loaded: st.warning("⚠️ Please complete Step 1 first.")
    else:
        # Check if the step is ALREADY complete
        if st.session_state.analysis_complete:
            # --- "COMPLETED" VIEW ---
            st.success("✅ Change analysis is complete.")
            st.subheader(f"Analysis Mode Used: {st.session_state.analysis_mode}")
            
            st.subheader("Define Land Cover Class Names")
            st.info("Edit the names in the 'Class Name' column below for use in legends.")
            edited_legends = st.data_editor(st.session_state.class_legends, use_container_width=True)
            st.session_state.class_legends = edited_legends
            
            st.subheader("Projected Transition Pixel Counts")
            st.dataframe(st.session_state.transition_counts.style.background_gradient(cmap='viridis'))

            # --- NEW: Display the saved plots ---
            if st.session_state.analysis_mode.startswith("Non-Linear") and "trend_plots" in st.session_state:
                with st.expander("View Non-Linear Trend Analysis Plots"):
                    if not st.session_state.trend_plots:
                        st.caption("No transitions were significant enough to plot.")
                    else:
                        for fig in st.session_state.trend_plots:
                            st.pyplot(fig)

        else:            
            num_targets = len(st.session_state.uploaded_targets_with_years)
            analysis_mode = "Non-Linear Trend" if num_targets > 2 else "Linear Trend"
            st.subheader(f"Analysis Mode: {analysis_mode}")

            if st.button("Run Change Analysis"):
                with st.spinner("Calculating transition matrix and trends..."):
                    targets_with_years = st.session_state.uploaded_targets_with_years
                    
                    if analysis_mode.startswith("Non-Linear"):
                        # Expect three return values now
                        matrix, counts, plots = change_analysis.analyze_non_linear_trends(targets_with_years)
                        st.session_state.trend_plots = plots # Save the plots
                    else:
                        # Also expect three return values
                        matrix, counts, _ = change_analysis.calculate_transition_matrix(targets_with_years[0]['file'], targets_with_years[-1]['file'])
                        st.session_state.trend_plots = [] # Save an empty list

                    if matrix is not None and counts is not None:
                        st.session_state.transition_matrix = matrix
                        st.session_state.transition_counts = counts
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_mode = analysis_mode
                        
                        class_ids = counts.index.tolist()
                        st.session_state.class_legends = pd.DataFrame({'Class Name': [f'Class {i}' for i in class_ids]}, index=class_ids)
                        
                        st.rerun()

elif st.session_state.active_step == "Training Model":
    st.header("Step 3: Train AI Suitability Models")
    st.markdown("""
    The third stage is to map future suitability:
    
    - The model will Train multiple, specialized model for each plausible transition (users will define the minimum number of pixels to be considered as plausible change).
    - The model will apply Targeted and balanced sampling of each changed and not changed pixels.
    - Create a training dataset for each transition, and learns the predictor conditions which result in a particular change.
    - Create a collection of probability maps for every possible change.
    """)
    if not st.session_state.analysis_complete: 
        st.warning("⚠️ Please complete Step 2 first.")
    else:
        # --- UI CONTROLS ARE ALWAYS VISIBLE ---
        st.subheader("Training Configuration")
        with st.expander("Advanced Options: Spatially-Aware AI"):
            # Use .get() to remember the user's last choice, and assign a key
            use_neighborhood = st.checkbox(
                "Enable Neighborhood Predictors (More Accurate, Slower)", 
                value=st.session_state.get('use_neighborhood_choice', False), 
                key='use_neighborhood_choice'
            )
            if use_neighborhood:
                radius = st.slider(
                    "Neighborhood radius (in pixels)", 1, 10, 
                    st.session_state.get('radius_choice', 5), 
                    key='radius_choice'
                )
            use_growth_modes = st.checkbox("Enable Advanced Growth Mode Simulation (Expander/Patcher)", value=st.session_state.get('use_growth_modes_choice', False), key='use_growth_modes_choice')

        threshold = st.number_input(
            "Minimum pixels for a transition to be 'significant'", 
            min_value=1, 
            value=st.session_state.get('threshold_choice', 100),
            key='threshold_choice'
        )
        
        # --- Calculate total models to train beforehand ---
        counts = st.session_state.transition_counts
        significant_transitions = [t for t in counts[counts > threshold].stack().index.tolist() if t[0] != t[1]]
        
        total_models_to_train = len(significant_transitions)
        if use_growth_modes:
            total_models_to_train *= 2
            
        st.info(f"Based on your settings, **{total_models_to_train}** AI models will be trained.")
        
        # --- BUTTON TO RUN OR RE-RUN THE TRAINING ---
        if st.button("Train All Models"):
            # When the button is clicked, invalidate any subsequent steps
            st.session_state.simulation_complete = False 
            
            all_predictors = st.session_state.uploaded_predictors
            if use_neighborhood:
                with st.spinner("Generating neighborhood predictors..."):
                    targets = st.session_state.uploaded_targets_with_years
                    neighborhood_predictors = scenarios.generate_neighborhood_predictors(targets[-1]['file'], st.session_state.temp_dir, radius_pixels=radius)
                    st.session_state.generated_predictors = neighborhood_predictors
                    all_predictors = st.session_state.uploaded_predictors + neighborhood_predictors
                    st.success(f"Generated {len(neighborhood_predictors)} neighborhood predictors.")

            counts = st.session_state.transition_counts
            transitions = counts[counts > threshold].stack().index.tolist()
            
            # Reset previous results before starting a new run
            st.session_state.model_paths = {}
            st.session_state.model_accuracies = {}

            status = st.status("Starting model training...", expanded=True)

            # --- Add a counter for progress reporting ---
            for i, (from_cls, to_cls) in enumerate(transitions):
                progress_text = f"Training model for {from_cls} -> {to_cls} ({i+1}/{total_models_to_train})"
                status.update(label=progress_text)
                
                X, y = training.create_transition_dataset(from_cls, to_cls, st.session_state.uploaded_targets_with_years[0]['file'], st.session_state.uploaded_targets_with_years[-1]['file'], all_predictors)
                if X is not None:
                    model, acc = training.train_rf_model(X, y)
                    if model:
                        model_path = os.path.join(st.session_state.temp_dir, f"model_{from_cls}_{to_cls}.joblib")
                        joblib.dump(model, model_path)
                        st.session_state.model_paths[(from_cls, to_cls)] = model_path
                        st.session_state.model_accuracies[f"{from_cls} -> {to_cls}"] = acc
                        status.write(f"✅ Model trained. Accuracy: {acc:.2%}")
            
            status.update(label="All AI models trained!", state="complete")
            st.session_state.training_complete = True
            st.rerun()

            for from_cls, to_cls in transitions:
                if from_cls == to_cls: continue
                status.update(label=f"Training model for transition: {from_cls} -> {to_cls}")
                X, y = training.create_transition_dataset(from_cls, to_cls, st.session_state.uploaded_targets_with_years[0]['file'], st.session_state.uploaded_targets_with_years[-1]['file'], all_predictors)
                if X is not None:
                    model, acc = training.train_rf_model(X, y)
                    if model:
                        model_path = os.path.join(st.session_state.temp_dir, f"model_{from_cls}_{to_cls}.joblib")
                        joblib.dump(model, model_path)
                        st.session_state.model_paths[(from_cls, to_cls)] = model_path
                        st.session_state.model_accuracies[f"{from_cls} -> {to_cls}"] = acc # Save accuracy for display
                        status.write(f"✅ Model for {from_cls} -> {to_cls} trained. Accuracy: {acc:.2%}")
            status.update(label="All AI models trained!", state="complete")
            st.session_state.training_complete = True
            st.rerun()

        # --- DISPLAY RESULTS IF TRAINING IS COMPLETE ---
        if st.session_state.training_complete:
            st.subheader("Results from Last Training Run")
            st.write("You can review the results below or change settings and re-run the training.")
            
            with st.expander("View Trained Model Accuracies", expanded=True):
                if st.session_state.model_accuracies:
                    acc_df = pd.DataFrame.from_dict(
                        st.session_state.model_accuracies, 
                        orient='index', 
                        columns=['Accuracy']
                    )
                    acc_df.index.name = "Transition"
                    st.dataframe(acc_df.style.format({'Accuracy': '{:.2%}'}))
                else:
                    st.caption("No model accuracies recorded from the last run.")

elif st.session_state.active_step == "Simulate Future":
    st.header("Step 4: Simulate Future Land Cover")
    st.markdown("""
    In the last stage, users will be given options to choose several approach to apply for future land use simulation.

    1. Cellular automata helps to allocate the quantity of change to the most suitable locations on the map.
    2. Stochastic allocation will select the pixels not only based on high suitability confidence, but also with stochastic elements, running the simulation multiple times to create equally plausible, future map.
    3. Growth modes will train two AI models for each transition​
        
        - Expander model: trained only on pixels that changed adjacent to existing target LU class patches​
        - Patcher model: trained only on pixels that changed and created a brand new, isolated patches of the target LU class.
    
    The final simulation would then use all selected models, allowing it to realistically simulate both the expansion of existing areas and the spontaneous creation of new ones.
    """)
    if not st.session_state.training_complete: 
        st.warning("⚠️ Please train AI models in Step 3 first.")
    else:
        # --- UI CONTROLS ARE ALWAYS VISIBLE ---
        st.subheader("Simulation Configuration")
        with st.expander("Scenario Builder: Create a Future Scenario (Optional)", expanded=True):
            st.markdown("**1. Upload New Future Predictor Layers**")
            st.info("Upload any new layers that represent a future state, like a planned road network or a new zoning map. These will be added to the simulation.")
            scenario_files_new = st.file_uploader(
                "Upload GeoTIFFs", type=["tif", "tiff"], 
                accept_multiple_files=True, # Allows multiple file uploads
                key="scenario_uploader_new"
            )
            if scenario_files_new: st.session_state.scenario_predictors_uploaded = scenario_files_new
            
            st.markdown("**2. Modify an Existing Predictor Layer**")
            st.info("Select one of your original predictors and apply a simple calculation to create a new scenario layer.")
            
            # Create a list of original predictor names for the selectbox
            original_predictor_names = [f.name for f in st.session_state.uploaded_predictors]
            
            col1, col2, col3 = st.columns(3)
            layer_to_modify_name = col1.selectbox("Select Predictor to Modify", options=original_predictor_names)
            operator = col2.selectbox("Operation", options=["Multiply", "Add", "Subtract", "Divide"])
            value = col3.number_input("Value", value=1.0, format="%.2f")

            if st.button("Apply Modification"):
                # Find the actual file object from the selected name
                layer_file_object = next((f for f in st.session_state.uploaded_predictors if f.name == layer_to_modify_name), None)
                if layer_file_object:
                    with st.spinner(f"Calculating '{layer_to_modify_name} {operator} {value}'..."):
                        modified_path = scenarios.modify_predictor_raster(
                            input_raster_file=layer_file_object,
                            operator=operator,
                            value=value,
                            temp_dir=st.session_state.temp_dir
                        )
                        if modified_path:
                            # We store the path, as the file object is temporary
                            st.session_state.scenario_predictors_modified.append(modified_path)
                            st.success(f"Created new scenario layer: {os.path.basename(modified_path)}")
                else:
                    st.error("Could not find the selected predictor file.")

            # Display the list of modified scenario predictors
            if st.session_state.scenario_predictors_modified:
                st.write("Modified Predictor Layers:")
                for path in st.session_state.scenario_predictors_modified:
                    st.caption(f"-> {os.path.basename(path)}")


        with st.expander("Advanced Options: Policy & Simulation Mode", expanded=True):
            use_policy_demand = st.checkbox(
                "Override historical trends with policy targets", 
                value=st.session_state.get('use_policy_demand_choice', False),
                key='use_policy_demand_choice'
            )
            
            if use_policy_demand:
                st.info("Edit the pixel counts below to set policy-driven demands for each transition.")
                edited_counts = st.data_editor(
                    st.session_state.transition_counts, 
                    use_container_width=True,
                    key='counts_editor'
                )
                final_counts = edited_counts
            else:
                final_counts = st.session_state.transition_counts
            
            use_stochastic = st.checkbox(
                "Enable Stochastic Simulation (for uncertainty analysis)", 
                value=st.session_state.get('use_stochastic_choice', False),
                key='use_stochastic_choice'
            )
        
        # --- BUTTON TO RUN OR RE-RUN THE SIMULATION ---
        if st.button("🛰️ Run Simulation"):
            progress_bar = st.progress(0)
            status = st.status("Starting simulation...", expanded=True)
            def cb(p, t): 
                status.update(label=t)
                progress_bar.progress(p)
            
            targets = st.session_state.uploaded_targets_with_years
            all_predictors = st.session_state.uploaded_predictors + st.session_state.get('generated_predictors', [])
            
            future_lc_path = prediction.run_simulation(
                lc_end_file=targets[-1]['file'],
                predictor_files=all_predictors,
                transition_counts=final_counts,
                trained_model_paths=st.session_state.model_paths,
                temp_dir=st.session_state.temp_dir,
                stochastic=use_stochastic,
                progress_callback=cb
            )
            st.session_state.predicted_filepath = future_lc_path
            st.session_state.simulation_complete = True
            
            # --- CRITICAL FIX: Removed st.rerun() ---
            # Now, the success message will appear reliably.
            status.update(label="Simulation Complete!", state="complete")
            st.success("✅ Simulation process finished successfully! You can now proceed to Visualization.")

        st.divider()
        if st.session_state.simulation_complete:
            st.subheader("Results from Last Simulation Run")
            st.write("You can view the results in the **Visualization** step or download the predicted map below.")

elif st.session_state.active_step == "Visualization":
    st.header("Step 5: Visualize and Export Results")
    st.markdown("""
    Here you can:

    - View prediction result as a map.
    - Export the predicted land cover as PNG or GeoTIFF.
    - Export the tables (e.g., Transition Pixel Counts Matrix) as CSV.
    """)
    if not st.session_state.simulation_complete: st.warning("⚠️ Please generate a simulation result in Step 4.")
    else:
        st.info("Displaying interactive map of the simulated future land cover.")
        with st.spinner("Generating and loading map..."):
            class_legends_dict = st.session_state.class_legends['Class Name'].to_dict()
            m = visualization.create_interactive_map(
                target_files_with_years=st.session_state.uploaded_targets_with_years,
                prediction_filepath=st.session_state.predicted_filepath,
                class_legends=class_legends_dict
            )
            
            st_folium(m, width=None, height=700)

        if st.session_state.simulation_complete and st.session_state.predicted_filepath:
            st.divider()
            st.subheader("Download Center")
            
            col1, col2 = st.columns(2)
            
            # Column 1: Map Downloads
            with col1:
                st.markdown("##### Map Outputs")
                # GeoTIFF Download
                with open(st.session_state.predicted_filepath, "rb") as fp:
                    st.download_button(
                        label="📥 Download Predicted Map (GeoTIFF)", 
                        data=fp, 
                        file_name="predicted_land_cover.tif"
                    )
                
                # --- NEW: Add a text input for the map title ---
                map_title = st.text_input(
                    "Enter Title for Map Image:", 
                    value="Simulated Future Land Cover"
                )

                # Map Image (PNG) Download
                with st.spinner("Generating map image for download..."):
                    # --- MODIFIED: Pass the user's title to the function ---
                    img_bytes = visualization.create_downloadable_map_image(
                        st.session_state.predicted_filepath, 
                        class_legends_dict,
                        title=map_title
                    )
                st.download_button(
                    label="🖼️ Download Map Image (PNG)",
                    data=img_bytes,
                    file_name="predicted_map.png"
                )

            # Column 2: Data Table Downloads
            with col2:
                st.markdown("##### Data Outputs")
                # Transition Counts CSV
                if st.session_state.transition_counts is not None:
                    csv_transitions = st.session_state.transition_counts.to_csv().encode('utf-8')
                    st.download_button(
                        label="📊 Download Transition Counts (CSV)", 
                        data=csv_transitions, 
                        file_name="transition_counts.csv"
                    )
                
                # Model Accuracy CSV
                if st.session_state.model_accuracies:
                    acc_df = pd.DataFrame.from_dict(st.session_state.model_accuracies, orient='index', columns=['Accuracy'])
                    acc_df.index.name = "Transition"
                    csv_accuracy = acc_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📈 Download Accuracy Matrix (CSV)",
                        data=csv_accuracy,
                        file_name="model_accuracies.csv"
                    )

