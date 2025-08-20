import streamlit as st
import leafmap
import tempfile
from landuse_tool import data_loader, utils, training, prediction, scenarios, visualization

st.set_page_config(layout="wide")
st.title("🌍 Land Use Monitoring & Prediction Tool")

# --- SESSION STATE RESET ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "data_by_year" not in st.session_state:
    st.session_state.data_by_year = None
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
uploaded_files = st.file_uploader(
    "Upload land cover rasters (≥2 years). Use filenames like 'landcover_2020.tif'",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

if uploaded_files and st.session_state.step == 0:
    raster_files = []
    for file in uploaded_files:
        try:
            year = int(file.name.split("_")[-1].split(".")[0])
        except:
            st.error(f"Filename {file.name} must end with year (e.g., landcover_2020.tif).")
            continue

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp.write(file.read())
        tmp.close()
        raster_files.append((year, tmp.name))

    if len(raster_files) < 2:
        st.warning("Please upload at least 2 rasters (different years).")
    else:
        st.success(f"Loaded {len(raster_files)} rasters.")
        st.session_state.data_by_year = data_loader.load_targets(raster_files)

    if st.button("➡️ Proceed to Training"):
        st.session_state.step = 1


# --- STEP 2: Training ---
if st.session_state.step >= 1:
    st.header("Step 2: Train Model")
    if st.button("⚡ Train Random Forest"):
        # Use latest year for training
        latest_year = max(st.session_state.data_by_year.keys())
        lc, mask, _ = st.session_state.data_by_year[latest_year]
        X_stack = data_loader.load_predictors(mask)
        X, y = utils.sample_training_data(X_stack, lc, mask)
        model = training.train_rf(X, y)
        st.session_state.model = model
        st.success("Model trained!")

    if st.button("➡️ Proceed to Prediction"):
        st.session_state.step = 2


# --- STEP 3: Prediction ---
if st.session_state.step >= 2:
    st.header("Step 3: Predict Land Cover")
    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        if st.button("🛰️ Run Prediction"):
            latest_year = max(st.session_state.data_by_year.keys())
            _, mask, profile = st.session_state.data_by_year[latest_year]
            X_stack = data_loader.load_predictors(mask)
            predicted = prediction.predict_map(st.session_state.model, X_stack, mask)
            st.session_state.predicted = predicted
            vis_path = visualization.save_prediction_as_tif(predicted, profile)
            m = leafmap.Map(center=[48.0, 67.0], zoom=4)
            m.add_raster(vis_path, layer_name="Predicted Land Cover")
            m.to_streamlit(height=500)
            st.success("Prediction complete!")

    if st.button("➡️ Proceed to Scenarios"):
        st.session_state.step = 3


# --- STEP 4: Scenarios ---
if st.session_state.step >= 3:
    st.header("Step 4: Run Scenarios")

    scenario_name = st.text_input("Scenario name", "Afforestation")
    layer_ops = st.text_area(
        "Define layer adjustments (JSON-style dict)",
        """{
    "distance_to_roads": { "operator": "increase", "value": 1000 },
    "biomass": { "operator": "increase", "value": 0.2 }
}"""
    )

    if st.button("🌱 Apply Scenario"):
        latest_year = max(st.session_state.data_by_year.keys())
        _, mask, _ = st.session_state.data_by_year[latest_year]
        X_stack = data_loader.load_predictors(mask)
        import json
        adjustments = json.loads(layer_ops)
        scenario_stack = scenarios.apply_scenario(X_stack, adjustments)
        st.session_state.scenario_stack = scenario_stack
        st.success(f"Scenario '{scenario_name}' applied!")

    if st.button("➡️ Proceed to Visualization"):
        st.session_state.step = 4


# --- STEP 5: Visualization ---
if st.session_state.step >= 4:
    st.header("Step 5: Visualization")

    if st.session_state.predicted is not None:
        st.success("✅ Prediction ready for visualization.")
        latest_year = max(st.session_state.data_by_year.keys())
        _, _, profile = st.session_state.data_by_year[latest_year]

        # Buttons for visualization
        cmap_list = ["#d9f0d3", "#addd8e", "#31a354", "#006d2c"]
        title = "Predicted Land Cover Map"

        if st.button("📊 Show Inline Map"):
            fig = visualization.plot_prediction(st.session_state.predicted, cmap_list, title)
            st.pyplot(fig)

        if st.button("💾 Save as GeoTIFF"):
            out_tif = visualization.save_prediction_as_tif(
                st.session_state.predicted, profile, "prediction.tif"
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