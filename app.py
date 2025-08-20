import streamlit as st
import leafmap
import tempfile
from landuse_tool import data_loader, utils, training, prediction, scenarios, visualization

# st.set_page_config(layout="wide")

# st.title("🌍 Land Use Monitoring & Prediction Tool")

# # Sidebar for controls
# st.sidebar.header("Data & Scenarios")

# # Option to upload raster instead of hardcoding path
# uploaded_file = st.sidebar.file_uploader("Upload a land cover raster (.tif)", type=["tif"])

# # Initialize map (OSM base)
# m = leafmap.Map(center=[48, 67], zoom=4)  # center on Kazakhstan, adjust as needed

# if uploaded_file:
#     # Save uploaded file to a temporary path
#     import tempfile
#     import rasterio

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#         tmp.write(uploaded_file.read())
#         raster_path = tmp.name

#     # Use your data_loader but pass path dynamically
#     lc, mask, profile = data_loader.load_target(raster_path)  # you’ll need to edit load_target() to accept a path
#     X_stack = data_loader.load_predictors(mask)

#     # Train model
#     X, y = utils.sample_training_data(X_stack, lc, mask)
#     model = training.train_rf(X, y)

#     # Predict
#     predicted = prediction.predict_map(model, X_stack, mask)

#     # Show prediction
#     vis_path = visualization.save_prediction_as_tif(predicted, profile)  # modify your vis module to save GeoTIFF
#     m.add_raster(vis_path, layer_name="Predicted Land Cover")

# else:
#     st.sidebar.info("Upload a raster to start analysis. Default map shows OSM basemap.")

# # Display map
# m.to_streamlit(height=600)

st.title("🌍 Land Use Monitoring & Prediction Tool")

# Multiple file uploader for land cover rasters
uploaded_files = st.file_uploader(
    "Upload land cover rasters (≥2 years). Use filenames like 'landcover_2020.tif'",
    type=["tif", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    raster_files = []
    temp_paths = []
    for file in uploaded_files:
        try:
            year = int(file.name.split("_")[-1].split(".")[0])
        except:
            st.error(f"Filename {file.name} must end with year (e.g., landcover_2020.tif).")
            continue
        
        # Save file to a temp location
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        tmp.write(file.read())
        tmp.close()
        temp_paths.append(tmp.name)  # keep track to maybe clean later

        raster_files.append((year, tmp.name))

    if len(raster_files) < 2:
        st.warning("Please upload at least 2 rasters (different years).")
    else:
        st.success(f"Loaded {len(raster_files)} rasters: {', '.join(str(y) for y, _ in raster_files)}")

        # Now load_targets gets file paths, as it expects
        data_by_year = data_loader.load_targets(raster_files)

        m = leafmap.Map(center=[48.0, 67.0], zoom=4)
        for year, (lc, mask, profile) in data_by_year.items():
            # add raster by the temp file path you saved earlier
            # find the corresponding temp path for the year
            temp_path = next(path for y, path in raster_files if y == year)
            m.add_raster(temp_path, layer_name=f"Land Cover {year}", fit_bounds=False)

        m.to_streamlit(height=500)