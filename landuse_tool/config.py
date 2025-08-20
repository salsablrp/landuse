import os

# Root data folder
DATA_DIR = r"E:\Land_Use\data"

# Target land cover raster (label)
TARGET_RASTER = os.path.join(DATA_DIR, "Kazakhstan_LandCover_2001.tif")

# List of predictor raster file names
PREDICTOR_FILES = [
    "kaz_pd_2020_1km_UNadj.tif",
    "dist_uni.tif",
    "dist_pop_area.tif",
    "dist_rail_pt.tif",
    "dist_remote_area.tif",
    "dist_rail_ln.tif",
    "dist_road.tif",
    "dist_waterways.tif",
    "Kazakhstan_SoilMoisture.tif",
    "Kazakhstan_Precip.tif",
    "Kazakhstan_AGB.tif",
    "Kazakhstan_Temperature.tif"
]

PREDICTOR_PATHS = [os.path.join(DATA_DIR, f) for f in PREDICTOR_FILES]
