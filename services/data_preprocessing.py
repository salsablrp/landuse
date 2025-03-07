import geopandas as gpd
import rasterio
from rasterio.merge import merge

def preprocess_vector_data(input_shapefiles):
    """
    Merge and clean multiple vector datasets
    """
    gdf_list = [gpd.read_file(shapefile) for shapefile in input_shapefiles]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    
    # Clean and standardize data
    merged_gdf.dropna(subset=['land_use_type'], inplace=True)
    merged_gdf = merged_gdf.to_crs('EPSG:4326')
    
    return merged_gdf

def merge_raster_datasets(raster_paths):
    """
    Merge multiple raster datasets
    """
    raster_to_mosaic = [rasterio.open(path) for path in raster_paths]
    mosaic, output_transform = merge(raster_to_mosaic)
    
    return mosaic, output_transform