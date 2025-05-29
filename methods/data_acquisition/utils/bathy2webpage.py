import xarray as xr
import numpy as np
from shapely.geometry import LineString, mapping
import json
import rasterio
from rasterio import features
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from pyproj import Transformer

def process_netcdf_bathymetry(nc_file, output_file, contour_levels=None):
    """
    Process NetCDF bathymetry file and create GeoJSON contours
    
    Args:
        nc_file (str): Path to NetCDF file
        output_file (str): Path to output GeoJSON file
        contour_levels (list): List of depth levels for contours. If None, auto-generated
    """
    # Open the NetCDF file
    ds = xr.open_dataset(nc_file)
    
    # Get the data using the known variable names
    try:
        bathy = ds['elevation'].values  # Bathymetry/elevation data
        x_utm = ds['cx'].values  # UTM x coordinates
        y_utm = ds['cy'].values  # UTM y coordinates
    except KeyError:
        # Print available variables to help user identify correct names
        print("Available variables in NC file:", list(ds.variables))
        raise
    
    # Create a meshgrid of coordinates
    x_grid, y_grid = np.meshgrid(x_utm, y_utm)
    
    # Create transformer from UTM 18N to WGS84
    transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    
    # Generate contour levels if not provided
    if contour_levels is None:
        contour_levels = np.linspace(np.nanmin(bathy), np.nanmax(bathy), 20)
    
    # Generate contours
    contours = plt.contour(x_grid, y_grid, bathy, levels=contour_levels)
    plt.close()  # Close the figure since we don't need to display it
    
    # Convert contours to GeoJSON features
    features = []
    
    # Iterate through each contour level
    for i, level in enumerate(contours.levels):
        # Get the paths for this level
        paths = contours.allsegs[i]
        
        # Process each separate path for this level
        for path in paths:
            if len(path) > 1:  # Ensure we have at least 2 points
                # Path is already a numpy array of vertices
                vertices_utm = path
                
                # Transform coordinates from UTM to WGS84
                lon_points, lat_points = transformer.transform(vertices_utm[:, 0], vertices_utm[:, 1])
                
                # Create coordinate pairs for the LineString
                coords = list(zip(lon_points, lat_points))
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "depth": float(level)
                    }
                }
                features.append(feature)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(geojson, f)

def process_geotiff_bathymetry(tiff_file, output_file, contour_interval=100):
    """
    Process GeoTIFF bathymetry file and create GeoJSON contours
    
    Args:
        tiff_file (str): Path to GeoTIFF file
        output_file (str): Path to output GeoJSON file
        contour_interval (float): Interval between contours in data units
    """
    with rasterio.open(tiff_file) as src:
        # Read the data and transform
        data = src.read(1)
        transform = src.transform
        
        # Generate contour levels
        levels = np.arange(
            np.floor(data.min()),
            np.ceil(data.max()),
            contour_interval
        )
        
        # Generate contours
        contours = measure.find_contours(data, levels=levels)
        
        # Convert to GeoJSON features
        features = []
        for level, contour in zip(levels, contours):
            # Transform contour coordinates to geographic coordinates
            coords = []
            for y, x in contour:
                lon, lat = transform * (x, y)
                coords.append([lon, lat])
            
            if len(coords) > 1:  # Ensure we have at least 2 points
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "depth": float(level)
                    }
                }
                features.append(feature)
        
        # Create GeoJSON structure
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(geojson, f)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python process_bathymetry.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Determine file type and process accordingly
    if input_file.endswith('.nc'):
        process_netcdf_bathymetry(input_file, output_file)
    elif input_file.endswith(('.tif', '.tiff')):
        process_geotiff_bathymetry(input_file, output_file)
    else:
        print("Unsupported file format. Please use .nc or .tif files.") 