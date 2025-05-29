import xarray as xr
import numpy as np
from pyproj import Transformer

def subset_carolinas(input_file, output_file):
    """
    Subset a NetCDF file to cover the Carolinas region:
    Longitude: 76.5°W to 74.5°W
    Latitude: 35°N to 37°N
    
    Parameters:
    -----------
    input_file : str
        Path to the input NetCDF file
    output_file : str
        Path where the subset NetCDF file will be saved
    """
    # Read the input NetCDF file
    ds = xr.open_dataset(input_file)
    
    # Create transformer from UTM to geographic coordinates
    # UTM zone 18N for the Carolinas
    transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    
    # Convert UTM coordinates to geographic
    cx, cy = np.meshgrid(ds.cx.values, ds.cy.values)
    lon, lat = transformer.transform(cx, cy)
    
    # Find indices for the region of interest
    lon_mask = (lon >= -76.5) & (lon <= -74.5)
    lat_mask = (lat >= 35) & (lat <= 37)
    mask = lon_mask & lat_mask
    
    # Get the indices where the mask is True
    cy_indices = np.where(np.any(mask, axis=1))[0]
    cx_indices = np.where(np.any(mask, axis=0))[0]
    
    # Select the region
    subset = ds.isel(cy=slice(cy_indices[0], cy_indices[-1] + 1),
                    cx=slice(cx_indices[0], cx_indices[-1] + 1))
    
    # Save the subset to a new NetCDF file
    subset.to_netcdf(output_file)
    print(f"Subset saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Subset NetCDF file for Carolinas region')
    parser.add_argument('input_file', help='Path to input NetCDF file')
    parser.add_argument('output_file', help='Path for output NetCDF file')
    
    args = parser.parse_args()
    
    subset_carolinas(args.input_file, args.output_file) 