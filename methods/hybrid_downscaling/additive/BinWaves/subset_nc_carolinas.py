import xarray as xr
import numpy as np

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
    
    # Select the region
    subset = ds.sel(
        longitude=slice(-76.5, -74.5),
        latitude=slice(35, 37)
    )
    
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