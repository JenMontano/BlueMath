import xarray as xr
import rioxarray
import numpy as np
from pyproj import CRS

def convert_gebco_to_BinWaves(input_file, output_file, input_crs="EPSG:4326", output_crs="EPSG:32618", rename_coords=True):
    """
    Convert bathymetry data from GEBCO to BinWaves format.
    
    Parameters:
    -----------
    input_file : str
        Path to the input bathymetry file in netCDF format
    output_file : str
        Path where the converted file will be saved
    input_crs : str
        Input coordinate reference system (default: "EPSG:4326" for WGS84)
    output_crs : str
        Output coordinate reference system (default: "EPSG:32618" for UTM18N)
    rename_coords : bool
        Whether to rename x/y coordinates to cx/cy (default: True)
    """
    # Read the input file
    ds = xr.open_dataset(input_file)

    # Set the spatial reference system for the input
    ds.rio.write_crs(input_crs, inplace=True)

    # Define the target projection
    target_crs = CRS.from_string(output_crs)

    # Reproject to target CRS
    ds_proj = ds.rio.reproject(target_crs)

    # Convert elevation to double and replace no_data values with NaN
    ds_proj['elevation'] = ds_proj.elevation.astype('float64')
    ds_proj['elevation'] = ds_proj.elevation.where(ds_proj.elevation != ds_proj.elevation.rio.nodata)

    # Rename x and y coordinates if requested
    if rename_coords:
        ds_proj = ds_proj.rename({'x': 'cx', 'y': 'cy'})
        # Add _FillValue attribute to cx and cy
        ds_proj['cx'].attrs['_FillValue'] = np.nan
        ds_proj['cy'].attrs['_FillValue'] = np.nan

    # Save to netCDF
    ds_proj.to_netcdf(output_file)
    print(f"Converted file saved as {output_file}")

# if __name__ == "__main__":
#     # Example usage for GEBCO to UTM18N conversion
#     input_file = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/raw_data/bathy/gebco_2024_n37.5763_s31.9852_w-79.3978_e-73.0076.nc'
#     output_file = 'gebco_utm18.nc'
    
#     # Default usage (GEBCO WGS84 to UTM18N)
#     convert_bathy_projection(input_file, output_file)
    
#     # Example with custom CRS (e.g., converting to UTM zone 17N)
#     # convert_bathy_projection(input_file, 'gebco_utm17.nc', 
#     #                        input_crs="EPSG:4326", 
#     #                        output_crs="EPSG:32617") 
  