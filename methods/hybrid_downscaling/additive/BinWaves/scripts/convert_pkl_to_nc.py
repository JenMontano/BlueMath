import pandas as pd
import xarray as xr
import numpy as np

# File paths
pkl_file = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/common_inputs/csiro_dataframe_44088_sat_corr.pkl'
ref_nc_file = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/outputs/jen_north_carolina_spec_utm.nc'
output_nc_file = pkl_file.replace('.pkl', '.nc')

# Read the pickle file
df = pd.read_pickle(pkl_file)

# Read the reference NetCDF file to get attributes
ref_ds = xr.open_dataset(ref_nc_file)

# Convert DataFrame to xarray Dataset
# First, we need to ensure the DataFrame has proper indexes
# We'll assume the DataFrame has appropriate columns and structure
# You might need to adjust this part based on your DataFrame's structure
ds = df.to_xarray()

# Copy attributes from reference file
for attr_name, attr_value in ref_ds.attrs.items():
    ds.attrs[attr_name] = attr_value

# Save to NetCDF
ds.to_netcdf(output_nc_file)

print(f"Conversion complete. Output saved to: {output_nc_file}") 