import pandas as pd
import xarray as xr
import numpy as np

# File paths
pkl_file = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/common_inputs/csiro_dataframe_44088_sat_corr.pkl'
nc_file = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/outputs/jen_north_carolina_spec_utm.nc'

# Read the files
print("=== Reading Files ===")
df = pd.read_pickle(pkl_file)
ds = xr.open_dataset(nc_file)

print("\n=== NetCDF File Structure ===")
print("\nVariables:")
for var_name, var in ds.variables.items():
    print(f"\nVariable: {var_name}")
    print(f"Dimensions: {var.dims}")
    print(f"Shape: {var.shape}")
    print(f"Attributes: {var.attrs}")

print("\nGlobal Attributes:")
for attr_name, attr_value in ds.attrs.items():
    print(f"{attr_name}: {attr_value}")

print("\n=== Pickle DataFrame Structure ===")
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Columns:")
print(df.columns.tolist())
print("\nDataFrame Sample:")
print(df.head())

# Close the dataset
ds.close() 