import xarray as xr
import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
era5_spectrum = xr.open_dataset("inputs/superpoint_result.nc")
model_parameters = pd.read_csv("CASES/swan_cases.csv").to_dict(orient="list")

# Get unique frequencies and directions from the case parameters
case_freqs = sorted(list(set(model_parameters['freq'])))
case_dirs = sorted(list(set(model_parameters['dir'])))

print(f"ERA5 original frequencies: {len(era5_spectrum.freq.values)} values")
print(f"ERA5 original directions: {len(era5_spectrum.dir.values)} values")
print(f"Case frequencies: {len(case_freqs)} values")
print(f"Case directions: {len(case_dirs)} values")

print(f"\nERA5 freq range: {era5_spectrum.freq.min().values:.4f} to {era5_spectrum.freq.max().values:.4f}")
print(f"Case freq range: {min(case_freqs):.4f} to {max(case_freqs):.4f}")
print(f"ERA5 dir range: {era5_spectrum.dir.min().values:.1f} to {era5_spectrum.dir.max().values:.1f}")
print(f"Case dir range: {min(case_dirs):.1f} to {max(case_dirs):.1f}")

# Interpolate the ERA5 data to match case frequencies and directions
print(f"\nInterpolating ERA5 data to case frequencies and directions...")

# Create new coordinates for interpolation
new_freq = xr.DataArray(case_freqs, dims=['freq'], coords={'freq': case_freqs})
new_dir = xr.DataArray(case_dirs, dims=['dir'], coords={'dir': case_dirs})

# Interpolate the spectrum data
interpolated_efth = era5_spectrum.efth.interp(freq=new_freq, dir=new_dir, method='linear')

# Create new dataset with interpolated data
interpolated_era5 = xr.Dataset({
    'efth': interpolated_efth,
    'Depth': era5_spectrum.Depth.interp(dir=new_dir, method='linear'),
    'Wspeed': era5_spectrum.Wspeed.interp(dir=new_dir, method='linear'),
    'Wdir': era5_spectrum.Wdir.interp(dir=new_dir, method='linear'),
    'longitude': era5_spectrum.longitude.interp(dir=new_dir, method='linear'),
    'latitude': era5_spectrum.latitude.interp(dir=new_dir, method='linear')
})

# Add the original coordinates
interpolated_era5 = interpolated_era5.assign_coords({
    'time': era5_spectrum.time,
    'freq': new_freq,
    'dir': new_dir
})

print(f"Interpolation complete!")
print(f"New dataset shape: {interpolated_efth.shape}")
print(f"New dataset min: {interpolated_efth.min().values}")
print(f"New dataset max: {interpolated_efth.max().values}")
print(f"New dataset mean: {interpolated_efth.mean().values}")

# Save the interpolated dataset
output_file = "inputs/superpoint_result_interpolated.nc"
print(f"\nSaving interpolated dataset to {output_file}...")
interpolated_era5.to_netcdf(output_file)
print(f"Saved successfully!")

# Close the original dataset
era5_spectrum.close() 