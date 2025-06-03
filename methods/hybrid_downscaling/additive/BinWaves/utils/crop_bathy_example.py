import xarray as xr
from utils.crop_bathy import crop_bathy_for_rotated_grid, plot_cropped_bathy

# Load the bathymetry data
bathy = (
    xr.open_dataset("common_inputs/North_Carolina_utm18.nc")
    .rename({"cx": "lon", "cy": "lat"})
    .transpose("lat", "lon")
    .sortby("lat", ascending=False)
    .elevation
)

# Define grid parameters
grid_params = {
    'xpc': 287634.73,    # UTM Easting of origin (meters)
    'ypc': 3676471.67,   # UTM Northing of origin (meters)
    'alpc': 40,          # Rotation angle (degrees)
    'xlenc': 450000,     # Grid length in x (meters)
    'ylenc': 250000,     # Grid length in y (meters)
}

# Crop bathymetry with 10km buffer
buffer_distance = 10000  # 10 km buffer
cropped_bathy = crop_bathy_for_rotated_grid(bathy, grid_params, buffer_distance)

# Plot the result
fig, ax = plot_cropped_bathy(cropped_bathy, grid_params, buffer_distance)

# Save the cropped bathymetry
output_file = "outputs/cropped_bathy_with_buffer.nc"
cropped_bathy.to_netcdf(output_file)
print(f"Saved cropped bathymetry to {output_file}")

# Save the plot
fig.savefig("outputs/cropped_bathy_plot.png", dpi=300, bbox_inches='tight')
print("Saved plot to outputs/cropped_bathy_plot.png") 