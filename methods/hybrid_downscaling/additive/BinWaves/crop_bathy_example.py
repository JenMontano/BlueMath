import xarray as xr
import numpy as np
from pathlib import Path
from utils.crop_bathy import plot_bathymetry, crop_bathy_to_grid
import matplotlib.pyplot as plt

# Create outputs directory if it doesn't exist
Path("outputs").mkdir(exist_ok=True)

# Parameters
name = 'north_carolina'
resolution = 500  # meters

# Grid parameters
grid_params = {
    'xpc': 287634.73,    # UTM Easting of origin (meters)
    'ypc': 3676471.67,   # UTM Northing of origin (meters)
    'alpc': 40,          # Rotation angle (degrees)
    'xlenc': 450000,     # Grid length in x (meters)
    'ylenc': 250000,     # Grid length in y (meters)
}

# Point of interest
point = (514397.61, 4051843.74)
point_label = "Point of Interest"

# Load bathymetry
print("Loading bathymetry data...")
bathy = (
    xr.open_dataset("common_inputs/North_Carolina_utm18.nc")
    .elevation
    .rename({'cx': 'lon', 'cy': 'lat'})
    .transpose('lat', 'lon')
    .sortby('lat', ascending=False)
)

# Print area extent
x_extent = float(bathy.lon.max() - bathy.lon.min()) / 1000  # km
y_extent = float(bathy.lat.max() - bathy.lat.min()) / 1000  # km
print(f'Area extent: {x_extent:.2f} km x {y_extent:.2f} km')

# Plot full domain with grid overlay and point
print("\nPlotting full domain with grid overlay...")
fig, ax, (x_coords, y_coords) = plot_bathymetry(
    bathy,
    grid_params=grid_params,
    min_z=-500,
    max_z=10,
    points=[point],
    point_labels=[point_label]
)

# Save full domain plot
fig.savefig('outputs/full_domain.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Crop bathymetry around grid with larger buffer
print("\nCropping bathymetry around grid...")
buffer_distance = 50000  # 50 km buffer
cropped_bathy = crop_bathy_to_grid(bathy, grid_params, buffer_distance)

# Plot cropped domain
print("\nPlotting cropped domain...")
fig, ax, _ = plot_bathymetry(
    cropped_bathy,
    grid_params=grid_params,
    min_z=-500,
    max_z=10,
    points=[point],
    point_labels=[point_label]
)

# Save cropped domain plot
fig.savefig('outputs/cropped_domain.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Save cropped bathymetry
output_file = f"outputs/bathy_{name}_{resolution}m.nc"
print(f"\nSaving cropped bathymetry to {output_file}")
cropped_bathy.to_netcdf(output_file)

print("\nDone!") 