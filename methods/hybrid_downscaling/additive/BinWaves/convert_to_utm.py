import xarray as xr
import numpy as np
from pyproj import Transformer

# Open the original dataset
ds = xr.open_dataset('jen_north_carolina_spec.nc')

# Create a transformer from WGS84 (degrees) to UTM Zone 18N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Get the original coordinates (assuming they're named 'lon' and 'lat')
lon = ds.lon.values
lat = ds.lat.values

# Convert to UTM
utm_x, utm_y = transformer.transform(lon, lat)

# Create a new dataset with UTM coordinates
ds_utm = ds.copy()

# Replace the coordinates
ds_utm = ds_utm.rename({'lon': 'utm_x', 'lat': 'utm_y'})
ds_utm = ds_utm.assign_coords(utm_x=('utm_x', utm_x))
ds_utm = ds_utm.assign_coords(utm_y=('utm_y', utm_y))

# Add coordinate attributes
ds_utm.utm_x.attrs = {
    'long_name': 'UTM Easting',
    'units': 'meters',
    'standard_name': 'projection_x_coordinate',
    'coordinate_system': 'UTM Zone 18N'
}

ds_utm.utm_y.attrs = {
    'long_name': 'UTM Northing',
    'units': 'meters',
    'standard_name': 'projection_y_coordinate',
    'coordinate_system': 'UTM Zone 18N'
}

# Add global attributes about the coordinate system
ds_utm.attrs['coordinate_system'] = 'UTM Zone 18N (EPSG:32618)'
ds_utm.attrs['original_coordinates'] = 'WGS84 (EPSG:4326)'

# Save the new dataset
output_file = 'jen_north_carolina_spec_utm.nc'
ds_utm.to_netcdf(output_file)
print(f"Created new file with UTM coordinates: {output_file}")

# Print some information about the conversion
print("\nCoordinate ranges:")
print(f"Original longitude range: {lon.min():.2f} to {lon.max():.2f} degrees")
print(f"Original latitude range: {lat.min():.2f} to {lat.max():.2f} degrees")
print(f"UTM X range: {utm_x.min():.2f} to {utm_x.max():.2f} meters")
print(f"UTM Y range: {utm_y.min():.2f} to {utm_y.max():.2f} meters") 