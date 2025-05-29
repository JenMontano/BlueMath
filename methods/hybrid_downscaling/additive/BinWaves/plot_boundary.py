import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Grid parameters
params = {
    'xpc': 480000,        # Moved further right
    'ypc': 3750000,       # Y position
    'alpc': -45,          # Negative 45 degrees for NE alignment
    'xlenc': 100000,      # Width
    'ylenc': 180000,      # Length - significantly longer for NE direction
    'mxc': 368,
    'myc': 351,
    'xpinp': np.float64(333763.57404016773),
    'ypinp': np.float64(3763616.449980566),
    'alpinp': 0,
    'mxinp': 368,
    'myinp': 351,
    'dxinp': np.float64(500.0),
    'dyinp': np.float64(500.0)
}

# Create the corners of the rectangle
rect_width = params['xlenc']   # Shorter dimension
rect_height = params['ylenc']  # Longer dimension
rect_center_x = params['xpc']
rect_center_y = params['ypc']

# First create a rectangle with longer axis vertical
corners_x = np.array([
    rect_center_x - rect_width/2,   # Left bottom
    rect_center_x + rect_width/2,   # Right bottom
    rect_center_x + rect_width/2,   # Right top
    rect_center_x - rect_width/2,   # Left top
    rect_center_x - rect_width/2    # Close the polygon
])

corners_y = np.array([
    rect_center_y - rect_height/2,  # Left bottom
    rect_center_y - rect_height/2,  # Right bottom
    rect_center_y + rect_height/2,  # Right top
    rect_center_y + rect_height/2,  # Left top
    rect_center_y - rect_height/2   # Close the polygon
])

# Rotate the corners
theta = np.deg2rad(params['alpc'])
corners_x_rot = rect_center_x + (corners_x - rect_center_x) * np.cos(theta) - (corners_y - rect_center_y) * np.sin(theta)
corners_y_rot = rect_center_y + (corners_x - rect_center_x) * np.sin(theta) + (corners_y - rect_center_y) * np.cos(theta)

# Load your bathymetry data
# Replace this with your actual bathymetry data loading
lon = np.linspace(-100000, 600000, 700)
lat = np.linspace(3.45e6, 4.1e6, 650)
lon_mesh, lat_mesh = np.meshgrid(lon, lat)
dummy_bathy = -2000 + 1000 * np.sin(lon_mesh/200000) * np.cos(lat_mesh/200000)
bathy = xr.DataArray(dummy_bathy, coords={'lat': lat, 'lon': lon})

# Create the plot
plt.figure(figsize=(12, 10))

# Plot bathymetry
bathy_plot = plt.pcolormesh(
    bathy.lon, bathy.lat, bathy, 
    cmap='Blues_r', 
    shading='auto'
)
plt.colorbar(bathy_plot, label='Depth (m)')

# Plot the rotated rectangle boundary
plt.plot(corners_x_rot, corners_y_rot, 'g-', linewidth=2, label='Grid boundary')

# Customize the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.title('Bathymetry with Rotated Grid Boundary')
plt.show() 