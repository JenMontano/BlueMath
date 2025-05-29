import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon

def generate_rotated_grid_parameters(bathy_data: xr.DataArray, rotation_angle: float, 
                               x_limits: tuple = None, y_limits: tuple = None,
                               buffer_cells: int = 0) -> dict:
    """
    Generate rotated grid parameters for the SWAN model.
    
    Parameters
    ----------
    bathy_data : xr.DataArray
        Bathymetry data with coordinates 'lon' and 'lat' in UTM coordinates
    rotation_angle : float
        Rotation angle in degrees (clockwise from North)
    x_limits : tuple
        (min_x, max_x) to limit the grid extent in UTM coordinates
    y_limits : tuple
        (min_y, max_y) to limit the grid extent in UTM coordinates
    buffer_cells : int, optional
        Number of buffer cells to add around the domain
    """
    # Get domain bounds, using limits if provided
    if x_limits is None:
        x_min, x_max = float(np.nanmin(bathy_data.lon)), float(np.nanmax(bathy_data.lon))
    else:
        x_min, x_max = x_limits
        
    if y_limits is None:
        y_min, y_max = float(np.nanmin(bathy_data.lat)), float(np.nanmax(bathy_data.lat))
    else:
        y_min, y_max = y_limits
    
    # Calculate grid spacing from input data
    dxinp = abs(bathy_data.lon[1].values - bathy_data.lon[0].values)
    dyinp = abs(bathy_data.lat[1].values - bathy_data.lat[0].values)
    
    # Calculate domain dimensions
    width = np.abs(x_max - x_min)
    height = np.abs(y_max - y_min)
    
    # Calculate the center of the domain
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    
    # Convert rotation angle to radians
    angle_rad = np.radians(rotation_angle)
    
    # Create rotation matrix for clockwise rotation
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Create the unrotated rectangle corners relative to center
    corners_unrot = np.array([
        [x_min - center_x, y_min - center_y],  # bottom-left
        [x_max - center_x, y_min - center_y],  # bottom-right
        [x_max - center_x, y_max - center_y],  # top-right
        [x_min - center_x, y_max - center_y]   # top-left
    ])
    
    # Rotate corners around center
    corners_rot = np.dot(corners_unrot, R.T)
    
    # Translate corners back
    corners_final = corners_rot + np.array([center_x, center_y])
    
    # Calculate bounding box of rotated grid
    x_min_rot, y_min_rot = np.min(corners_final, axis=0)
    x_max_rot, y_max_rot = np.max(corners_final, axis=0)
    
    # Calculate dimensions of rotated grid
    xlenc = np.abs(x_max_rot - x_min_rot)
    ylenc = np.abs(y_max_rot - y_min_rot)
    
    # Calculate number of points
    mxc = int(xlenc / dxinp) + 2 * buffer_cells - 1
    myc = int(ylenc / dyinp) + 2 * buffer_cells - 1
    
    # Calculate origin point (bottom-left corner with buffer)
    xpc = x_min_rot - buffer_cells * dxinp
    ypc = y_min_rot - buffer_cells * dyinp
    
    return {
        "xpc": xpc,  # x origin computational grid
        "ypc": ypc,  # y origin computational grid
        "alpc": rotation_angle,  # x-axis direction computational grid
        "xlenc": xlenc,  # grid length in x
        "ylenc": ylenc,  # grid length in y
        "mxc": mxc,  # number mesh x computational grid
        "myc": myc,  # number mesh y computational grid
        "xpinp": x_min,  # x origin input grid
        "ypinp": y_min,  # y origin input grid
        "alpinp": 0,  # x-axis direction input grid (always 0 as input is not rotated)
        "mxinp": len(bathy_data.lon) - 1,  # number mesh x input
        "myinp": len(bathy_data.lat) - 1,  # number mesh y input
        "dxinp": dxinp,  # size mesh x input
        "dyinp": dyinp,  # size mesh y input
        "corners": corners_final.tolist()  # corner coordinates for plotting
    }

def plot_bathy_with_grid(bathy: xr.DataArray, grid_params: dict, utm_zone: int = 18, skip_lines: int = 5):
    """
    Plot bathymetry with rotated SWAN grid overlay.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    grid_params : dict
        Grid parameters from generate_rotated_grid_parameters
    utm_zone : int
        UTM zone number
    skip_lines : int
        Plot every nth grid line to reduce density
    """
    # Set up the projection
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})
    
    # Create custom levels for bathymetry (negative for water, positive for land)
    levels = [-1000, -500, -200, -100, -50, -25, -10, 0, 10]
    
    # Plot bathymetry
    contour = bathy.plot.contourf(
        ax=ax,
        x='lon',
        y='lat',
        levels=levels,
        cmap="Blues_r",
        add_colorbar=True,
        extend='both',
        transform=proj
    )
    
    # Modify colorbar
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('elevation relative to sea level [m]')
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20, edgecolor='black')
    
    # Get corner coordinates and convert to numpy array if not already
    corners = np.array(grid_params['corners'])
    
    # Plot rotated grid outline
    corners_plot = np.vstack([corners, corners[0]])  # Close the polygon
    ax.plot(corners_plot[:, 0], corners_plot[:, 1], 
            'r-', linewidth=2, transform=proj, zorder=10, label='Grid boundary')
    
    # Calculate vectors for grid directions
    bottom_left = corners[0]
    bottom_right = corners[1]
    top_right = corners[2]
    
    x_vector = (bottom_right - bottom_left) / grid_params['mxc']
    y_vector = (top_right - bottom_right) / grid_params['myc']
    
    # Plot vertical grid lines (every skip_lines)
    for i in range(0, grid_params['mxc'] + 1, skip_lines):
        start_point = bottom_left + x_vector * i
        end_point = start_point + y_vector * grid_params['myc']
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]],
                'r-', linewidth=0.5, alpha=0.5, transform=proj, zorder=5)
    
    # Plot horizontal grid lines (every skip_lines)
    for j in range(0, grid_params['myc'] + 1, skip_lines):
        start_point = bottom_left + y_vector * j
        end_point = start_point + x_vector * grid_params['mxc']
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]],
                'r-', linewidth=0.5, alpha=0.5, transform=proj, zorder=5)
    
    # Set map boundaries based on bathymetry extent with a small buffer
    lon_min, lon_max = float(bathy.lon.min()), float(bathy.lon.max())
    lat_min, lat_max = float(bathy.lat.min()), float(bathy.lat.max())
    
    # Add 5% buffer to the boundaries
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    buffer = 0.05  # 5% buffer
    
    ax.set_extent([
        lon_min - lon_range * buffer,
        lon_max + lon_range * buffer,
        lat_min - lat_range * buffer,
        lat_max + lat_range * buffer
    ], crs=proj)
    
    # Set aspect ratio
    ax.set_aspect('equal')
    
    # Add title
    plt.title('Bathymetry with Rotated SWAN Grid')
    
    return fig, ax 