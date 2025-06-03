import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_rotated_grid_corners(corner_utm, angle_deg, xlenc, ylenc):
    """
    Calculate corners of a rotated grid.
    
    Parameters
    ----------
    corner_utm : tuple
        (x, y) coordinates of the bottom-left corner in UTM
    angle_deg : float
        Rotation angle in degrees (clockwise from north)
    xlenc : float
        Grid length in x direction (meters)
    ylenc : float
        Grid length in y direction (meters)
    
    Returns
    -------
    tuple
        Arrays of x and y coordinates of the grid corners
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create rotation matrix
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Create unrotated rectangle corners
    dx = np.array([0, xlenc, xlenc, 0, 0])
    dy = np.array([0, 0, ylenc, ylenc, 0])
    points = np.column_stack([dx, dy])
    
    # Rotate points
    rotated = np.dot(points, R.T)
    
    # Translate to corner position
    x = rotated[:, 0] + corner_utm[0]
    y = rotated[:, 1] + corner_utm[1]
    
    return x, y

def plot_bathymetry(bathy, grid_params=None, max_z=None, min_z=-500, figsize=(15, 10)):
    """
    Plot bathymetry with optional grid overlay.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    grid_params : dict, optional
        Grid parameters containing:
        - corner: (x, y) tuple of grid origin
        - angle: rotation angle in degrees
        - xlenc: grid length in x (meters)
        - ylenc: grid length in y (meters)
    max_z : float, optional
        Maximum depth to plot
    min_z : float, optional
        Minimum depth to plot
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    tuple
        Figure and axis objects, and grid corner coordinates if grid_params provided
    """
    # Set up the projection
    proj = ccrs.UTM(zone=18)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
    
    # Create levels for bathymetry
    if max_z is None:
        max_z = float(bathy.max())
    levels = np.linspace(min_z, max_z, 20)
    
    # Plot bathymetry
    x_coord = 'lon' if 'lon' in bathy.coords else 'cx'
    y_coord = 'lat' if 'lat' in bathy.coords else 'cy'
    
    bathy.plot.contourf(
        ax=ax,
        x=x_coord,
        y=y_coord,
        levels=levels,
        cmap='Blues_r',
        add_colorbar=True,
        extend='both',
        transform=proj
    )
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20, edgecolor='black')
    
    # Plot grid if parameters provided
    grid_corners = None
    if grid_params is not None:
        corner = (grid_params['xpc'], grid_params['ypc'])
        x_coords, y_coords = get_rotated_grid_corners(
            corner, 
            grid_params['alpc'], 
            grid_params['xlenc'], 
            grid_params['ylenc']
        )
        ax.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8, 
                label='Computational grid', transform=proj)
        ax.legend()
        grid_corners = (x_coords, y_coords)
    
    return fig, ax, grid_corners

def crop_bathy_to_grid(bathy, grid_params, buffer_distance=10000):
    """
    Crop bathymetry to cover a rotated grid area with buffer.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Input bathymetry data
    grid_params : dict
        Dictionary containing:
        - xpc: UTM x-coordinate of origin
        - ypc: UTM y-coordinate of origin
        - alpc: rotation angle in degrees
        - xlenc: grid length in x (meters)
        - ylenc: grid length in y (meters)
    buffer_distance : float
        Buffer distance in meters around the grid
    
    Returns
    -------
    xr.DataArray
        Cropped bathymetry
    """
    # Get grid corners
    corner = (grid_params['xpc'], grid_params['ypc'])
    x_coords, y_coords = get_rotated_grid_corners(
        corner, 
        grid_params['alpc'], 
        grid_params['xlenc'], 
        grid_params['ylenc']
    )
    
    # Get bounds with buffer
    x_min = np.min(x_coords) - buffer_distance
    x_max = np.max(x_coords) + buffer_distance
    y_min = np.min(y_coords) - buffer_distance
    y_max = np.max(y_coords) + buffer_distance
    
    print(f"Cropping bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # Get coordinate names
    x_coord = 'lon' if 'lon' in bathy.coords else 'cx'
    y_coord = 'lat' if 'lat' in bathy.coords else 'cy'
    
    # Crop bathymetry
    cropped = bathy.sel(
        {x_coord: slice(x_min, x_max),
         y_coord: slice(y_max, y_min)}  # Note: slice from max to min for descending coordinates
    )
    
    return cropped 