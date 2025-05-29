import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as colors

def plot_grid_and_bathy():
    # Load bathymetry
    bathy = -(
        xr.open_dataset("outputs/carolinas_gebco_utm18_cy_cx.nc")
        .rename({"cx": "lon", "cy": "lat"})
        .transpose("lat", "lon")
        .sortby("lat", ascending=False)
        .elevation
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot bathymetry
    bathy_mesh = ax.pcolormesh(
        bathy.lon, 
        bathy.lat, 
        bathy.values,
        norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=-5000, vmax=100),
        cmap='ocean'
    )
    plt.colorbar(bathy_mesh, label='Depth (m)')
    
    # Grid parameters
    xpc = 257634.73    # Origin x
    ypc = 3676471.67   # Origin y
    xlenc = 550000     # Length in x
    ylenc = 300000     # Length in y
    alpc = 30          # Rotation angle in degrees
    spacing = 50000    # Increased spacing for visualization
    
    # Calculate number of lines to draw
    nx = int(xlenc / spacing) + 1
    ny = int(ylenc / spacing) + 1
    
    # Calculate angle in radians
    angle_rad = np.radians(alpc)
    
    # Create vectors for the parallel lines
    dx = spacing * np.cos(angle_rad)
    dy = spacing * np.sin(angle_rad)
    dx_perp = -spacing * np.sin(angle_rad)
    dy_perp = spacing * np.cos(angle_rad)
    
    # Plot parallel lines
    for i in range(ny):
        start_x = xpc + i * dx_perp
        start_y = ypc + i * dy_perp
        end_x = start_x + xlenc * np.cos(angle_rad)
        end_y = start_y + xlenc * np.sin(angle_rad)
        ax.plot([start_x, end_x], [start_y, end_y], 'r-', alpha=0.5, linewidth=1)
    
    # Plot perpendicular lines
    for i in range(nx):
        start_x = xpc + i * dx
        start_y = ypc + i * dy
        end_x = start_x - ylenc * np.sin(angle_rad)
        end_y = start_y + ylenc * np.cos(angle_rad)
        ax.plot([start_x, end_x], [start_y, end_y], 'r-', alpha=0.5, linewidth=1)
    
    # Plot boundary
    corners = np.array([
        [xpc, ypc],  # Origin
        [xpc + xlenc * np.cos(angle_rad), ypc + xlenc * np.sin(angle_rad)],  # Right
        [xpc + xlenc * np.cos(angle_rad) - ylenc * np.sin(angle_rad), 
         ypc + xlenc * np.sin(angle_rad) + ylenc * np.cos(angle_rad)],  # Top-right
        [xpc - ylenc * np.sin(angle_rad), ypc + ylenc * np.cos(angle_rad)],  # Top-left
        [xpc, ypc]  # Close the polygon
    ])
    ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, label='Grid Boundary')
    
    # Plot origin point
    ax.scatter(xpc, ypc, c='red', s=100, marker='*', label='Origin')
    
    # Generate and plot some sample points
    points = []
    for i in range(0, ny, 2):  # Take every other line for visibility
        start_x = xpc + i * dx_perp
        start_y = ypc + i * dy_perp
        for j in range(0, nx, 2):  # Take every other point for visibility
            x = start_x + j * dx
            y = start_y + j * dy
            points.append([x, y])
    
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], c='yellow', s=10, alpha=0.5, label='Grid Points')
    
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('Bathymetry with Parallel Grid Overlay')
    ax.set_xlabel('UTM Easting (m)')
    ax.set_ylabel('UTM Northing (m)')
    
    plt.savefig('grid_check.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_grid_and_bathy() 