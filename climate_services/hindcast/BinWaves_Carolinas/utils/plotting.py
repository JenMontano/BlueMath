import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import wavespectra
import xarray as xr
from bluemath_tk.core.operations import get_uv_components
from bluemath_tk.core.plotting.colors import colormap_spectra
from matplotlib import colors
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import gc


def detect_coordinate_system(bathy):
    """
    Detect the coordinate system and extract coordinate variables from bathymetry data.
    
    Parameters
    ----------
    bathy : xr.DataArray or xr.Dataset
        Input bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)
    
    Returns
    -------
    dict
        Dictionary containing:
        - is_geographic: bool, whether the coordinates are geographic
        - x_coord: str, name of x coordinate
        - y_coord: str, name of y coordinate
        - proj: cartopy.crs projection or None
        - transform: cartopy.crs transform or None
    """
    import cartopy.crs as ccrs
    
    # Determine coordinate system based on coordinate names
    coord_names = list(bathy.coords)
    is_geographic = any(name in ['lon', 'longitude'] for name in coord_names) and \
                   any(name in ['lat', 'latitude'] for name in coord_names)
    
    # Get coordinate variables
    if is_geographic:
        x_coord = next(name for name in coord_names if name in ['lon', 'longitude'])
        y_coord = next(name for name in coord_names if name in ['lat', 'latitude'])
        proj = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()
    else:
        x_coord = next(name for name in coord_names if name in ['x', 'X', 'cx', 'easting'])
        y_coord = next(name for name in coord_names if name in ['y', 'Y', 'cy', 'northing'])
        proj = None
        transform = None
    
    return {
        'is_geographic': is_geographic,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'proj': proj,
        'transform': transform
    }

def plot_selected_bathy(bathy: xr.DataArray, utm_zone=None, buoys=None):
    """
    Plot bathymetry data in either UTM or geographic coordinates.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)
    utm_zone : int, optional
        UTM zone number if data is in UTM coordinates
    buoy_coords : dict, optional
        Dictionary of buoy names and their coordinates
        Example: {'NDBC-41001': (x1, y1), 'NDBC-41002': (x2, y2)}
    """
       
    # Get coordinate variables
    coords = detect_coordinate_system(bathy)
    is_geographic = coords['is_geographic']
    x_coord = coords['x_coord']
    y_coord = coords['y_coord']
    proj = coords['proj']
    transform = coords['transform']
    
    # If UTM and no zone provided, raise error
    if not is_geographic and utm_zone is None:
        raise ValueError("UTM zone must be provided for UTM coordinates")
    elif not is_geographic:
        proj = ccrs.UTM(zone=utm_zone)
        transform = proj

    # Create figure with proper projection
    fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={'projection': proj})

    # Plot bathymetry contours
    bathy.plot.contourf(
        x=x_coord,
        y=y_coord,
        ax=ax,
        levels=[10, 25, 50, 100, 200, 300, 500, 1000],
        cmap="Blues",
        transform=transform
     
    )
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20)
    
   # Plot buoys if provided
    if buoys is not None:
        # Separate coordinates and names
        x_buoys = [coord[0] for coord in buoys.values()]
        y_buoys = [coord[1] for coord in buoys.values()]
        
        # Plot all buoys
        ax.scatter(x_buoys, y_buoys, c="darkred", transform=transform, label='Available Buoys')
        
        for name, (x, y) in buoys.items():
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(3, 3),  # Small offset from point
                textcoords="offset points",
                fontsize=8,  # Smaller font size
                color="darkred",
                transform=transform,
                ha='left',  # Horizontal alignment
                va='bottom'  # Vertical alignment
            )
        
        # Add legend
        ax.legend(loc='upper right')  # Specify legend location
    
    # Set appropriate extent if geographic
    if is_geographic:
        x_min, x_max = bathy[x_coord].min().item(), bathy[x_coord].max().item()
        y_min, y_max = bathy[y_coord].min().item(), bathy[y_coord].max().item()
        ax.set_extent([x_min, x_max, y_min, y_max], crs=transform)
    
    ax.set_aspect('auto')
    plt.show()

def plot_bathy_swan_grid(bathy: xr.DataArray, fixed_params: dict, utm_zone=None, buoys=None):
    """
    Plot bathymetry with SWAN computational grid overlay.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)
    fixed_params : dict
        Dictionary containing grid parameters (xpc, ypc, alpc, xlenc, ylenc, mxc, myc)
    utm_zone : int, optional
        UTM zone number if data is in UTM coordinates
    buoys : dict, optional
        Dictionary of buoy names and their coordinates
        Example: {'NDBC-41001': (x1, y1), 'NDBC-41002': (x2, y2)}
    """
    # Get coordinate variables
    coords = detect_coordinate_system(bathy)
    is_geographic = coords['is_geographic']
    x_coord = coords['x_coord']
    y_coord = coords['y_coord']
    proj = coords['proj']
    transform = coords['transform']
    
    # If UTM and no zone provided, raise error
    if not is_geographic and utm_zone is None:
        raise ValueError("UTM zone must be provided for UTM coordinates")
    elif not is_geographic:
        proj = ccrs.UTM(zone=utm_zone)
        transform = proj

    # Create figure with proper projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})

    # Plot bathymetry contours with updated levels and colors
    levels = [10, 25, 50, 100, 200, 300, 500, 1000]
    bathy.plot.contourf(
        x=x_coord,
        y=y_coord,
        ax=ax,
        levels=levels,
        cmap="Blues",
        transform=transform,
        cbar_kwargs={
            'label': 'Elevation relative to sea\nlevel [m]',
            'orientation': 'vertical'
        }
    )
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=20)
    
    # Plot buoys if provided
    if buoys is not None:
        # Separate coordinates and names
        x_buoys = [coord[0] for coord in buoys.values()]
        y_buoys = [coord[1] for coord in buoys.values()]
        
        # Plot all buoys
        ax.scatter(x_buoys, y_buoys, c="darkred", transform=transform, label='Available Buoys')
        
        # Add annotations for each buoy
        for name, (x, y) in buoys.items():
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(3, 3),  # Small offset from point
                textcoords="offset points",
                fontsize=8,  # Smaller font size
                color="darkred",
                transform=transform,
                ha='left',  # Horizontal alignment
                va='bottom'  # Vertical alignment
            )
        
        # Add legend
        ax.legend(loc='upper right')
    
    # Extract grid parameters and convert if necessary
    xpc = fixed_params['xpc']
    ypc = fixed_params['ypc']
    alpc = fixed_params['alpc']
    xlenc = fixed_params['xlenc']
    ylenc = fixed_params['ylenc']
 
    # Calculate grid corners
    angle_rad = np.radians(alpc)
    corners = np.array([
        [xpc, ypc],  # Bottom left
        [xpc + xlenc * np.cos(angle_rad), ypc + xlenc * np.sin(angle_rad)],  # Bottom right
        [xpc + xlenc * np.cos(angle_rad) - ylenc * np.sin(angle_rad), 
         ypc + xlenc * np.sin(angle_rad) + ylenc * np.cos(angle_rad)],  # Top right
        [xpc - ylenc * np.sin(angle_rad), ypc + ylenc * np.cos(angle_rad)],  # Top left
        [xpc, ypc]  # Close the polygon
    ])
    
    # Create and add grid polygon
    polygon = plt.Polygon(corners, facecolor='gray', alpha=0.2, edgecolor='black', 
                         linewidth=1.5, zorder=15, transform=transform)
    ax.add_patch(polygon)
    
    # Add 'B' label in the middle of the grid
    grid_center_x = np.mean([corners[0, 0], corners[1, 0], corners[2, 0], corners[3, 0]])
    grid_center_y = np.mean([corners[0, 1], corners[1, 1], corners[2, 1], corners[3, 1]])
    ax.text(grid_center_x, grid_center_y, 'B', fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7), zorder=25,
            transform=transform)
    
    # Set appropriate extent
    x_min, x_max = bathy[x_coord].min().item(), bathy[x_coord].max().item()
    y_min, y_max = bathy[y_coord].min().item(), bathy[y_coord].max().item()
    ax.set_extent([x_min, x_max, y_min, y_max], crs=transform)
    
    ax.set_aspect('auto')
    plt.show()

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


def crop_bathy_swan_grid(bathy, grid_params, utm_zone=None, buffer_distance=10000):
    """
    Crop bathymetry to cover a rotated grid area with buffer.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Input bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)
    grid_params : dict
        Dictionary containing:
        - xpc: x-coordinate of origin (UTM or geographic)
        - ypc: y-coordinate of origin (UTM or geographic)
        - alpc: rotation angle in degrees
        - xlenc: grid length in x (meters or degrees)
        - ylenc: grid length in y (meters or degrees)
    utm_zone : int, optional
        UTM zone number if data is in UTM coordinates
    buffer_distance : float
        Buffer distance in meters (for UTM) or degrees (for geographic)
        For geographic coordinates, typical values might be 0.1-0.5 degrees
    
    Returns
    -------
    xr.DataArray
        Cropped bathymetry
    """
    # Use the utility function
    coords = detect_coordinate_system(bathy)
    is_geographic = coords['is_geographic']
    x_coord = coords['x_coord']
    y_coord = coords['y_coord']
    
    # If UTM and no zone provided, raise error
    if not is_geographic and utm_zone is None:
        raise ValueError("UTM zone must be provided for UTM coordinates")

    # Extract grid parameters
    xpc = grid_params['xpc']
    ypc = grid_params['ypc']
    alpc = grid_params['alpc']
    xlenc = grid_params['xlenc']
    ylenc = grid_params['ylenc']
    
    # Get grid corners
    corner = (xpc, ypc)
    x_coords, y_coords = get_rotated_grid_corners(
        corner, 
        alpc, 
        xlenc, 
        ylenc
    )
    
    # Get bounds with buffer
    x_min = np.min(x_coords) - buffer_distance
    x_max = np.max(x_coords) + buffer_distance
    y_min = np.min(y_coords) - buffer_distance
    y_max = np.max(y_coords) + buffer_distance
    
    print(f"Cropping bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # Crop bathymetry
    cropped = bathy.sel(
        {x_coord: slice(x_min, x_max),
         y_coord: slice(y_max, y_min)}  # Note: slice from max to min for descending coordinates
    )
    
    return cropped

def plot_cases_grid(
    data: xr.DataArray,
    cases_to_plot: list = [0, 320, 615],
    colors_to_plot: list = ["green", "orange", "purple"],
    num_directions: int = 24,
    num_frequencies: int = 29,
):
    # Plot all cases in a grid
    fig, axes = plt.subplots(
        ncols=num_frequencies, nrows=num_directions, figsize=(29, 15)
    )
    for i, ax in enumerate(axes.flat):
        try:
            ax.pcolor(
                (
                    data.sel(case_num=i)
                    .isel(Xp=slice(None, None, 3), Yp=slice(None, None, 3))
                    .values
                ),
                cmap="RdBu_r",
                vmin=0,
                vmax=2,
            )
        except Exception as e:
            print(e)
    for i, ax in enumerate(axes.flat):
        ax.set_aspect("equal")
        ax.set_title("")
        ax.axis("off")
    fig.tight_layout()
    # Set texts in left part of grid and top part
    fig.text(
        0, 0.5, "Directions", ha="center", va="center", rotation="vertical", fontsize=20
    )
    fig.text(0.5, 1, "Frequencies", ha="center", va="center", fontsize=20)
    # Plot selected cases in a grid
    fig_sel, axes_sel = plt.subplots(
        ncols=len(cases_to_plot), nrows=1, figsize=(5 * len(cases_to_plot), 4)
    )
    for ax, ax_sel, case_to_plot, color_to_plot in zip(
        axes.flat[cases_to_plot], axes_sel.flat, cases_to_plot, colors_to_plot
    ):
        try:
            data.sel(case_num=case_to_plot).plot(
                ax=ax_sel,
                cmap="RdBu_r",
                vmin=0,
                vmax=2,
                add_colorbar=True,
                cbar_kwargs={"orientation": "horizontal", "shrink": 0.8},
            )
            ax_sel.set_aspect("equal")
            ax_sel.set_title("")
            # Remove ticks and labels
            ax_sel.set_xticks([])
            ax_sel.set_yticks([])
            ax_sel.set_xticklabels([])
            ax_sel.set_yticklabels([])
            ax_sel.set_xlabel("")
            ax_sel.set_ylabel("")
            # Set axis of color to indicate it is plotted
            ax_sel.spines["top"].set_color(color_to_plot)
            ax_sel.spines["top"].set_linewidth(2)
            ax_sel.spines["right"].set_color(color_to_plot)
            ax_sel.spines["right"].set_linewidth(2)
            ax_sel.spines["bottom"].set_color(color_to_plot)
            ax_sel.spines["bottom"].set_linewidth(2)
            ax_sel.spines["left"].set_color(color_to_plot)
            ax_sel.spines["left"].set_linewidth(2)
            # Set axis of color to indicate it is plotted
            ax.axis("on")
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Set axis of color to indicate it is plotted
            ax.spines["top"].set_color(color_to_plot)
            ax.spines["top"].set_linewidth(2)
            ax.spines["right"].set_color(color_to_plot)
            ax.spines["right"].set_linewidth(2)
            ax.spines["bottom"].set_color(color_to_plot)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_color(color_to_plot)
            ax.spines["left"].set_linewidth(2)
        except Exception as e:
            print(e)
    fig_sel.tight_layout()


def plot_case_variables(data: xr.Dataset):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    data["Hsig"].plot(
        ax=axes[0],
        cbar_kwargs={"label": "Hsig [m]", "orientation": "horizontal", "shrink": 0.7},
        cmap="RdBu_r",
        vmin=0,
        vmax=2,
        
    )
    data["Tm02"].plot(
        ax=axes[1],
        cbar_kwargs={"label": "Tm02 [s]", "orientation": "horizontal", "shrink": 0.7},
        cmap="magma",
        vmin=0,
        vmax=20,
    )
    data["Dir"].plot(
        ax=axes[2],
        cbar_kwargs={"label": "Dir [deg]", "orientation": "horizontal", "shrink": 0.7},
        cmap="twilight",
        vmin=0,
        vmax=360,
    )

    dir_u, dir_v = get_uv_components(data["Dir"].values)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        step = 3
        ax.quiver(
            data["Xp"][::step],
            data["Yp"][::step],
            -dir_u[::step, ::step],
            -dir_v[::step, ::step],
            color="grey",
            scale=25,
        )

    fig.tight_layout()
   


def create_text_with_metrics(array1: np.ndarray, array2: np.ndarray):
    """
    Create a text with metrics comparing two arrays.
    """

    # Calculate metrics
    mae = np.mean(np.abs(array1 - array2))
    rmse = np.sqrt(np.mean((array1 - array2) ** 2))
    r2 = np.corrcoef(array1, array2)[0, 1] ** 2

    # Create text
    text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"

    return text


def plot_wave_series(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    offshore_data: wavespectra.SpecArray,
    times: np.ndarray,
):
    buoy_color = "lightcoral"
    binwaves_color = "royalblue"
    offshore_color = "gold"

    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    buoy_data["Hs_Buoy"].plot(ax=axes[0], label="Buoy", c=buoy_color, alpha=0.8, lw=1)
    buoy_data["Tp_Buoy"].plot(ax=axes[1], label="Buoy", c=buoy_color, alpha=0.8, lw=1)
    axes[2].scatter(
        times,
        buoy_data["Dir_Buoy"].values,
        c=buoy_color,
        label="Buoy",
        alpha=0.8,
        s=1,
    )
    binwaves_data.hs().plot(
        ax=axes[0], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    binwaves_data.tp().plot(
        ax=axes[1], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    axes[2].scatter(
        times,
        binwaves_data.dpm().values,
        c=binwaves_color,
        label="BinWaves",
        alpha=0.8,
        s=1,
    )
    offshore_data.hs().plot(
        ax=axes[0], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    )
    offshore_data.tp().plot(
        ax=axes[1], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    )
    axes[2].scatter(
        times,
        offshore_data.dpm().values,
        c=offshore_color,
        label="Offshore",
        alpha=0.8,
        s=1,
    )

    # Set labels
    axes[0].set_ylabel("Hs [m]")
    axes[0].legend()
    axes[1].set_ylabel("T [s] - tp")
    axes[2].set_ylabel("Dir [°] - dm")
    for ax in axes:
        ax.set_title("")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hs = np.vstack([buoy_data["Hs_Buoy"].values, binwaves_data.hs().values])
    hs = gaussian_kde(hs)(hs)
    axes[0].scatter(
        buoy_data["Hs_Buoy"].values,
        binwaves_data.hs().values,
        s=1,
        c=hs,
        cmap="turbo",
    )
    axes[0].text(
        5,
        0.5,
        create_text_with_metrics(
            buoy_data["Hs_Buoy"].values, binwaves_data.hs().values
        ),
        color="darkred",
    )
    axes[0].plot([0, 7], [0, 7], c="darkred", linestyle="--")
    axes[0].set_xlabel("Hs - Buoy [m]")
    axes[0].set_ylabel("Hs - BinWaves [m]")
    axes[0].set_xlim([0, 7])
    axes[0].set_ylim([0, 7])
    tp = np.vstack([buoy_data["Tp_Buoy"].values, binwaves_data.tp().values])
    tp = gaussian_kde(tp)(tp)
    axes[1].scatter(
        buoy_data["Tp_Buoy"].values,
        binwaves_data.tp().values,
        s=1,
        c=tp,
        cmap="turbo",
        label="Tp",
    )
    axes[1].text(
        15,
        1.25,
        create_text_with_metrics(
            buoy_data["Tp_Buoy"].values, binwaves_data.tp().values
        ),
        color="darkred",
    )
    axes[1].plot([0, 20], [0, 20], c="darkred", linestyle="--")
    axes[1].set_xlabel("Tp - Buoy [s]")
    axes[1].set_ylabel("Tp - BinWaves [s]")
    axes[1].set_xlim([0, 20])
    axes[1].set_ylim([0, 20])
    dpm = np.vstack([buoy_data["Dir_Buoy"].values, binwaves_data.dpm().values])
    dpm = gaussian_kde(dpm)(dpm)
    axes[2].scatter(
        buoy_data["Dir_Buoy"].values,
        binwaves_data.dpm().values,
        s=1,
        c=dpm,
        cmap="turbo",
        label="Dpm",
    )
    axes[2].text(
        250,
        25,
        create_text_with_metrics(
            buoy_data["Dir_Buoy"].values, binwaves_data.dpm().values
        ),
        color="darkred",
    )
    axes[2].plot([0, 360], [0, 360], c="darkred", linestyle="--")
    axes[2].set_xlabel("Dir - Buoy [°]")
    axes[2].set_ylabel("Dir - BinWaves [°]")
    axes[2].set_xlim([0, 360])
    axes[2].set_ylim([0, 360])

    for ax in axes:
        ax.set_aspect("equal")
        # Delete top and right axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Set axis color and ticks to darkred
        ax.spines["left"].set_color("darkred")
        ax.spines["bottom"].set_color("darkred")
        ax.yaxis.label.set_color("darkred")
        ax.xaxis.label.set_color("darkred")
        ax.tick_params(axis="x", colors="darkred")
        ax.tick_params(axis="y", colors="darkred")

    return fig, axes


def create_white_zero_colormap(cmap_name="Spectral"):
    """
    Create a colormap with white at zero, and selected colormap for positive values
    """

    # Get the base colormap
    base_cmap = plt.cm.get_cmap(cmap_name)

    # Create a colormap list with white at the beginning
    colors_list = [(0.0, (1.0, 1.0, 1.0, 1.0))]  # White at the start for zero

    # Add colors from the original colormap for positive values
    for i in np.linspace(0, 1, 100):
        colors_list.append((0.01 + 0.99 * i, base_cmap(i)))

    # Create the custom colormap
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        f"white_{cmap_name}", colors_list
    )

    return custom_cmap


def create_custom_bathy_cmap():
    # Define your colors
    custom_colors = [
        "#4a84b5",
        "#5493c8",
        "#5fa9d1",
        "#74c3dc",
        "#8ed7e8",
        "#a0e2ef",
        "#b7f1eb",
        "#c8ebd8",
        "#d7e8c3",
        "#e2e5a5",
        "#f4cda0",
        "#f1e2c6",
    ]
    # Create the custom colormap
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        "custom_bathy_cmap", custom_colors
    )
    return custom_cmap


def plot_spectrum_in_coastline(
    bathy: xr.DataArray,
    reconstructed_onshore_spectra: xr.Dataset,
    reconstruction_kps: xr.Dataset, #TODO: This is not used
    offshore_spectra: xr.Dataset,
    time_to_plot: str,
    sites_for_spectrum: list,
):
    """
    Plot gridded graph with wave spectra visualization.
    Handles both geographic (lat/lon) and Cartesian (UTM) coordinates.
    """
    try:
        # Print shapes for debugging
    
        # list(reconstruction_kps.coords)
        time_slice = reconstructed_onshore_spectra.sel(time=time_to_plot, method="nearest")

        # Use the utility function
        coords = detect_coordinate_system(bathy)
        is_geographic = coords['is_geographic']
        x_coord = coords['x_coord']
        y_coord = coords['y_coord']
        proj = coords['proj']
        transform = coords['transform']

        # Create figure with proper projection if geographic
        if is_geographic:
            fig, ax = plt.subplots(figsize=(15, 6), subplot_kw={'projection': proj})
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
        else:
            fig, ax = plt.subplots(figsize=(15, 6))

        # Plot bathymetry as a contour
        plot_kwargs = {
            'ax': ax,
            'levels': [0, -10, -25, -50, -100, -200, -500, -1000],
            'cmap': "Blues_r",
            'add_colorbar': False,
        }
        if is_geographic:
            plot_kwargs.update({
                'x': x_coord,
                'y': y_coord,
                'transform': transform
            })
        bathy.plot.contourf(**plot_kwargs)
        #TODO: This should take lat lon coordinates or utm in future codes
        # Get x and y coordinates for scatter plot
        if is_geographic:
            # Use reconstructed_onshore_spectra coordinates since they match the sites
            x_vals = reconstructed_onshore_spectra.utm_x.values
            y_vals = reconstructed_onshore_spectra.utm_y.values
        else:
            x_vals = reconstructed_onshore_spectra.utm_x.values
            y_vals = reconstructed_onshore_spectra.utm_y.values
            
        # Calculate Hs directly from the time slice
        hs_values = time_slice.kp.spec.hs().values

        
        scatter_kwargs = {
            'c': hs_values,
            'cmap': colormap_spectra(),
            's': 20, 
        }
        if is_geographic:
            scatter_kwargs['transform'] = transform
            
        phs = ax.scatter(x_vals, y_vals, **scatter_kwargs)
        plt.colorbar(phs).set_label("Hs [m]")

        #TODO: This should take lat lon coordinates or utm in future codes
        # Plot onshore spectra at sites
        for site in sites_for_spectrum:
            try:
                # Get site coordinates
                if is_geographic:
                    lon = reconstructed_onshore_spectra.utm_x.values[site]
                    lat = reconstructed_onshore_spectra.utm_y.values[site]
                else:
                    lon = reconstructed_onshore_spectra.utm_x.values[site]
                    lat = reconstructed_onshore_spectra.utm_y.values[site]
                
                # Create inset with explicit size relative to data coordinates
                inset_width = (bathy[x_coord].max() - bathy[x_coord].min()) * 0.1  # 10% of plot width
                axin = ax.inset_axes(
                    [lon - inset_width/2, lat - inset_width/2, inset_width, inset_width], 
                    transform=ax.transData, 
                    projection="polar"
                )
                
                # Mark the site location
                site_scatter_kwargs = {'c': "black", 'marker': "*", 's': 100}
                if is_geographic:
                    site_scatter_kwargs['transform'] = transform
                ax.scatter(lon, lat, **site_scatter_kwargs)
                
                # Get and plot the spectrum
                spectrum = time_slice.isel(site=site).kp
                if not np.all(np.isnan(spectrum)):
                    pcm = axin.pcolormesh(
                        np.deg2rad(reconstructed_onshore_spectra.dir.values),
                        reconstructed_onshore_spectra.freq.values,
                        np.sqrt(spectrum),
                        cmap=colormap_spectra(),
                    )
                    axin.set_theta_zero_location("N", offset=0)
                    axin.set_theta_direction(-1)
                    axin.axis("off")
                else:
                    print(f"Warning: NaN values found in spectrum for site {site}")
            except Exception as e:
                print(f"Error plotting site {site}: {str(e)}")
                continue

        # Set reasonable axis limits based on bathymetry extent
        if is_geographic:
            ax.set_extent([
                float(bathy[x_coord].min()), 
                float(bathy[x_coord].max()), 
                float(bathy[y_coord].min()), 
                float(bathy[y_coord].max())
            ], crs=transform)
            ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        else:
            ax.set_xlim([bathy[x_coord].min(), bathy[x_coord].max()])
            ax.set_ylim([bathy[y_coord].min(), bathy[y_coord].max()])
            ax.grid(True, linestyle='--', alpha=0.5)

        
        return fig, ax
    except Exception as e:
        print(f"Error in plot_spectrum_in_coastline: {str(e)}")
        raise