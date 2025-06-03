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
import xarray as xr
import time


# def plot_selected_bathy(bathy: xr.DataArray):
#     # Plot ortophoto of Cantabria
#     ortophoto = mimg.imread("outputs/ortophoto_cantabria.png")
#     image_bounds = (
#         410000.0,
#         479197.6875,
#         4802379.0,
#         4837093.5,
#     )
#     fig, ax = plt.subplots(figsize=(12, 5))
#     bathy.plot.contourf(
#         ax=ax, levels=[10, 25, 50, 100, 200, 300, 500, 1000], cmap="Blues"
#     )
#     grid = ax.pcolor(
#         bathy.lon.values[:-1],
#         bathy.lat.values[1:],
#         bathy.values[:-1, 1:],
#         edgecolors="black",  # Color of the grid lines
#         linewidth=0.5,  # Thickness of the grid lines
#         facecolors="none",  # No face filling
#         alpha=0.5,  # Transparency of the grid lines
#     )
#     grid.set_edgecolor("black")  # Ensure edges are black
#     ax.scatter(428845.10, 4815606.89, c="darkred")
#     ax.annotate(
#         "Validation Buoy",
#         xy=(428845.10, 4815606.89),  # Arrow tip
#         xytext=(428845.10 - 1000, 4815606.89 + 5000),  # Text position
#         arrowprops=dict(color="darkred", arrowstyle="->"),
#         fontsize=10,
#         color="darkred",
#     )
#     ax.imshow(
#         ortophoto,
#         extent=image_bounds,
#         zorder=10,
#     )
#     rect = Rectangle(
#         (bathy.lon.min(), bathy.lat.min()),  # Bottom-left corner
#         bathy.lon.max() - bathy.lon.min(),  # Width
#         bathy.lat.max() - bathy.lat.min(),  # Height
#         linewidth=3,
#         edgecolor="orange",
#         facecolor="none",
#         zorder=15,
#     )
#     ax.annotate(
#         "Bathymetry Area",
#         xy=(bathy.lon.max(), bathy.lat.max() - 5000),  # Arrow tip
#         xytext=(bathy.lon.max() + 2500, bathy.lat.max() - 10000),  # Text position
#         arrowprops=dict(color="orange", arrowstyle="->"),
#         fontsize=10,
#         color="orange",
#     )
#     ax.add_patch(rect)
#     ax.axis(image_bounds)
#     ax.set_aspect("equal")


def plot_selected_bathy(bathy: xr.DataArray, utm_zone=18):
    # Set up the projection (change zone if needed)
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={'projection': proj})

    # Plot bathymetry
    bathy.plot.contourf(
        ax=ax,
        x='lon',
        y='lat',
        levels=[0, 10, 25, 50, 100, 200, 500, 1000],
        cmap="Blues_r",
        add_colorbar=False,
        transform=proj
    )
    grid = ax.pcolor(
        bathy.lon.values[:-1],
        bathy.lat.values[1:],
        bathy.values[:-1, 1:],
        edgecolors="black",
        linewidth=0.5,
        facecolors="none",
        alpha=0.5,
        transform=proj
    )
    grid.set_edgecolor("black")

    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20)

    # Rectangle and annotation
    # rect = Rectangle(
    #     (bathy.lon.min(), bathy.lat.min()),
    #     bathy.lon.max() - bathy.lon.min(),
    #     bathy.lat.max() - bathy.lat.min(),
    #     linewidth=3,
    #     edgecolor="orange",
    #     facecolor="none",
    #     zorder=15,
    #     transform=proj
    # )
    # ax.add_patch(rect)
    # ax.annotate(
    #     "Bathymetry Area",
    #     xy=(bathy.lon.max(), bathy.lat.max() - 5000),
    #     xytext=(bathy.lon.max() + 2500, bathy.lat.max() - 10000),
    #     arrowprops=dict(color="orange", arrowstyle="->"),
    #     fontsize=10,
    #     color="orange",
    #     transform=proj
    # )
    ax.set_aspect("equal")

    # Optional: set extent if you want to zoom in
    # ax.set_extent([bathy.lon.min(), bathy.lon.max(), bathy.lat.min(), bathy.lat.max()], crs=proj)

    plt.show()

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
    # Remove NaN values before calculating metrics
    mask = np.isfinite(array1) & np.isfinite(array2)
    array1_clean = array1[mask]
    array2_clean = array2[mask]

    if len(array1_clean) < 2:  # Need at least 2 points for statistics
        return "MAE: nan\nRMSE: nan\nR²: nan"

    # Calculate metrics
    mae = np.mean(np.abs(array1_clean - array2_clean))
    rmse = np.sqrt(np.mean((array1_clean - array2_clean) ** 2))
    r2 = np.corrcoef(array1_clean, array2_clean)[0, 1] ** 2

    # Create text
    text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"

    return text


def clean_wave_data(buoy_data: wavespectra.SpecArray, binwaves_data: wavespectra.SpecArray, offshore_data: wavespectra.SpecArray):
    """
    Clean wave data by handling NaN and infinite values before plotting.
    
    Parameters
    ----------
    buoy_data : wavespectra.SpecArray
        Buoy wave data
    binwaves_data : wavespectra.SpecArray
        BinWaves reconstructed data
    offshore_data : wavespectra.SpecArray
        Offshore wave data
        
    Returns
    -------
    tuple
        Cleaned versions of (buoy_data, binwaves_data, offshore_data)
    """
    # Make copies to avoid modifying original data
    buoy_clean = buoy_data.copy()
    binwaves_clean = binwaves_data.copy()
    offshore_clean = offshore_data.copy()
    
    # Clean Hs data
    if 'Hs_Buoy' in buoy_clean:
        hs_mean = np.nanmean(buoy_clean['Hs_Buoy'].values)
        buoy_clean['Hs_Buoy'] = buoy_clean['Hs_Buoy'].fillna(hs_mean)
    
    # Clean Tp data
    if 'Tp_Buoy' in buoy_clean:
        tp_mean = np.nanmean(buoy_clean['Tp_Buoy'].values)
        buoy_clean['Tp_Buoy'] = buoy_clean['Tp_Buoy'].fillna(tp_mean)
    
    # Clean Dir data
    if 'Dir_Buoy' in buoy_clean:
        dir_mean = np.nanmean(buoy_clean['Dir_Buoy'].values)
        buoy_clean['Dir_Buoy'] = buoy_clean['Dir_Buoy'].fillna(dir_mean)
    
    # Handle infinite values in buoy data
    for var in ['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy']:
        if var in buoy_clean:
            buoy_clean[var] = np.nan_to_num(buoy_clean[var].values, 
                                          posinf=np.nanmean(buoy_clean[var].values),
                                          neginf=np.nanmean(buoy_clean[var].values))
    
    # Clean BinWaves data
    try:
        # Get the wave parameters
        hs_vals = binwaves_clean.spec.hs().values
        tp_vals = binwaves_clean.spec.tp().values
        dpm_vals = binwaves_clean.spec.dpm().values
        
        # Replace non-finite values with means
        hs_vals[~np.isfinite(hs_vals)] = np.nanmean(hs_vals)
        tp_vals[~np.isfinite(tp_vals)] = np.nanmean(tp_vals)
        dpm_vals[~np.isfinite(dpm_vals)] = np.nanmean(dpm_vals)
        
        # Update the values in the dataset
        binwaves_clean.spec.hs().values[:] = hs_vals
        binwaves_clean.spec.tp().values[:] = tp_vals
        binwaves_clean.spec.dpm().values[:] = dpm_vals
    except AttributeError as e:
        print(f"Warning: Could not clean BinWaves data: {str(e)}")
    
    # Clean Offshore data
    try:
        # Get the wave parameters
        hs_vals = offshore_clean.spec.hs().values
        tp_vals = offshore_clean.spec.tp().values
        dpm_vals = offshore_clean.spec.dpm().values
        
        # Replace non-finite values with means
        hs_vals[~np.isfinite(hs_vals)] = np.nanmean(hs_vals)
        tp_vals[~np.isfinite(tp_vals)] = np.nanmean(tp_vals)
        dpm_vals[~np.isfinite(dpm_vals)] = np.nanmean(dpm_vals)
        
        # Update the values in the dataset
        offshore_clean.spec.hs().values[:] = hs_vals
        offshore_clean.spec.tp().values[:] = tp_vals
        offshore_clean.spec.dpm().values[:] = dpm_vals
    except AttributeError as e:
        print(f"Warning: Could not clean Offshore data: {str(e)}")
    
    return buoy_clean, binwaves_clean, offshore_clean


def safe_kde(data):
    """
    Safely compute KDE for stacked data, handling NaN and Inf values.
    
    Parameters
    ----------
    data : np.ndarray
        Stacked array of shape (2, N) containing the two variables to compare
        
    Returns
    -------
    np.ndarray
        KDE values or None if computation fails
    """
    try:
        # Remove any rows with NaN or Inf
        mask = np.isfinite(data).all(axis=0)
        clean_data = data[:, mask]
        
        if clean_data.shape[1] < 2:  # Need at least 2 points for KDE
            return None
            
        # Compute KDE on clean data and evaluate on original points
        kde = gaussian_kde(clean_data)
        return kde(data)
    except Exception as e:
        print(f"KDE calculation failed: {str(e)}")
        return None

def plot_wave_series(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    offshore_data: wavespectra.SpecArray,
    times: np.ndarray,
    save_dir: str = None,
    buoyId: int = None,
):
    buoy_color = "lightcoral"
    binwaves_color = "royalblue"
    offshore_color = "gold"

    # First figure - Time series
    fig1, axes = plt.subplots(3, 1, figsize=(20, 10))
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
    binwaves_data.spec.hs().plot(
        ax=axes[0], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    binwaves_data.spec.tp().plot(
        ax=axes[1], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    axes[2].scatter(
        times,
        binwaves_data.spec.dpm().values,
        c=binwaves_color,
        label="BinWaves",
        alpha=0.8,
        s=1,
    )
    offshore_data.spec.hs().plot(
        ax=axes[0], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    )
    offshore_data.spec.tp().plot(
        ax=axes[1], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    )
    axes[2].scatter(
        times,
        offshore_data.spec.dpm().values,
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

    # Second figure - Scatter plots
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    hs = np.vstack([buoy_data["Hs_Buoy"].values, binwaves_data.spec.hs().values])
    hs = gaussian_kde(hs)(hs)
    axes[0].scatter(
        buoy_data["Hs_Buoy"].values,
        binwaves_data.spec.hs().values,
        s=1,
        c=hs,
        cmap="turbo",
    )
    axes[0].text(
        5,
        0.5,
        create_text_with_metrics(
            buoy_data["Hs_Buoy"].values, binwaves_data.spec.hs().values
        ),
        color="darkred",
    )
    axes[0].plot([0, 7], [0, 7], c="darkred", linestyle="--")
    axes[0].set_xlabel("Hs - Buoy [m]")
    axes[0].set_ylabel("Hs - BinWaves [m]")
    axes[0].set_xlim([0, 7])
    axes[0].set_ylim([0, 7])
    tp = np.vstack([buoy_data["Tp_Buoy"].values, binwaves_data.spec.tp().values])
    tp = gaussian_kde(tp)(tp)
    axes[1].scatter(
        buoy_data["Tp_Buoy"].values,
        binwaves_data.spec.tp().values,
        s=1,
        c=tp,
        cmap="turbo",
        label="Tp",
    )
    axes[1].text(
        15,
        1.25,
        create_text_with_metrics(
            buoy_data["Tp_Buoy"].values, binwaves_data.spec.tp().values
        ),
        color="darkred",
    )
    axes[1].plot([0, 20], [0, 20], c="darkred", linestyle="--")
    axes[1].set_xlabel("Tp - Buoy [s]")
    axes[1].set_ylabel("Tp - BinWaves [s]")
    axes[1].set_xlim([0, 20])
    axes[1].set_ylim([0, 20])
    dpm = np.vstack([buoy_data["Dir_Buoy"].values, binwaves_data.spec.dpm().values])
    dpm = gaussian_kde(dpm)(dpm)
    axes[2].scatter(
        buoy_data["Dir_Buoy"].values,
        binwaves_data.spec.dpm().values,
        s=1,
        c=dpm,
        cmap="turbo",
        label="Dpm",
    )
    axes[2].text(
        250,
        25,
        create_text_with_metrics(
            buoy_data["Dir_Buoy"].values, binwaves_data.spec.dpm().values
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

    # Save figures if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f'timeseries_buoy_{buoyId}_validation_sat.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, f'scatter_buoy_{buoyId}_validation_sat.png'), dpi=300, bbox_inches='tight')

    return fig1, fig2, axes


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
    reconstruction_kps: xr.Dataset,
    offshore_spectra: xr.Dataset,
    time_to_plot: str,
    sites_for_spectrum: list = None,
):
    """
    Plot gridded graph with wave spectra at different locations.
    
    All coordinates are in UTM Zone 18N (EPSG:32618):
    - bathy uses 'cx' and 'cy' for UTM coordinates
    - reconstructed_onshore_spectra uses 'utm_x' and 'utm_y' for UTM coordinates
    - reconstruction_kps uses 'utm_x' and 'utm_y' for UTM coordinates
    - offshore_spectra uses 'longitude' and 'latitude' for UTM coordinates (despite the names)
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    reconstructed_onshore_spectra : xr.Dataset
        Reconstructed onshore spectra
    reconstruction_kps : xr.Dataset
        Reconstruction key points
    offshore_spectra : xr.Dataset
        Offshore spectra
    time_to_plot : str
        Time to plot in ISO format (e.g., '2009-01-01T00:00:00')
    sites_for_spectrum : list, optional
        List of site indices to plot spectra for
    """

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot bathymetry as a countour (using UTM coordinates cx, cy)
    bathy.plot.contourf(
        ax=ax,
        x='lon',
        y='lat',
        levels=[0, -10, -25, -50, -100, -200, -500, -1000],
        cmap="Blues_r",
        add_colorbar=False,
    )

    # Find the closest time index for all datasets
    time_to_plot_dt = np.datetime64(time_to_plot)
    reconstructed_time_idx = np.abs(reconstructed_onshore_spectra.time.values - time_to_plot_dt).argmin()
    offshore_time_idx = np.abs(offshore_spectra.time.values - time_to_plot_dt).argmin()
    
    # Get Hs for the specific time
    hs_map = (
        reconstructed_onshore_spectra.isel(time=reconstructed_time_idx)
        .kp.spec.hs()
        .values
    )
    
    # Plot Hs in grid (using UTM coordinates utm_x, utm_y)
    phs = ax.scatter(
        reconstruction_kps.utm_x.values,
        reconstruction_kps.utm_y.values,
        c=hs_map,
        cmap=colormap_spectra(),
    )
    plt.colorbar(phs).set_label("Hs [m]")

    # Plot onshore spectra at sites (using UTM coordinates utm_x, utm_y)
    if sites_for_spectrum is not None:
        for site in sites_for_spectrum:
            lon = reconstructed_onshore_spectra.utm_x.values[site]
            lat = reconstructed_onshore_spectra.utm_y.values[site]
            axin = ax.inset_axes(
                [lon, lat, 55000, 55000], transform=ax.transData, projection="polar"
            )
            ax.scatter(lon, lat, c="black", marker="*", s=500)
            axin.pcolormesh(
                np.deg2rad(reconstructed_onshore_spectra.dir.values),
                reconstructed_onshore_spectra.freq.values,
                np.sqrt(
                    reconstructed_onshore_spectra.isel(time=reconstructed_time_idx)
                    .isel(site=site)
                    .kp
                ),
                cmap=colormap_spectra(),
            )
            axin.set_theta_zero_location("N", offset=0)
            axin.set_theta_direction(-1)
            axin.axis("off")

    # Plot offshore spectrum at the actual spectral point location
    # Note: offshore_spectra uses 'longitude' and 'latitude' names but they are actually UTM coordinates
    offshore_x = float(np.unique(offshore_spectra.longitude.values)[0])
    offshore_y = float(np.unique(offshore_spectra.latitude.values)[0])
    axoff = ax.inset_axes(
        [offshore_x, offshore_y, 55000, 55000], 
        transform=ax.transData, 
        projection="polar"
    )
    
    axoff.pcolormesh(
        np.deg2rad(offshore_spectra.dir.values),
        offshore_spectra.freq.values,
        np.sqrt(offshore_spectra.isel(time=offshore_time_idx).efth),
        cmap=colormap_spectra(),
    )
    axoff.set_theta_zero_location("N", offset=0)
    axoff.set_theta_direction(-1)
    axoff.axis("off")

    return fig, ax


def plot_bathy_swan_grid(bathy: xr.DataArray, fixed_params: dict, utm_zone=18, skip_lines=20, plot_all_lines=False, points=None, point_labels=None):
    """
    Plot bathymetry with SWAN computational grid overlay.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    fixed_params : dict
        Dictionary containing grid parameters (xpc, ypc, alpc, xlenc, ylenc, mxc, myc)
    utm_zone : int
        UTM zone number
    skip_lines : int
        Plot every nth line if plot_all_lines is False
    plot_all_lines : bool
        If True, plots all grid lines (warning: may be very dense)
    points : list of tuples, optional
        List of (x, y) coordinates to plot as points
    point_labels : list of str, optional
        Labels for the points to show in legend
    """
    # Set up the projection
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})
    
    # Plot bathymetry
    bathy.plot.contourf(
        ax=ax,
        x='lon',
        y='lat',
        levels=[0,10, 25, 50, 100, 200, 500, 1000],
        cmap="Blues_r",
        add_colorbar=True,
        transform=proj
    )
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20)
    
    # Extract grid parameters
    xpc = fixed_params['xpc']
    ypc = fixed_params['ypc']
    alpc = fixed_params['alpc']
    xlenc = fixed_params['xlenc']
    ylenc = fixed_params['ylenc']
    mxc = fixed_params['mxc']
    myc = fixed_params['myc']
    
    # Convert angle to radians
    angle_rad = np.radians(alpc)
    
    # Calculate actual grid spacing (500m x 500m)
    dx = xlenc / mxc  # Should be 500m
    dy = ylenc / myc  # Should be 500m
    
    # Calculate rotated vectors
    dx_rot = dx * np.cos(angle_rad)
    dy_rot = dx * np.sin(angle_rad)
    dx_perp = -dy * np.sin(angle_rad)
    dy_perp = dy * np.cos(angle_rad)
    
    # Determine line plotting interval
    skip = 1 if plot_all_lines else skip_lines
    
    # Define grid line style
    grid_color = '0.7'  # Light gray in matplotlib notation (0.7 means 70% white)
    grid_alpha = 0.4
    
    # Plot x-direction lines
    for i in range(0, myc + 1, skip):
        start_x = xpc + i * dx_perp
        start_y = ypc + i * dy_perp
        end_x = start_x + xlenc * np.cos(angle_rad)
        end_y = start_y + xlenc * np.sin(angle_rad)
        ax.plot([start_x, end_x], [start_y, end_y], 
                color=grid_color, alpha=grid_alpha, linewidth=0.5, transform=proj)
    
    # Plot y-direction lines
    for i in range(0, mxc + 1, skip):
        start_x = xpc + i * dx_rot
        start_y = ypc + i * dy_rot
        end_x = start_x - ylenc * np.sin(angle_rad)
        end_y = start_y + ylenc * np.cos(angle_rad)
        ax.plot([start_x, end_x], [start_y, end_y], 
                color=grid_color, alpha=grid_alpha, linewidth=0.5, transform=proj)
    
    # Plot grid outline with more emphasis
    corners = np.array([
        [xpc, ypc],  # Bottom left
        [xpc + xlenc * np.cos(angle_rad), ypc + xlenc * np.sin(angle_rad)],  # Bottom right
        [xpc + xlenc * np.cos(angle_rad) - ylenc * np.sin(angle_rad), 
         ypc + xlenc * np.sin(angle_rad) + ylenc * np.cos(angle_rad)],  # Top right
        [xpc - ylenc * np.sin(angle_rad), ypc + ylenc * np.cos(angle_rad)],  # Top left
        [xpc, ypc]  # Close the polygon
    ])
    # Plot outline in a darker gray with more opacity
    ax.plot(corners[:, 0], corners[:, 1], color='0.4', alpha=0.8, 
            linewidth=2, transform=proj, label=f'SWAN Grid\nResolution: {dx:.1f}m x {dy:.1f}m')
    
    # Plot additional points if provided
    if points is not None:
        if not isinstance(points, list):
            points = [points]  # Convert single point to list
        if point_labels is None:
            point_labels = [f'Point {i+1}' for i in range(len(points))]
        
        for (x, y), label in zip(points, point_labels):
            ax.plot(x, y, 'k*', markersize=10, label=label, transform=proj)
            # Add text annotation with white background
            ax.text(x + 1000, y + 1000, label, transform=proj,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add scale bar (10 km)
    scale_bar_length = 10000  # 10 km
    scale_bar_x = corners[0, 0] + 20000  # Offset from bottom left corner
    scale_bar_y = corners[0, 1] + 20000
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
            [scale_bar_y, scale_bar_y], 
            'k-', linewidth=2, transform=proj)
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 2000, 
            '10 km', ha='center', transform=proj,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_aspect('equal')
    
    # Set extent to show both bathymetry and grid
    margin = 50000  # 50km margin (increased from 10km)
    ax.set_extent([
        min(bathy.lon.min(), corners[:, 0].min()) - margin,
        max(bathy.lon.max(), corners[:, 0].max()) + margin,
        min(bathy.lat.min(), corners[:, 1].min()) - margin,
        max(bathy.lat.max(), corners[:, 1].max()) + margin
    ], crs=proj)
    
    plt.show()


def plot_bathy_with_locations(bathy: xr.DataArray, locations_file: str, utm_zone=18):
    """
    Plot bathymetry with SWAN output locations overlay.
    
    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    locations_file : str
        Path to the SWAN locations file
    utm_zone : int
        UTM zone number
    """
    # Set up the projection
    proj = ccrs.UTM(zone=utm_zone)
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})
    
    # Plot bathymetry
    bathy.plot.contourf(
        ax=ax,
        x='lon',
        y='lat',
        levels=[-1000, -500, -200, -100, -50, -25, -10, 0],
        cmap="Blues_r",
        add_colorbar=True,
        transform=proj
    )
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20)
    
    # Load and plot locations
    locations = np.loadtxt(locations_file)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    
    # Plot locations with small dots
    scatter = ax.scatter(x_coords, y_coords, 
                        c='0.4',           # Dark gray color
                        alpha=0.6,         # 60% opacity
                        s=2,               # Small point size
                        transform=proj,
                        label=f'Output locations\n({len(x_coords)} points)')
    
    # Add scale bar (10 km)
    scale_bar_length = 10000  # 10 km
    scale_bar_x = x_coords[0] + 20000  # Offset from first point
    scale_bar_y = y_coords[0] + 20000
    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], 
            [scale_bar_y, scale_bar_y], 
            'k-', linewidth=2, transform=proj)
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 2000, 
            '10 km', ha='center', transform=proj)
    
    ax.legend()
    ax.set_aspect('equal')
    
    # Set extent to show both bathymetry and locations
    margin = 10000  # 10km margin
    ax.set_extent([
        min(bathy.lon.min(), x_coords.min()) - margin,
        max(bathy.lon.max(), x_coords.max()) + margin,
        min(bathy.lat.min(), y_coords.min()) - margin,
        max(bathy.lat.max(), y_coords.max()) + margin
    ], crs=proj)
    
    plt.show()
    
    return fig, ax
