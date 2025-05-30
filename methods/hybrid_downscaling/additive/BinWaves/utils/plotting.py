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
        levels=[0, -10, -25, -50, -100, -200, -500, -1000],
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

    # Second figure - Scatter plots
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
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

    # Save figures if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig1.savefig(os.path.join(save_dir, f'timeseries_buoy_{buoyId}_validation.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, f'scatter_buoy_{buoyId}_validation.png'), dpi=300, bbox_inches='tight')

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
    sites_for_spectrum: list,
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
    sites_for_spectrum : list
        List of site indices to plot spectra for
    """

    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot bathymetry as a countour (using UTM coordinates cx, cy)
    bathy.plot.contourf(
        ax=ax,
        x='cx',
        y='cy',
        levels=[0, -10, -25, -50, -100, -200, -500, -1000],
        cmap="Blues_r",
        add_colorbar=False,
    )

    # Find the closest time index for all datasets
    time_to_plot_dt = np.datetime64(time_to_plot)
    reconstructed_time_idx = np.abs(reconstructed_onshore_spectra.time.values - time_to_plot_dt).argmin()
    offshore_time_idx = np.abs(offshore_spectra.time.values - time_to_plot_dt).argmin()

    # Plot reconstructed Hs in grid (using UTM coordinates utm_x, utm_y)
    hs_map = (
        reconstructed_onshore_spectra.isel(time=reconstructed_time_idx)
        .kp.spec.hs()
        .values
    )
    phs = ax.scatter(
        reconstruction_kps.utm_x.values,
        reconstruction_kps.utm_y.values,
        c=hs_map,
        cmap=colormap_spectra(),
    )
    plt.colorbar(phs).set_label("Hs [m]")

    # Plot onshore spectra at sites (using UTM coordinates utm_x, utm_y)
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

    # Set the plot bounds
    plot_bounds = (
        363000.0,    # min easting (slightly less than GEBCO min: 363166.25)
        546000.0,    # max easting (slightly more than GEBCO max: 545566.25)
        3873000.0,   # min northing (slightly less than GEBCO min: 3873132.82)
        4096000.0,   # max northing (slightly more than GEBCO max: 4095832.82)
    )
    ax.axis(plot_bounds)

    return fig, ax
