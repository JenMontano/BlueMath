import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import wavespectra
import xarray as xr
from bluemath_tk.core.operations import get_uv_components
from matplotlib import colors
from scipy.stats import gaussian_kde

my_feature = cfeature.NaturalEarthFeature(
    "physical",
    "coastline",
    "50m",
    edgecolor="black",
)


def plot_bathymetry(data: xr.DataArray, **kwargs):
    p = data.plot.contourf(
        levels=[0, -50, -100, -250, -500, -1000, -2000],
        transform=ccrs.PlateCarree(),
        subplot_kws={"projection": ccrs.Orthographic(-3.5, 43.5)},
        **kwargs,
    )
    # p.axes.add_feature(my_feature)
    p.axes.gridlines(draw_labels=True)


def plot_cases_grid(data: xr.DataArray, **kwargs):
    p = data.plot(
        col="case_num",
        col_wrap=25,
        **kwargs,
    )
    for ax in p.axes.flat:
        ax.set_aspect("equal")
        ax.set_title("")
        ax.axis("off")


def plot_case_variables(data: xr.Dataset):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    data["Hsig"].plot(
        ax=axes[0],
        cbar_kwargs={"label": "Hsig [m]", "orientation": "horizontal", "shrink": 0.7},
        cmap="bwr",
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
        step = 50
        ax.quiver(
            data["Xp"][::step],
            data["Yp"][::step],
            -dir_u[::step, ::step],
            -dir_v[::step, ::step],
            color="grey",
            scale=25,
        )


def plot_wave_series(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    offshore_data: wavespectra.SpecArray,
    times: np.ndarray,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    buoy_data.hs().plot(ax=axes[0], label="Buoy", c="darkred", alpha=0.8, lw=1)
    buoy_data.tp().plot(ax=axes[1], label="Buoy", c="darkred", alpha=0.8, lw=1)
    axes[2].scatter(
        times,
        buoy_data.dpm().values,
        c="darkred",
        label="Buoy",
        alpha=0.8,
        s=1,
    )
    binwaves_data.hs().plot(
        ax=axes[0], label="BinWaves", c="dodgerblue", alpha=0.8, lw=1
    )
    binwaves_data.tp().plot(
        ax=axes[1], label="BinWaves", c="dodgerblue", alpha=0.8, lw=1
    )
    axes[2].scatter(
        times,
        binwaves_data.dpm().values,
        c="dodgerblue",
        label="BinWaves",
        alpha=0.8,
        s=1,
    )
    offshore_data.hs().plot(ax=axes[0], label="Offshore", c="orange", alpha=0.5, lw=1)
    offshore_data.tp().plot(ax=axes[1], label="Offshore", c="orange", alpha=0.5, lw=1)
    axes[2].scatter(
        times,
        offshore_data.dpm().values,
        c="orange",
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
    hs = np.vstack([buoy_data.hs().values, binwaves_data.hs().values])
    hs = gaussian_kde(hs)(hs)
    axes[0].scatter(
        buoy_data.hs().values,
        binwaves_data.hs().values,
        s=1,
        c=hs,
        cmap="turbo",
    )
    axes[0].plot([0, 7], [0, 7], c="darkred", linestyle="--")
    axes[0].set_xlabel("Hs - Buoy [m]")
    axes[0].set_ylabel("Hs - BinWaves [m]")
    axes[0].set_xlim([0, 7])
    axes[0].set_ylim([0, 7])
    tp = np.vstack([buoy_data.tp().values, binwaves_data.tp().values])
    tp = gaussian_kde(tp)(tp)
    axes[1].scatter(
        buoy_data.tp().values,
        binwaves_data.tp().values,
        s=1,
        c=tp,
        cmap="turbo",
        label="Tp",
    )
    axes[1].plot([0, 20], [0, 20], c="darkred", linestyle="--")
    axes[1].set_xlabel("Tp - Buoy [s]")
    axes[1].set_ylabel("Tp - BinWaves [s]")
    axes[1].set_xlim([0, 20])
    axes[1].set_ylim([0, 20])
    dpm = np.vstack([buoy_data.dpm().values, binwaves_data.dpm().values])
    dpm = gaussian_kde(dpm)(dpm)
    axes[2].scatter(
        buoy_data.dpm().values,
        binwaves_data.dpm().values,
        s=1,
        c=dpm,
        cmap="turbo",
        label="Dpm",
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


def plot_spectrum_in_coastline(bathy, reconstructed_onshore_spectra, offshore_spectra):
    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    bathy.elevation.T.plot(
        ax=ax,
        add_colorbar=False,
        robust=True,
        # cbar_kwargs={
        #     "label": "Elevation [m]",
        #     "orientation": "horizontal",
        # },
        cmap=create_custom_bathy_cmap(),
    )
    ax.set_ylim(43.2, 43.65)
    ax.set_xlim(-4.1, -3.25)
    ax.set_aspect("equal")

    interp_dir = np.linspace(0, 360, 24)
    interp_freq = np.linspace(1 / 20, 1 / 5, 10)
    lons = [-3.46000001, -3.87999998]
    lats = [43.48999999, 43.49000004]

    # Plot onshore spectra
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        axin = ax.inset_axes(
            [lon, lat, 0.1, 0.1], transform=ax.transData, projection="polar"
        )
        axin.pcolor(
            np.deg2rad(interp_dir),
            interp_freq,
            (
                reconstructed_onshore_spectra.sel(site=(i + 1), time="2009-12")
                .efth.mean(dim="time")
                .spec.interp(freq=interp_freq, dir=interp_dir)
                .values
            ),
            zorder=10,
            cmap=create_white_zero_colormap("Spectral"),
        )
        axin.set_theta_zero_location("N", offset=0)
        axin.set_theta_direction(-1)
        axin.axis("off")

    # Plot offshore spectrum
    axoff = ax.inset_axes(
        [-4.05, 43.25, 0.1, 0.1], transform=ax.transData, projection="polar"
    )
    axoff.pcolor(
        np.deg2rad(interp_dir),
        interp_freq,
        (
            offshore_spectra.sel(time="2009-12")
            .efth.mean(dim="time")
            .spec.interp(freq=interp_freq, dir=interp_dir)
            .values
        ),
        zorder=10,
        cmap=create_white_zero_colormap("Spectral"),
    )
    axoff.set_theta_zero_location("N", offset=0)
    axoff.set_theta_direction(-1)
    axoff.axis("off")
    # Add text descriptio in figuro
    ax.text(
        -3.92,
        43.25,
        "Offshore Spectra",
        fontsize=12,
        bbox=dict(facecolor="darkred", alpha=0.5),
    )

    return fig, ax
