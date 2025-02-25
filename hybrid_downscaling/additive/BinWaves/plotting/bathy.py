import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from bluemath_tk.core.plotting.colors import colormap_bathy


def plot_bathymetry(
    bathy: xr.DataArray,
    min_z: int = None,
    max_z: int = None,
    alpha: int = 1,
    figsize: tuple = (20, 10),
    cbar: bool = True,
    ax: plt.Axes = None,
    bathy_lines: bool = False,
):
    """
    Plots bathymetry map
    """

    if min_z is None:
        min_z = int(np.floor(np.nanmin(-bathy.values)))
    if max_z is None:
        max_z = int(np.floor(np.nanmax(-bathy.values)))

    if ax is None:
        # generate figure and axes
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

    # ax.axis('equal')

    # custom colormap for bathymetry
    c1 = colormap_bathy(max_z, min_z)

    # plot bathymetry
    im = ax.pcolorfast(
        bathy.lon.values,
        bathy.lat.values,
        -bathy.values,
        cmap=c1,
        vmax=max_z,
        vmin=min_z,
        zorder=1,
        alpha=alpha,
    )

    ax.contour(bathy.lon.values, bathy.lat.values, -bathy.values, [0], colors="black")

    if bathy_lines:
        CS = ax.contour(
            bathy.lon.values,
            bathy.lat.values,
            -bathy.values,
            [-4000, -3000, -2000, -1000],
            colors="grey",
            alpha=0.7,
            linestyles="solid",
        )
        ax.clabel(CS, CS.levels, inline=True, fontsize=12)

    ax.grid(":", color="grey")
    ax.set_aspect("equal")

    # ax.set_ylim([np.nanmin(bathy.lat.values), np.nanmax(bathy.lat.values)])
    # ax.set_xlim([np.nanmin(bathy.lon.values), np.nanmax(bathy.lon.values)])

    ax.tick_params(axis="both", which="major", labelsize=16)

    if cbar:
        # add colorbar
        plt.colorbar(im, shrink=0.8).set_label("Elevation (m)", fontsize=16)

    return fig, ax


def plot_grid_polygon(
    ax, corner, angle, xlenc, ylenc, label="polygon", alpha=0.2, color="magenta"
):
    angle = np.deg2rad(angle)

    xs = [
        corner[0],
        corner[0] + xlenc * np.cos(angle),
        corner[0] + xlenc * np.cos(angle) - np.sin(angle) * ylenc,
        corner[0] - ylenc * np.sin(np.pi - angle),
    ]
    ys = [
        corner[1],
        corner[1] + xlenc * np.sin(angle),
        corner[1] + xlenc * np.sin(angle) + np.cos(angle) * ylenc,
        corner[1] - ylenc * np.cos(np.pi - angle),
    ]

    ax.fill(xs, ys, color=color, alpha=alpha, label=label)

    return xs, ys
