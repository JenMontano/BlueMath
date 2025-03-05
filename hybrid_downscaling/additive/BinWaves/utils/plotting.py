import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import wavespectra
import xarray as xr
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


def plot_case_variables(data: xr.Dataset, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    data["Hsig"].plot(ax=axes[0], **kwargs)
    data["Tm02"].plot(ax=axes[1], **kwargs)
    data["Dir"].plot(ax=axes[2], **kwargs)
    axes[0].set_aspect("equal")
    axes[1].set_aspect("equal")
    axes[2].set_aspect("equal")


def plot_wave_series(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    buoy_data.hs().plot(ax=axes[0], label="Buoy", c="darkred", alpha=0.8)
    buoy_data.tp().plot(ax=axes[1], label="Buoy", c="darkred", alpha=0.8)
    buoy_data.dpm().plot(ax=axes[2], label="Buoy", c="darkred", alpha=0.8)
    binwaves_data.hs().plot(ax=axes[0], label="BinWaves", c="dodgerblue", alpha=0.8)
    binwaves_data.tp().plot(ax=axes[1], label="BinWaves", c="dodgerblue", alpha=0.8)
    binwaves_data.dpm().plot(ax=axes[2], label="BinWaves", c="dodgerblue", alpha=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    hs = np.vstack([buoy_data.hs().values, binwaves_data.hs().values])
    hs = gaussian_kde(hs)(hs)
    axes[0].scatter(
        buoy_data.hs().values,
        binwaves_data.hs().values,
        s=1,
        c=hs,
        cmap="turbo",
        label="Hs",
    )
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
    for ax in axes:
        ax.set_aspect("equal")
