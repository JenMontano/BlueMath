import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde

import xarray as xr
import numpy as np
import os

from bluemath_tk.core.operations import spatial_gradient
from bluemath_tk.predictor.xwt import get_dynamic_estela_predictor



class ERA5:
    """
    A class for working with ERA5 climate data.

    Args:
        path (str): The path to the ERA5 data file.
        output_dir (str, optional): The output directory for saving processed data. Defaults to "data".

    Attributes:
        path (str): The path to the ERA5 data file.
        output_dir (str): The output directory for saving processed data.

    Methods:
        load_msl_data(coarsen: int = 2, compute_gradient: bool = True) -> xr.Dataset:
            Loads and processes mean sea level (msl) data from ERA5.

        get_spatial_gradient(data: xr.DataArray) -> xr.DataArray:
            Computes the spatial gradient of the given data.

        get_dynamic_estela_predictor(data: xr.Dataset, estela) -> xr.Dataset:
            Generates a dynamic predictor dataset based on the given data and estela.

    """

    def __init__(self, path: str, output_dir: str = "data") -> None:
        self.path = path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _check_file_exists(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.output_dir, filename))

    def _save_data(self, data: xr.Dataset, filename: str) -> None:
        filepath = os.path.join(self.output_dir, filename)
        data.to_netcdf(filepath)
        print(f"Saved: {filepath}")

    def load_msl_data(
        self, coarsen: int = 2, area: np.ndarray = None, compute_gradient: bool = True
    ) -> xr.Dataset:
        """
        Loads and processes mean sea level (msl) data from ERA5.

        Args:
            coarsen (int, optional): The coarsening factor for latitude and longitude. Defaults to 2.
            compute_gradient (bool, optional): Whether to compute the spatial gradient of msl data. Defaults to True.

        Returns:
            xr.Dataset: The processed ERA5 dataset.

        """
        filename = "era5.nc"
        if area:
            filename = f"era5_coars_{coarsen}_{area[0]}_{area[1]}_{area[2]}_{area[3]}.nc"
        else:
            filename = f"era5_coars_{coarsen}.nc"
        if self._check_file_exists(filename):
            print(f"Loading existing file: {filename}")
            return xr.load_dataset(os.path.join(self.output_dir, filename))

        if area is None:
            area = [-180, 360, -90, 90]

        era5 = (
            xr.load_dataset(self.path)
            .sortby(['longitude', 'latitude'])
            # .sel(longitude = slice(area[0], area[1]), latitude = slice(area[2], area[3]))
            # .coarsen(latitude=coarsen, longitude=coarsen, boundary="pad")
            # .mean()
        )


        lon_min, lon_max, lat_min, lat_max = area
        if lon_min < lon_max:
            era5 = era5.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_min, lat_max))
        else:
            era5 = era5.assign_coords(longitude=((era5.longitude + 180) % 360) - 180)
            era5 = era5.sortby("longitude")
            era5 = era5.sel(longitude=slice(lon_min-360, lon_max), latitude=slice(lat_min, lat_max))

        era5 = (era5.coarsen(latitude=coarsen, longitude=coarsen, boundary="pad")
                .mean()
                )

        era5["time"] = era5["time"].astype("timedelta64[D]") + np.datetime64(
            "1940-01-01"
        )

        if compute_gradient:
            era5["msl_gradient"] = spatial_gradient(era5["msl"])

        self._save_data(era5, filename)
        return era5

    def get_spatial_gradient(self, data: xr.DataArray) -> xr.DataArray:
        """
        Computes the spatial gradient of the given data.

        Args:
            data (xr.DataArray): The input data.

        Returns:
            xr.DataArray: The computed spatial gradient.

        """
        return spatial_gradient(data)

    def get_dynamic_estela_predictor(self, data: xr.Dataset, estela) -> xr.Dataset:
        """
        Generates a dynamic predictor dataset based on the given data and estela.

        Args:
            data (xr.Dataset): The input data.
            estela: The estela parameter.

        Returns:
            xr.Dataset: The generated dynamic predictor dataset.

        """
        filename = "era5_dynamic.nc"
        if self._check_file_exists(filename):
            print(f"Loading existing file: {filename}")
            return xr.load_dataset(os.path.join(self.output_dir, filename))

        predictor_ds = get_dynamic_estela_predictor(data=data, estela=estela)

        self._save_data(predictor_ds, filename)
        return predictor_ds


def plot_pcs(pca, n_pcs):
    """
    Plots the principal components (PCs) from a PCA analysis.

    Parameters:
    - pca: The PCA object containing the principal components.
    - n_pcs: The number of principal components to plot.

    Returns:
    None
    """
    fig, axs = plt.subplots(n_pcs, 1, figsize=(15, 5 * n_pcs))
    for i in range(n_pcs):
        pca.pcs.PCs.isel(n_component=i).plot(ax=axs[i], color="darkmagenta")
        axs[i].axhline(0, color="black", lw=1)
        axs[i].grid(":", lw=0.5)
        axs[i].set_title(f"PC {i + 1}")
        axs[i].set_xlim([pca.pcs.time.min(), pca.pcs.time.max()])
        axs[i].set_ylim(
            [
                -np.nanmax(np.abs(pca.pcs.PCs.isel(n_component=range(n_pcs)))),
                np.nanmax(np.abs(pca.pcs.PCs.isel(n_component=range(n_pcs)))),
            ]
        )


def plot_waves(waves):
    """
    Plots the wave data.

    This function creates a subplots figure and plots the wave data for each variable.

    Returns:
        None
    """

    fig, axs = plt.subplots(
        len(waves.data_vars.keys()), 1, figsize=(20, 4 * len(waves.data_vars.keys()))
    )
    for iv, var in enumerate(list(waves.data_vars.keys())):
        try:
            ax = axs[iv]
        except IndexError:
            ax = axs

        if var == "mwd":
            waves[var].plot(ax=ax, color="crimson", marker=".", lw=0, ms=1)
        else:
            waves[var].plot(ax=ax, color="crimson")

        ax.grid(":", lw=0.5)
        ax.set_title(var)
        ax.set_xlim([waves.time.min(), waves.time.max()])
        ax.set_ylim([waves[var].min(), waves[var].max()])


def fit_plot_linear_model(X, Ys, keys=None, perc_train=0.8):
    """
    Fits and plots linear regression models for multiple dependent variables.

    Parameters:
        X (array-like): The independent variable data.
        Ys (array-like): The dependent variable data.
        perc_train (float, optional): The percentage of data to use for training. Defaults to 0.8.

    Returns:
        list: A list of fitted linear regression models.

    """
    fig, axs = plt.subplots(1, Ys.shape[1], figsize=(7 * Ys.shape[1], 5))

    MODELS = []

    if not keys:
        keys = [f"Y{i}" for i in range(Ys.shape[1])]

    for i in range(Ys.shape[1]):
        try:
            ax = axs[i]
        except IndexError:
            ax = axs

        X = sm.add_constant(X)  # Adds a constant column (for the intercept term)
        y = Ys[:, i]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=1 - perc_train, random_state=42
        )

        # Add constant (for the intercept) in X_train and X_val
        X_train = sm.add_constant(X_train)
        X_val = sm.add_constant(X_val)

        # Create and fit the OLS model with the training data
        model = sm.OLS(y_train, X_train)
        results = model.fit()

        MODELS.append(results)

        # Get predictions for the validation set
        y_val_pred = results.predict(X_val)

        # Compute density using Gaussian KDE
        xy = np.vstack([y_val, y_val_pred])
        density = gaussian_kde(xy)(xy)

        # Create the scatter plot for the validation set with density coloring
        sc = ax.scatter(y_val, y_val_pred, c=density, cmap="viridis", alpha=0.8, s=2)
        ax.plot(
            [min(y_val), max(y_val)],
            [min(y_val), max(y_val)],
            color="red",
            linestyle="--",
        )  # 45° reference line

        ax.set_xlabel(f"Actual Values ({keys[i]}) - Validation")
        ax.set_ylabel(f"Predicted Values ({keys[i]}) - Validation")
        ax.set_aspect("equal", "box")

        ax.set_title(f"{keys[i]}")

        # Add colorbar
        fig.colorbar(sc, ax=ax, label="Density", shrink=0.6)

    return MODELS
