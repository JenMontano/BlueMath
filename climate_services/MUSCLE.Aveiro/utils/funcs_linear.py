import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split


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
