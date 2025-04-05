import ipywidgets as widgets
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.rbf import RBF
from ipywidgets import interact


def animate_case_propagation(
    case_dataset, depth, tini=0, tend=30, tstep=2, figsize=(15, 5)
):
    """
    Function to animate the propagation of the swash for a single case
    """

    fig, ax = plt.subplots(1, figsize=figsize)

    # Init animation
    def init():
        return []

    # Función de actualización de la animación
    def update(frame):
        ax.clear()

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(400, 1250)
        ax.set_ylim(-4, 2)
        ax.set_xlabel("Cross-shore Distance (m)", fontsize=12)
        ax.set_ylabel("Elevation (m)", fontsize=12)

        # bathymetry
        ax.fill_between(
            np.arange(len(depth)),
            np.ones(len(depth)) * depth[-1],
            -depth,
            fc="wheat",
            zorder=2,
        )

        # waves
        elev = case_dataset.isel(Tsec=frame)["Watlev"].values
        ax.fill_between(
            np.arange(len(depth)),
            np.ones(len(depth)) * depth[-1],
            elev,
            fc="deepskyblue",
            alpha=0.5,
            zorder=1,
        )
        ax.set_title("Time : {0} s".format(frame), fontsize=12)

        return []

    # Crear animación
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(tini, tend, tstep), init_func=init, blit=True
    )
    plt.close()

    # Mostrar animación
    return ani


def show_graph_for_different_parameters(pca: PCA, rbf: RBF):
    """
    Show graph for different parameters
    """

    # Function to update the plot based on widget input
    def update_plot(hs=1.5, hs_l0=0.02, vegetation=1):
        # Create dataframe
        df_dataset_single_case = pd.DataFrame(
            data={
                "Hs": [hs],
                "Hs_L0": [hs_l0],
                "VegetationHeight": [vegetation],
            }
        )

        # Spatial Reconstruction
        predicted_hs = rbf.predict(dataset=df_dataset_single_case)

        predicted_hs_ds = xr.Dataset(
            {
                "PCs": (["case_num", "n_component"], predicted_hs.values),
            },
            coords={
                "case_num": [0],
                "n_component": np.arange(len(pca.pcs_df.columns)),
            },
        )

        # Get reconstructed Hs
        ds_output_all = pca.inverse_transform(PCs=predicted_hs_ds)

        fig, ax = plt.subplots(figsize=(14, 6))
        ds_output_all["Hs"].sel(case_num=0).plot(x="Xp", ax=ax, color="k")
        # sm.plot_depthfile(ax=ax)
        # ax.plot(
        #     np.arange(int(pp.swash_proj.np_ini), int(pp.swash_proj.np_fin)),
        #     np.repeat(-2.5, int(pp.swash_proj.np_fin - pp.swash_proj.np_ini)),
        #     color="darkgreen",
        #     linewidth=int(25 * vegetation),
        # )
        ax.set_ylim(-7, 4)
        ax.set_xlim(400, 1160)
        ax.grid(True)

        ax.set_title(
            f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0} and VegetationHeight: {vegetation}"
        )

    # Creating widgets
    widget_hs = widgets.FloatSlider(
        value=1.5, min=0.5, max=3, step=0.5, description="Hs:"
    )
    widget_hs_l0 = widgets.FloatSlider(
        value=0.02, min=0.01, max=0.03, step=0.01, description="Hs_L0:"
    )
    widget_vegetation = widgets.FloatSlider(
        value=1, min=0, max=1.5, step=0.5, description="VegetationHeight:"
    )

    # Using interact to link widgets to the function
    return interact(
        update_plot, hs=widget_hs, hs_l0=widget_hs_l0, vegetation=widget_vegetation
    )


def show_graph_for_all_vegetations(pca: PCA, rbf: RBF, depth, hs=2.0, hs_l0=0.02):
    """
    Show graph for all vegetations
    """

    # Create dataframe
    df_dataset_same_hs = pd.DataFrame(
        data={
            "Hs": np.repeat(hs, 100),
            "Hs_L0": np.repeat(hs_l0, 100),
            "VegetationHeight": np.linspace(0, 1.5, 100),
        }
    )

    # Spatial Reconstruction
    predicted_hs = rbf.predict(dataset=df_dataset_same_hs)

    predicted_hs_ds = xr.Dataset(
        {
            "PCs": (["case_num", "n_component"], predicted_hs.values),
        },
        coords={
            "case_num": range(100),
            "n_component": np.arange(len(pca.pcs_df.columns)),
        },
    )

    # Get reconstructed Hs
    ds_output_all = pca.inverse_transform(PCs=predicted_hs_ds)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create a colormap and a normalization based on vegetation values
    norm = colors.Normalize(vmin=0, vmax=1.5)
    cmap = cm.get_cmap("YlGn", 100)

    for i, case_num in enumerate(ds_output_all["case_num"].values):
        vegetation = df_dataset_same_hs["VegetationHeight"].iloc[case_num]
        color = cmap(norm(vegetation))
        ds_output_all["Hs"].isel(case_num=case_num).plot(
            x="Xp",
            # label=f'Case {case} (Veg={vegetation:.2f})',
            color=color,
            ax=ax,
        )

    # Add colorbar for vegetation scale
    smp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smp.set_array([])
    cbar = plt.colorbar(smp, ax=ax)
    cbar.set_label("Vegetation")

    # bathymetry
    # ax.fill_between(
    #     np.arange(len(depth)),
    #     np.ones(len(depth)) * depth[-1],
    #     -depth,
    #     fc="wheat",
    #     zorder=2,
    # )
    # ax.plot(
    #     np.arange(int(pp.swash_proj.np_ini), int(pp.swash_proj.np_fin)),
    #     np.repeat(-2.5, int(pp.swash_proj.np_fin - pp.swash_proj.np_ini)),
    #     color="darkgreen",
    #     linewidth=10,
    # )
    ax.set_ylim(-7, 4)
    ax.set_xlim(400, 1160)
    ax.grid(True)

    ax.set_title(
        f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0} and different vegetation heights"
    )

def plot_depthfile(depthfile):
        'Plot bathymetry data including friction or vegetation area in case active commands'

        depth = np.loadtxt(depthfile)
        x = range(len(depth))

        fig, ax = plt.subplots(1, figsize = (11, 4))

        ax.fill_between(
            x, - depth[0],  np.zeros((len(depth))),
            facecolor = "deepskyblue",
            alpha = 0.5,
            zorder = 1,
        )
        ax.fill_between(
            x, np.zeros((len(depth))) - depth[0],  -depth,
            facecolor = "wheat",
            alpha = 1,
            zorder = 2,
        )
        ax.plot(
            x, -depth,
            color = 'k',
            zorder = 3,
        )

        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-depth[0], None)

        