import os.path as op

import numpy as np
import wavespectra
import xarray as xr
from bluemath_tk.wrappers._utils_wrappers import write_array_in_file
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper
from wavespectra.construct import construct_partition

example_directions = [
    7.5,
    22.5,
    37.5,
    52.5,
    67.5,
    82.5,
    97.5,
    112.5,
    127.5,
    142.5,
    157.5,
    172.5,
    187.5,
    202.5,
    217.5,
    232.5,
    247.5,
    262.5,
    277.5,
    292.5,
    307.5,
    322.5,
    337.5,
    352.5,
]
example_frequencies = [
    0.035,
    0.0385,
    0.04235,
    0.046585,
    0.0512435,
    0.05636785,
    0.06200463,
    0.0682051,
    0.07502561,
    0.08252817,
    0.090781,
    0.0998591,
    0.10984501,
    0.12082952,
    0.13291247,
    0.14620373,
    0.1608241,
    0.17690653,
    0.19459718,
    0.21405691,
    0.2354626,
    0.25900885,
    0.28490975,
    0.31340075,
    0.3447408,
    0.37921488,
    0.4171364,
    0.45885003,
    0.50473505,
]


class BinWavesWrapper(SwanModelWrapper):
    """
    Wrapper example for the BinWaves model.
    """

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        depth_dataarray: xr.DataArray = None,
        templates_name: dict = "all",
        debug: bool = True,
        lon_range: tuple = (454373, 544488),
        lat_range: tuple = (3873157, 4094989),
        buoy_locations: list = None,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
        
        Args:
            templates_dir (str): Directory containing template files
            metamodel_parameters (dict): Parameters for the metamodel
            fixed_parameters (dict): Fixed parameters for the model
            output_dir (str): Directory for output files
            depth_dataarray (xr.DataArray, optional): Depth data array
            templates_name (dict, optional): Template names. Defaults to "all"
            debug (bool, optional): Debug mode flag. Defaults to True
            lon_range (tuple, optional): Longitude range (min, max). Defaults to (454373, 544488)
            lat_range (tuple, optional): Latitude range (min, max). Defaults to (3873157, 4094989)
            buoy_locations (list, optional): List of buoy locations as [lon, lat] pairs. Defaults to None
        """
        depth_array = depth_dataarray.values
        locations_x, locations_y = np.meshgrid(
            depth_dataarray.sel(
                lon=slice(lon_range[0], lon_range[1]), 
                lat=slice(lat_range[0], lat_range[1])
            ).lon.values,
            depth_dataarray.sel(
                lon=slice(lon_range[0], lon_range[1]), 
                lat=slice(lat_range[0], lat_range[1])
            ).lat.values
        )
        self.locations = np.column_stack((locations_x.ravel(), locations_y.ravel()))
        
        # Add buoy locations if provided
        if buoy_locations is None:
            # Default buoy locations if none provided
            buoy_locations = [
                [514397.61, 4051843.74],  # 36.6120N, -74.8390W - buoy 44088
                [446728.66, 4012728.08],  # 36.2580, -75.5930 - buoy 44100
                [462056.64, 3984141.31],  # 36.0010, -75.4210 - buoy 44086
                [435811.25, 4006367.97],  # 36.2000, -75.7140 - buoy 44056
                [470164.24, 3956270.58],  # 35.7500, -75.3300 - buoy 44095
                [458576.64, 3874246.18],  # 35.0100, -75.4540 - buoy 41025
            ]
        
        for buoy_loc in buoy_locations:
            self.locations = np.vstack((self.locations, buoy_loc))
        
        # Print dataset structure for debugging
        print("Dataset coordinates:", depth_dataarray.coords)
        print("Dataset dimensions:", depth_dataarray.dims)
        print("Dataset variables:", depth_dataarray.variables if hasattr(depth_dataarray, 'variables') else "No variables")

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            depth_array=depth_array,
            debug=debug,
        )

    def build_case(self, case_dir: str, case_context: dict) -> None:
        super().build_case(case_context=case_context, case_dir=case_dir)
        write_array_in_file(
            array=self.locations, filename=op.join(case_dir, "locations.loc")
        )

        input_spectrum = construct_partition(
            freq_name="jonswap",
            freq_kwargs={
                "freq": sorted(example_frequencies),
                "fp": 1.0 / case_context.get("tp"),
                "hs": case_context.get("hs"),
            },
            dir_name="cartwright",
            dir_kwargs={
                "dir": sorted(example_directions),
                "dm": case_context.get("dir"),
                "dspr": case_context.get("spr"),
            },
        )
        argmax_bin = np.argmax(input_spectrum.values)
        mono_spec_array = np.zeros(input_spectrum.freq.size * input_spectrum.dir.size)
        mono_spec_array[argmax_bin] = input_spectrum.sum(dim=["freq", "dir"])
        mono_spec_array = mono_spec_array.reshape(
            input_spectrum.freq.size, input_spectrum.dir.size
        )
        mono_input_spectrum = xr.Dataset(
            {
                "efth": (["freq", "dir"], mono_spec_array),
            },
            coords={
                "freq": input_spectrum.freq,
                "dir": input_spectrum.dir,
            },
        )
        for side in ["N", "S", "E", "W"]:
            wavespectra.SpecDataset(mono_input_spectrum).to_swan(
                op.join(case_dir, f"input_spectra_{side}.bnd")
            )

    def plot_grid_on_bathy(self, bathy_dataarray: xr.DataArray, figsize=(12, 8)):
        """
        Plot the grid points on top of the bathymetry.
        
        Args:
            bathy_dataarray (xr.DataArray): Bathymetry data array
            figsize (tuple, optional): Figure size. Defaults to (12, 8)
        """
        import matplotlib.pyplot as plt
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bathymetry
        bathy_plot = ax.pcolormesh(
            bathy_dataarray.lon,
            bathy_dataarray.lat,
            bathy_dataarray,
            cmap='Blues_r',
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(bathy_plot, ax=ax)
        cbar.set_label('Depth (m)')
        
        # Plot grid points
        ax.scatter(
            self.locations[:, 0],
            self.locations[:, 1],
            c='red',
            s=10,
            alpha=0.5,
            label='Grid points'
        )
        
        # Plot buoy locations (last 6 points)
        ax.scatter(
            self.locations[-6:, 0],
            self.locations[-6:, 1],
            c='yellow',
            s=50,
            marker='*',
            label='Buoy locations'
        )
        
        # Customize the plot
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Grid Points on Bathymetry')
        ax.legend()
        
        # Make the plot look better
        plt.tight_layout()
        
        return fig, ax
