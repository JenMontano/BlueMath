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
    ) -> None:
        """
        Initialize the SWAN model wrapper.
        """

        depth_array = depth_dataarray.values
        print("Original depth_dataarray shape:", depth_dataarray.shape)
        print("Original lat values:", depth_dataarray.lat.values)
        
        # Get the actual lat values first
        lat_values = depth_dataarray.lat.values
        lon_values = depth_dataarray.lon.values
        
        # Find the indices for our desired range
        lat_mask = (lat_values >= 3851757.252582) & (lat_values <= 4072853.979589)
        lon_mask = (lon_values >= 371933.139407) & (lon_values <= 553525.417190)
        
        # Get the filtered values
        filtered_lat = lat_values[lat_mask][::10]  # Take every 10th point
        filtered_lon = lon_values[lon_mask][::10]  # Take every 10th point
        
        print("Filtered lat values:", filtered_lat)
        print("Filtered lon values:", filtered_lon)
        
        locations_x, locations_y = np.meshgrid(filtered_lon, filtered_lat)
        print("Meshgrid shapes - x:", locations_x.shape, "y:", locations_y.shape)
        
        self.locations = np.column_stack((locations_x.ravel(), locations_y.ravel()))
        print("Final locations shape:", self.locations.shape)
        print("First few locations:", self.locations[:5])
        
        # Add specific buoy locations
        buoy_locations = np.array([
            [514397.61, 4051843.74],  # 36.6120N, -74.8390W -buoy 44088
            [446728.66, 4012728.08],  # 36.2580, -75.5930 (WGS84) - buoy 44100
            [462056.64, 3984141.31],  # 36.0010, -75.4210 (WGS84) - buoy 44086
            [435811.25, 4006367.97],  # 36.2000, -75.7140 (WGS84) - buoy 44056
            [470164.24, 3956270.58],  # 35.7500, -75.3300 (WGS84) - buoy 44095
            [474075.136, 3901692.114], # 35.2580, -75.2850 (WGS84) buoy 41120
            [458576.64, 3874246.18],  # 35.0100, -75.4540 (WGS84) - buoy 41025
        ])
        
        # Add buoy locations to the grid
        self.locations = np.vstack((self.locations, buoy_locations))
        print("Total number of locations after adding buoys:", len(self.locations))
        print("Number of grid locations:", len(self.locations) - len(buoy_locations))
        print("Number of buoy locations:", len(buoy_locations))
        
        # Get the last 6 positions (buoy locations)
        self.sites_for_spectrum = list(range(-6, 0))  # This will give [-6, -5, -4, -3, -2, -1]
        print("Sites for spectrum (last 6 positions):", self.sites_for_spectrum)
        
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
