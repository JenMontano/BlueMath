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
        lat_mask = (lat_values >= 3807836.13) & (lat_values <= 3939782.97)
        lon_mask = (lon_values >= 326019.60) & (lon_values <= 474964.41)
        
        # Get the filtered values
        filtered_lat = lat_values[lat_mask][::10]  # Take every 10th point
        filtered_lon = lon_values[lon_mask][::10]  # Take every 10th point
        
        print("Filtered lat values:", filtered_lat)
        print("Filtered lon values:", filtered_lon)
        
        locations_x, locations_y = np.meshgrid(filtered_lon, filtered_lat)
        print("Meshgrid shapes - x:", locations_x.shape, "y:", locations_y.shape)

        # Flatten the meshgrid to (N, 2) array
        points = np.column_stack((locations_x.ravel(), locations_y.ravel()))

        # Define rotation
        angle_deg = 47  # for example, 45 degrees
        angle_rad = np.deg2rad(angle_deg)

        # Rotation matrix
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])

        # Choose a rotation center (e.g., the center of your mesh)
        center = points.mean(axis=0)

        # Shift points to origin, rotate, then shift back
        rotated_points = (points - center) @ R.T + center

        # Now use rotated_points instead of self.locations
        self.locations = rotated_points
        
        self.locations = np.column_stack((locations_x.ravel(), locations_y.ravel()))
        print("Final locations shape:", self.locations.shape)
        print("First few locations:", self.locations[:5])
        
        # Add specific buoy locations
        buoy_locations = np.array([

            [458576.64, 3874246.18],  # 35.0100, -75.4540 (WGS84) - buoy 41025
        ])
        
        # Add buoy locations to the grid
        self.locations = np.vstack((self.locations, buoy_locations))
        # print("Total number of locations after adding buoys:", len(self.locations))
        print("Number of grid locations:", len(self.locations) - len(buoy_locations))
        print("Number of buoy locations:", len(buoy_locations))
        
      
    
        
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
