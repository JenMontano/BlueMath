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
        
        # Get grid parameters from fixed_parameters
        xpc = fixed_parameters['xpc']      # Origin x
        ypc = fixed_parameters['ypc']      # Origin y
        xlenc = fixed_parameters['xlenc']  # Length in x
        ylenc = fixed_parameters['ylenc']  # Length in y
        alpc = fixed_parameters['alpc']    # Rotation angle in degrees
        
        # Use 500m spacing in both directions
        spacing = 5000  # 500m spacing
        
        # Calculate number of points in each direction
        nx = int(xlenc / spacing) + 1  # Number of points along each line
        ny = int(ylenc / spacing) + 1  # Number of parallel lines
        
        # Calculate angle in radians
        angle_rad = np.radians(alpc)
        
        # Create vectors for the parallel lines
        dx = spacing * np.cos(angle_rad)  # x-component for points along line
        dy = spacing * np.sin(angle_rad)  # y-component for points along line
        
        # Create perpendicular vectors for the line spacing
        dx_perp = -spacing * np.sin(angle_rad)  # perpendicular x-component
        dy_perp = spacing * np.cos(angle_rad)   # perpendicular y-component
        
        # Initialize points list
        points = []
        
        # Generate points along parallel lines
        for i in range(ny):  # for each parallel line
            # Starting point for this line
            start_x = xpc + i * dx_perp
            start_y = ypc + i * dy_perp
            
            # Generate points along this line
            for j in range(nx):
                x = start_x + j * dx
                y = start_y + j * dy
                points.append([x, y])
        
        # Convert to numpy array and keep all points
        self.locations = np.array(points)
        print("Grid points shape:", self.locations.shape)
        
        # Add buoy locations
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
