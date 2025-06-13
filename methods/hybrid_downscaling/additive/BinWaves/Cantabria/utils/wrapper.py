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
        locations_x, locations_y = np.meshgrid(
            depth_dataarray.sel(
                lon=slice(425000, 440000), lat=slice(4820000, None)
            ).lon.values,
            depth_dataarray.sel(
                lon=slice(425000, 440000), lat=slice(4820000, None)
            ).lat.values,
        )
        self.locations = np.column_stack((locations_x.ravel(), locations_y.ravel()))
        # Add Virgen del Mar exact buoy location
        self.locations = np.vstack((self.locations, [428845.10, 4815606.89]))

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
