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
        spacing: float = None,
        buoy_locations: dict = None,
        depth_dataarray: xr.DataArray = None,
        outputs_limits: dict = None,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.

        Parameters
        ----------
        templates_dir : str
            Directory containing the templates
        metamodel_parameters : dict
            Dictionary containing the metamodel parameters
        fixed_parameters : dict
            - xpc, ypc: Origin coordinates
            - alpc: Grid rotation angle in degrees
            - xlenc, ylenc: Grid lengths in x and y
            - mxc, myc: Number of grid cells
        output_dir : str
            Directory where the output files will be stored
        spacing : float, optional
            Outputs locations spacing. Units should match the coordinate system of depth_dataarray. If None, it will be the same as the depth_dataarray.
        buoy_locations : dict, optional
            Dictionary containing the buoy locations in the form {name: (x, y)}
        depth_dataarray : xr.DataArray
            DataArray containing the depth data
        templates_name : dict, optional
            Dictionary containing the template names
        debug : bool, optional
            Whether to print debug information
        outputs_limits : dict, optional
            Dictionary containing the geographic limits for the outputs (locations), i.e. the outputs can be in a smaller region than the bathy and computational grid. If None, the output locations will be created from the depth_dataarray having the same extent.
            Example for UTM: {
                'x': (425000, 440000),  # (min_x, max_x)
                'y': (4820000, None)    # (min_y, max_y), None means no upper limit
            }
        """
        depth_array = depth_dataarray.values
        
        # Get grid parameters from fixed_parameters
        xpc = fixed_parameters['xpc']      # Origin y
        ypc = fixed_parameters['ypc']      # Origin y
        xlenc = fixed_parameters['xlenc']  # Length in x
        ylenc = fixed_parameters['ylenc']  # Length in y
        alpc = fixed_parameters['alpc']    # Rotation angle in degrees
        mxc = fixed_parameters.get('mxc')  # Number of cells in x
        myc = fixed_parameters.get('myc')  # Number of cells in y

        # Get available coordinate names
        coord_names = list(depth_dataarray.coords)
        is_geographic = any(name in ['lon', 'longitude'] for name in coord_names) and \
                   any(name in ['lat', 'latitude'] for name in coord_names)
        # Get coordinate variables
        if is_geographic:
            x_coord = next(name for name in coord_names if name in ['lon', 'longitude'])
            y_coord = next(name for name in coord_names if name in ['lat', 'latitude'])
        else:
            x_coord = next(name for name in coord_names if name in ['x', 'X', 'cx', 'easting'])
            y_coord = next(name for name in coord_names if name in ['y', 'Y', 'cy', 'northing'])
        
        units = "degrees" if is_geographic else "m"
        # Handle spacing of the outputs locationsbased on coordinate system
        if spacing is None:
            # Get the resolution from depth_dataarray coordinates
            x_coords = depth_dataarray[x_coord].values
            y_coords = depth_dataarray[y_coord].values
            if len(x_coords) > 1 and len(y_coords) > 1:
                x_spacing = abs(x_coords[1] - x_coords[0])
                y_spacing = abs(y_coords[1] - y_coords[0])
                spacing = max(x_spacing, y_spacing)  
                
                print(f"Outputs locations spacing: {spacing:.6f} {units}")
            else:
                raise ValueError("depth_dataarray must have at least 2 points in each dimension to calculate spacing")

        if depth_dataarray.shape[0] < 2 or depth_dataarray.shape[1] < 2:
            raise ValueError("depth_dataarray must have at least 2 points in each dimension to calculate spacing")
        
        # Calculate outputs dimensions based on desiredspacing
        if spacing is not None:
    
            # Determine if grid is rotated based on alpc
            is_rotated = abs(alpc) > 1e-10
            
            if is_rotated:
                # For rotated grids, always use spacing to determine size
                print(f"Using spacing input {spacing:.6f} {units} for rotated outputs locations")
                mxc = int(np.ceil(xlenc / spacing))
                myc = int(np.ceil(ylenc / spacing))
            else:
                mxc = depth_dataarray.shape[1] - 1
                myc = depth_dataarray.shape[0] - 1
            
            # Adjust lengths to match exact number of cells
            xlenc = mxc * spacing
            ylenc = myc * spacing
        
        # Create output locations for Kp 
        xc = np.linspace(0, xlenc, mxc + 1)  
        yc = np.linspace(0, ylenc, myc + 1)  
        XC, YC = np.meshgrid(xc, yc)        
        
        angle_rad = np.radians(alpc)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        if is_geographic:
            # For geographic coordinates, we need to handle the latitude-dependent scaling
            mean_lat = ypc
            # Scale factor for longitude based on latitude
            lon_scale = np.cos(np.radians(mean_lat))
            # Transform from computational to problem coordinates
            # First apply rotation
            X_rot = xpc + (XC * cos_angle - YC * sin_angle) / lon_scale
            Y_rot = ypc + (XC * sin_angle + YC * cos_angle)
        else:
            # For Cartesian coordinates (e.g., Cantabria case)
            if abs(alpc) < 1e-10:  # If rotation is effectively zero
                X_rot = xpc + XC
                Y_rot = ypc + YC
            else:
                # For projected coordinates, simple rotation and translation
                X_rot = xpc + XC * cos_angle - YC * sin_angle
                Y_rot = ypc + XC * sin_angle + YC * cos_angle
        
        # Stack coordinates and create locations array
        locations_before_stack = np.column_stack((X_rot.ravel(), Y_rot.ravel()))
        
        # Filter locations based on outputs_limits if specified
        if outputs_limits is not None:
            x_limits = outputs_limits.get('x')
            y_limits = outputs_limits.get('y')
            
            mask = np.ones(len(locations_before_stack), dtype=bool)
            
            if x_limits is not None:
                mask &= (locations_before_stack[:, 0] >= x_limits[0]) & (locations_before_stack[:, 0] <= x_limits[1])
            
            if y_limits is not None:
                y_min, y_max = y_limits
                if y_min is not None:
                    mask &= (locations_before_stack[:, 1] >= y_min)
                if y_max is not None:
                    mask &= (locations_before_stack[:, 1] <= y_max)
            
            locations_before_stack = locations_before_stack[mask]
            
        self.locations = locations_before_stack
        
        # Add buoy locations if provided
        if buoy_locations is not None:
            buoy_coords = np.array(list(buoy_locations.values()))
            self.locations = np.vstack((self.locations, buoy_coords))
      
        
        print(f"\nAdded {len(buoy_locations)} buoy locations")
        print(f"Final # of outputs locations {self.locations.shape[0]}")
        
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
