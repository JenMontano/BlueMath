import numpy as np
import xarray as xr


# TODO: Chande ERA5 to CAWCAR
def transform_ERA5_spectrum(
    era5_spectrum: xr.Dataset,
    subset_parameters: dict,
    available_case_num: np.ndarray,
    satellite_correction: bool = False,
) -> xr.Dataset:
    """
    Transform the wave spectra from ERA5/CAWCAR format to binwaves format.

    Parameters
    ----------
    era5_spectrum : xr.Dataset
        The wave spectra dataset in ERA5/CAWCAR format.
    subset_parameters : dict
        A dictionary containing parameters for the subset processing.
    available_case_num : np.ndarray
        The available case numbers.

    Returns
    -------
    xr.Dataset
        The wave spectra dataset in binwaves format.
    """

    # First, reproject the wave spectra to the binwaves format
    # Only rename if the dimensions don't already have the correct names
    rename_dict = {}
    if "frequency" in era5_spectrum.dims:
        rename_dict["frequency"] = "freq"
    if "direction" in era5_spectrum.dims:
        rename_dict["direction"] = "dir"

    ds = era5_spectrum.rename(rename_dict) if rename_dict else era5_spectrum.copy()

    # Spectrum directly downloaded from CAWCAR
    if not satellite_correction:
        ds["efth"] = ds["efth"] * np.pi / 180.0
        ds["dir"] = ds["dir"] - 180.0
        ds["dir"] = np.where(ds["dir"] < 0, ds["dir"] + 360, ds["dir"])
        ds = ds.sortby("dir").sortby("freq")

    # Second, reproject into the available case numbers dimension
    case_num_spectra = []
    for case_num, case_dir, case_freq in zip(
        available_case_num,
        np.array(subset_parameters.get("dir"))[available_case_num],
        np.array(subset_parameters.get("freq"))[available_case_num],
    ):
        try:
            closest_case = (
                ds.efth.sel(freq=case_freq, method="nearest", tolerance=0.001)
                .sel(dir=case_dir, method="nearest", tolerance=2)
                .expand_dims({"case_num": [case_num]})
            )
            case_num_spectra.append(closest_case)
        except Exception as _e:
            # Add a zeros array if the case number is not available
            case_num_spectra.append(
                xr.zeros_like(ds.efth.isel(freq=0, dir=0)).expand_dims(
                    {"case_num": [case_num]}
                )
            )
    ds_case_num = (
        xr.concat(case_num_spectra, dim="case_num").drop_vars("dir").drop_vars("freq")
    )

    return ds, ds_case_num


def locations_grid_outputs(
    fixed_parameters,
    is_geographic: bool = False,
    out_dx=None,
    out_dy=None,
    outputs_limits=None,
    buoy_locations=None,
):
    """
    Generate a rotated output grid (optionally filtered and with appended buoys).

    Parameters
    ----------
    fixed_parameters : dict
        Dictionary containing the fixed parameters of the computational grid.
        It must contain the following keys:
        xpc, ypc : float
            Origin of the computational grid.
        xlenc, ylenc : float
            Lengths of the grid in x and y (in computational space).
        alpc : float
            Rotation angle in degrees (counterclockwise).
        out_dx, out_dy : float
            Desired output grid spacing in x and y (in computational space units).
        is_geographic : bool, optional
            If True, applies longitude scaling for geographic coordinates.
        outputs_limits : dict, optional
        Dictionary with 'x' and/or 'y' keys for min/max filtering in physical space.
    buoy_locations : dict or array-like, optional
        Dictionary or array of buoy locations to append (shape (N,2)).

    Returns
    -------
    locations : ndarray, shape (N,2)
        Array of output locations (x, y) in physical space.
    """

    alpc = fixed_parameters.get("alpc")
    xpc = fixed_parameters.get("xpc")
    ypc = fixed_parameters.get("ypc")
    xlenc = fixed_parameters.get("xlenc")
    ylenc = fixed_parameters.get("ylenc")
    mxc = fixed_parameters.get("mxc")
    myc = fixed_parameters.get("myc")
    is_geographic = fixed_parameters.get("is_geographic", False)

    # Get original grid spacing if out_dx/out_dy not provided
    if out_dx is None:
        out_dx = xlenc / mxc
    if out_dy is None:
        out_dy = ylenc / myc

    # 1. Create output grid in computational space
    mxo = int(np.ceil(xlenc / out_dx))  # number of cells in x with the desired spacing
    myo = int(np.ceil(ylenc / out_dy))  # number of cells in y with the desired spacing
    xo = np.linspace(0, xlenc, mxo + 1)  # x coordinates of the output grid
    yo = np.linspace(0, ylenc, myo + 1)  # y coordinates of the output grid
    XO, YO = np.meshgrid(xo, yo)

    # 2. Handle rotation and translation
    if alpc != 0:
        # Rotated case
        angle_rad = np.radians(alpc)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        if is_geographic:
            mean_lat = ypc
            lon_scale = np.cos(np.radians(mean_lat))
            X_rot = xpc + (XO * cos_angle - YO * sin_angle) / lon_scale
            Y_rot = ypc + (XO * sin_angle + YO * cos_angle)
        else:
            X_rot = xpc + XO * cos_angle - YO * sin_angle
            Y_rot = ypc + XO * sin_angle + YO * cos_angle
    else:
        # Non-rotated case
        if is_geographic:
            mean_lat = ypc
            lon_scale = np.cos(np.radians(mean_lat))
            X_rot = xpc + XO / lon_scale
            Y_rot = ypc + YO
        else:
            X_rot = xpc + XO
            Y_rot = ypc + YO

    # 3. Stack coordinates
    locations = np.column_stack((X_rot.ravel(), Y_rot.ravel()))

    # 4. Filter by output limits (i.e, in case the outputs are in a smaller region than the bathy and computational grid)
    if outputs_limits is not None:
        x_limits = outputs_limits.get("x")
        y_limits = outputs_limits.get("y")
        mask = np.ones(len(locations), dtype=bool)
        if x_limits is not None:
            mask &= (locations[:, 0] >= x_limits[0]) & (locations[:, 0] <= x_limits[1])
        if y_limits is not None:
            y_min, y_max = y_limits
            if y_min is not None:
                mask &= locations[:, 1] >= y_min
            if y_max is not None:
                mask &= locations[:, 1] <= y_max
        locations = locations[mask]

    # 5. Append buoy locations if provided
    if buoy_locations is not None:
        if isinstance(buoy_locations, dict):
            buoy_coords = np.array(list(buoy_locations.values()))
        else:
            buoy_coords = np.array(buoy_locations)
        locations = np.vstack((locations, buoy_coords))

    return locations
