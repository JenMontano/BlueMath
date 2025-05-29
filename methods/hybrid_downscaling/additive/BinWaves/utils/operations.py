import numpy as np
import xarray as xr
from pyproj import Transformer


def transform_ERA5_spectrum(
    era5_spectrum: xr.Dataset,
    subset_parameters: dict,
    available_case_num: np.ndarray,
) -> xr.Dataset:
    """
    Transform the wave spectra from ERA5 format to binwaves format.

    Parameters
    ----------
    era5_spectrum : xr.Dataset
        The wave spectra dataset in ERA5 format.
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
    ds = era5_spectrum.rename({"frequency": "freq", "direction": "dir"})
    
    # Round timestamps to nearest hour to ensure unique time values
    ds = ds.assign_coords(time=ds.time.dt.round('H'))
    
    # Handle spatial coordinates
    if 'utm_x' in ds.dims and 'utm_y' in ds.dims:
        # Dataset already has UTM coordinates as dimensions
        pass
    elif 'longitude' in ds.coords and 'latitude' in ds.coords:
        # Convert geographic coordinates to UTM coordinates
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
        
        # Get the original coordinates
        lon = ds.longitude.values
        lat = ds.latitude.values
        
        # Convert to UTM
        utm_x, utm_y = transformer.transform(lon, lat)
        
        # Create new dimensions for UTM coordinates
        ds = ds.expand_dims({'utm_x': [utm_x[0]], 'utm_y': [utm_y[0]]})
        
        # Drop the old coordinates
        ds = ds.drop_vars(['longitude', 'latitude'])
    
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
