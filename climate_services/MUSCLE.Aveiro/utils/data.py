from typing import List, Tuple

import numpy as np
import xarray as xr
from bluemath_tk.core.operations import spatial_gradient
from bluemath_tk.predictor.xwt import get_dynamic_estela_predictor


def load_atmospheric_predictor(
    variables: List[str], region: Tuple[float], estela: bool = False
):
    """
    Load atmospheric predictor data for the given region
    """

    era5 = xr.open_dataset("https://geoocean.sci.unican.es/thredds/dodsC/geoocean/era5-msl")
    era5["time"] = era5["time"].astype("timedelta64[D]") + np.datetime64("1940-01-01")
    era5 = (
        era5
        .sel(
            latitude=slice(region[3], region[1], 4), 
            longitude=slice(region[0], region[2], 4), 
            time=slice("2000", None)
        )
        .load()
        # .coarsen(longitude=2, latitude=2).mean()
    )

    if "msl_gradient" in variables:
        era5["msl_gradient"] = spatial_gradient(era5["msl"])

    if estela:
        estela = xr.open_dataset("data/estela_sea.nc")
        era5 = get_dynamic_estela_predictor(data=era5, estela=estela)

    return era5[variables]  # .dropna(dim=["latitude", "longitude"], how="any")
