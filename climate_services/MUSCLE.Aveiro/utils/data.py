from typing import List, Tuple

import xarray as xr
from bluemath_tk.core.operations import spatial_gradient
from bluemath_tk.predictor.xwt import get_dynamic_estela_predictor


def load_atmospheric_predictor(
    variables: List[str], region: Tuple[float], estela: bool = False
):
    """
    Load atmospheric predictor data for the given region
    """

    era5 = xr.open_dataset("data/era5_atmospheric.nc")
    # era5 = era5.sel(
    #     latitude=slice(region[3], region[1]), longitude=slice(region[0], region[2])
    # )

    # if "msl_gradient" in variables:
    #     era5["msl_gradient"] = spatial_gradient(era5["msl"])

    if estela:
        estela = xr.open_dataset("data/estela_sea.nc")
        era5 = get_dynamic_estela_predictor(data=era5, estela=estela)

    return era5[variables]  # .dropna(dim=["latitude", "longitude"], how="any")
