import pandas as pd
import numpy as np
import os

def save_validation_csv_with_interpolation(
    buoy_data: pd.DataFrame,
    binwaves_hs: np.ndarray,
    binwaves_tp: np.ndarray,
    binwaves_dpm: np.ndarray,
    buoy_id: str,
    save_path: str = "outputs",
    filename_prefix: str = "buoy",
    target_resolution: str = "1D",
):
    """
    Save validation data to CSV files, creating hourly data and interpolating to target resolution.
    Assumes buoy_data and binwaves_* arrays are already time-aligned and of the same length.
    
    Parameters:
    -----------
    buoy_data : pd.DataFrame
        Buoy wave data with columns: Hs_Buoy, Tp_Buoy, Dir_Buoy
    binwaves_hs : np.ndarray
        BinWaves significant wave height time series
    binwaves_tp : np.ndarray
        BinWaves peak period time series
    binwaves_dpm : np.ndarray
        BinWaves mean direction time series
    buoy_id : str
        Buoy ID for filename
    save_path : str, optional
        Directory to save the CSV files (default: "outputs")
    filename_prefix : str, optional
        Prefix for the filename (default: "buoy")
    target_resolution : str, optional
        Target resolution for interpolation. Options: "3h", "6h", "1D"/"daily" (default: "1D")
    
    Returns:
    --------
    tuple
        (hourly_filepath, interpolated_filepath) - Paths to both saved CSV files
    """
    import os
    import numpy as np
    import pandas as pd

    print(f"Creating validation data with interpolation to {target_resolution}...")

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Check that all arrays are the same length
    n = len(buoy_data)
    if not (len(binwaves_hs) == len(binwaves_tp) == len(binwaves_dpm) == n):
        raise ValueError(f"All input arrays must have the same length. Got: len(buoy_data)={n}, len(binwaves_hs)={len(binwaves_hs)}, len(binwaves_tp)={len(binwaves_tp)}, len(binwaves_dpm)={len(binwaves_dpm)}")

    # Create hourly DataFrame (original data)
    hourly_df = pd.DataFrame({
        'datetime': buoy_data.index,
        'Hs_Buoy': buoy_data['Hs_Buoy'].values,
        'Tp_Buoy': buoy_data['Tp_Buoy'].values,
        'Dir_Buoy': buoy_data['Dir_Buoy'].values,
        'Hs_BinWaves': binwaves_hs,
        'Tp_BinWaves': binwaves_tp,
        'Dir_BinWaves': binwaves_dpm,
    })
    hourly_df.set_index('datetime', inplace=True)

    # Create regular hourly time index for interpolation
    start_time = hourly_df.index.min()
    end_time = hourly_df.index.max()

    # Create hourly time index
    hourly_times = pd.date_range(start=start_time, end=end_time, freq='1H')

    # Reindex to hourly and interpolate, but only for small gaps (e.g., up to 3 hours)
    hourly_df_interpolated = hourly_df.reindex(hourly_times).interpolate(
        method='linear', limit=3, limit_direction='both', limit_area='inside'
    )

    # Create target resolution time index
    if target_resolution == "3h":
        target_times = pd.date_range(start=start_time, end=end_time, freq='3H')
    elif target_resolution == "6h":
        target_times = pd.date_range(start=start_time, end=end_time, freq='6H')
    elif target_resolution == "1D" or target_resolution == "daily":
        target_times = pd.date_range(start=start_time, end=end_time, freq='1D')
    else:
        # Default to 3h if not specified
        target_times = pd.date_range(start=start_time, end=end_time, freq='3H')

    # Interpolate to target resolution
    target_df = hourly_df_interpolated.reindex(target_times).interpolate(method='linear')

    # Save hourly data
    hourly_filename = f"{filename_prefix}_{buoy_id}_1h.csv"
    hourly_filepath = os.path.join(save_path, hourly_filename)
    hourly_df_interpolated.to_csv(hourly_filepath)

    # Save target resolution data
    target_filename = f"{filename_prefix}_{buoy_id}_{target_resolution}.csv"
    target_filepath = os.path.join(save_path, target_filename)
    target_df.to_csv(target_filepath)

    print(f"Hourly data saved as: {hourly_filepath}")
    print(f"Hourly data shape: {hourly_df_interpolated.shape}")
    print(f"Hourly time range: {hourly_df_interpolated.index.min()} to {hourly_df_interpolated.index.max()}")

    print(f"\n{target_resolution} data saved as: {target_filepath}")
    print(f"{target_resolution} data shape: {target_df.shape}")
    print(f"{target_resolution} time range: {target_df.index.min()} to {target_df.index.max()}")

    return hourly_filepath, target_filepath