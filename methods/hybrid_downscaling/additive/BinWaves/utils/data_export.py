import pandas as pd
import numpy as np
import wavespectra

def save_wave_series_to_csv(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    times: np.ndarray,
    output_file: str
):
    """
    Save wave series data to a CSV file with the specified format.
    
    Parameters:
    -----------
    buoy_data : wavespectra.SpecArray
        Buoy data containing Hs_Buoy, Tp_Buoy, and Dir_Buoy
    binwaves_data : wavespectra.SpecArray
        BinWaves data containing hs, tp, and dpm
    times : np.ndarray
        Array of timestamps
    output_file : str
        Path to the output CSV file
    """
    # Create a DataFrame with the specified columns
    df = pd.DataFrame({
        'date': pd.to_datetime(times),
        'Hs_Buoy': buoy_data['Hs_Buoy'].values,
        'T_Buoy': buoy_data['Tp_Buoy'].values,
        'Dir_Buoy': buoy_data['Dir_Buoy'].values,
        'Hs_Model': binwaves_data.hs().values,
        'T_Model': binwaves_data.tp().values,
        'Dir_Model': binwaves_data.dpm().values
    })
    
    # Save to CSV with the specified datetime format
    df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S') 