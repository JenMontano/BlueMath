import pandas as pd
import numpy as np
import wavespectra
import os

def save_wave_series_to_csv(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    times: np.ndarray,
    output_file: str
):
    """
    Save wave series data to a CSV file with the specified format.
    Also saves a 3-hourly version of the same data.
    
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
    print(f"Starting to save wave series data...")
    # print(f"Input data shapes - times: {times.shape}, buoy_data: {buoy_data.shape}, binwaves_data: {binwaves_data.shape}")
    
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
    
    print(f"Created DataFrame with shape: {df.shape}")
    print(f"Saving to: {output_file}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV with the specified datetime format
    df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Successfully saved original data")
    
    # Create and save 3-hourly version
    print("Creating 3-hourly version...")
    df_3h = df.resample('3H', on='date').mean()
    # Insert '_3h' before the .csv extension
    output_file_3h = output_file.replace('_validation.csv', '_validation_3h.csv')
    print(f"Saving 3-hourly data to: {output_file_3h}")
    df_3h.to_csv(output_file_3h, date_format='%Y-%m-%d %H:%M:%S')
    print("Successfully saved 3-hourly data") 