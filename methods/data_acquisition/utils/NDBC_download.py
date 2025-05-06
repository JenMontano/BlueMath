import requests
import gzip
import io
import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def download_buoy_data(buoy_id, year):
    """
    Download data for a specific buoy and year, handling different formats:
    - Pre-1990s: YY MM DD hh (no minutes)
    - Post-2012: YYYY MM DD hh mm
    - In between: YYYY MM DD hh
    """
    urls = [
        f"https://www.ndbc.noaa.gov/data/historical/stdmet/{buoy_id}h{year}.txt.gz",
        f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/",
        f"https://www.ndbc.noaa.gov/data/stdmet/{year}/{buoy_id}h{year}.txt.gz"
    ]
    
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content = gzip.decompress(response.content).decode('utf-8')
                
                # Skip the header rows and read the data
                data = []
                lines = content.split('\n')[2:]  # Skip first two lines (headers)
                
                # Check format by looking at the first data line
                first_line = next(line for line in lines if line.strip())
                cols = first_line.split()
                
                # Determine format based on number of columns and year format
                year_val = int(cols[0])
                has_minutes = len(cols) == 18  # Post-2012 format has 18 columns
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            # Convert 2-digit year to 4 digits if needed
                            if int(parts[0]) < 100:
                                parts[0] = str(int(parts[0]) + 1900)
                            
                            # Add minutes column if it doesn't exist
                            if not has_minutes:
                                parts.insert(4, '00')
                            
                            data.append(' '.join(parts))
                
                # Read the modified data
                df = pd.read_csv(io.StringIO('\n'.join(data)), sep='\s+',
                            names=['YYYY', 'MM', 'DD', 'hh', 'mm', 'WD', 'WSPD', 'GST', 'WVHT', 
                                  'DPD', 'APD', 'MWD', 'BAR', 'ATMP', 'WTMP', 'DEWP', 
                                  'VIS', 'TIDE'])
                
                # Validate dates
                valid_dates = (
                    (df['MM'] >= 1) & (df['MM'] <= 12) &
                    (df['DD'] >= 1) & (df['DD'] <= 31) &
                    (df['hh'] >= 0) & (df['hh'] <= 23) &
                    (df['mm'] >= 0) & (df['mm'] <= 59)
                )
                
                df = df[valid_dates].copy()
                
                if len(df) > 0:
                    return df
        except:
            continue
            
    return None

def process_buoy(buoy_id, start_year=1960, end_year=None):
    """
    Download and combine all available data for a buoy between start_year and end_year
    """
    if end_year is None:
        end_year = datetime.now().year
        
    all_data = []

    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory for specific buoy if it doesn't exist
    buoy_dir = os.path.join(base_dir, buoy_id)
    os.makedirs(buoy_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        df = download_buoy_data(buoy_id, year)
        
        if df is not None:
            all_data.append(df)
            print(f"Buoy {buoy_id}: Data found for year {year}")
        else:
            print(f"Buoy {buoy_id}: No data available for year {year}")
    
    if all_data:
        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by date and time
        combined_df = combined_df.sort_values(['YYYY', 'MM', 'DD', 'hh'])
        
        # Save to CSV in the buoy's directory
        output_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_combined.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        return combined_df
    else:
        print(f"No data found for buoy {buoy_id}")
        return None
    
def download_wave_spectra(buoy_id, years, base_dir):
    """
    Download wave spectra data for a specific buoy, saving each year separately
    """
    buoy_dir = os.path.join(base_dir, buoy_id)
    os.makedirs(buoy_dir, exist_ok=True)
    
    for year in years:
        url = f"https://www.ndbc.noaa.gov/data/historical/swden/{buoy_id}w{year}.txt.gz"
        
        try:
            # Read the data
            df = pd.read_csv(url, compression='gzip', sep='\s+',
                           na_values=['MM', '99.00', '999.0'])
            
            # Skip if empty or invalid data
            if df.empty or len(df.columns) < 5:
                print(f"No valid data for {buoy_id} - {year}")
                continue
            
            # Process datetime
            try:
                # Convert 2-digit years to 4-digit years if necessary
                if df.iloc[:, 0].max() < 100:
                    df.iloc[:, 0] = df.iloc[:, 0] + 2000
                
                # Create datetime column
                df['datetime'] = pd.to_datetime(
                    df.iloc[:, 0].astype(str) + '-' +
                    df.iloc[:, 1].astype(str).str.zfill(2) + '-' +
                    df.iloc[:, 2].astype(str).str.zfill(2) + ' ' +
                    df.iloc[:, 3].astype(str).str.zfill(2) + ':' +
                    df.iloc[:, 4].fillna('00').astype(str).str.zfill(2)
                )
                
                # Set datetime as index and drop original date/time columns
                df = df.set_index('datetime').drop(df.columns[:5], axis=1)
                
                # Convert column names to float
                df.columns = df.columns.astype(float)
                
                # Save individual year file
                output_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_spectra_{year}.csv")
                df.to_csv(output_file)
                print(f"Successfully saved data for {buoy_id} - {year}")
                
            except Exception as e:
                print(f"No data found for: {buoy_id} - {year}")
                continue
                
        except Exception as e:
            print(f"No data found for: {buoy_id} - {year}")
            continue

def download_buoy_data(buoy_id, years, base_dir):
    """
    Download directional wave spectra data for specified buoy and years.
    
    Parameters:
    -----------
    buoy_id : str
        Buoy identifier (e.g., '41001')
    years : list
        List of years to download data for
    base_dir : str or Path
        Base directory for buoy data ('/Users/jenmontano/Duke/data/buoys')
    """
 
    
    # Convert base_dir to Path object if it's a string
    base_dir = Path(base_dir)
    buoy_id = str(buoy_id)
    
    # Coefficients and their corresponding file identifiers and URLs
    coefficients = {
        'd': {
            'name': 'alpha1',
            'url': f'https://www.ndbc.noaa.gov/data/historical/swdir'
        },
        'i': {
            'name': 'alpha2',
            'url': f'https://www.ndbc.noaa.gov/data/historical/swdir2'
        },
        'j': {
            'name': 'r1',
            'url': f'https://www.ndbc.noaa.gov/data/historical/swr1'
        },
        'k': {
            'name': 'r2',
            'url': f'https://www.ndbc.noaa.gov/data/historical/swr2'
        },
        'w': {
            'name': 'c11',
            'url': f'https://www.ndbc.noaa.gov/data/historical/swden'
        }
    }
    
    # Create buoy-specific directory and its directional_spectra subdirectory
    buoy_dir = base_dir / buoy_id
    dir_spectra_dir = buoy_dir / 'directional_spectra'
    dir_spectra_dir.mkdir(parents=True, exist_ok=True)
    
    for year in years:
        for coef, info in coefficients.items():
            # Construct filename and URL
            filename = f"{buoy_id}{coef}{year}.txt.gz"
            url = f"{info['url']}/{filename}"
            
            # Local path for saving the file
            save_path = dir_spectra_dir / filename
            
            try:
                print(f"Downloading {info['name']} data for {year}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an error for bad status codes
                
                # Save the compressed file
                with open(save_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                    
                print(f"Successfully downloaded {filename}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")
                print(f"Attempted URL: {url}")
                continue

def calculate_directional_spectrum(C11, freq, alpha1, alpha2, r1, r2):
    """
    Calculate normalized directional wave spectrum.
    """
    # Create angle array (0 to 360 degrees)
    angles = np.linspace(0, 2*np.pi, 360)
    
    # Create meshgrid for frequency and direction
    angle_mesh, freq_mesh = np.meshgrid(angles, freq)
    
    # Convert angles to radians
    alpha1_rad = np.deg2rad(alpha1)
    alpha2_rad = np.deg2rad(alpha2)
    
    # Convert r1 and r2 from percentages to decimals
    r1 = np.array(r1) / 100
    r2 = np.array(r2) / 100

    # Initialize spectrum array
    E = np.zeros((len(freq), len(angles)))
    
    # Calculate directional spectrum for each frequency
    for i in range(len(freq)):
        # Compute spreading function D(f,θ)
        D = (1 / np.pi) * (0.5 + r1[i] * np.cos(angles - alpha1_rad[i]) + r2[i] * np.cos(2 * (angles - alpha2_rad[i])))
        
        # Normalize so it integrates to 1
        D = D / np.trapz(D, angles)
        
        # Ensure non-negative values
        D[D < 0] = 0
        
        # Calculate full directional spectrum
        E[i, :] = C11[i] * D
    
    return E.T, freq_mesh.T, angle_mesh.T

def read_directional_file(file_path):
    with gzip.open(file_path, 'rt') as f:
        header = f.readline().strip()
        freqs = [float(x) for x in header.split()[5:]]
        
        data, dates = [], []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                year, month, day, hour, minute = map(int, parts[:5])
                values = [float(x) for x in parts[5:]]
                if len(values) == len(freqs):
                    dates.append(datetime(year, month, day, hour, minute))
                    data.append(values)
    
    return pd.DataFrame(data, index=dates, columns=freqs)

def read_spectra_file(file_path):
    with gzip.open(file_path, 'rt') as f:
        header = f.readline().strip()
        freqs = [float(x) for x in header.split()[5:]]
        
        data, dates = [], []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                year, month, day, hour, minute = map(int, parts[:5])
                values = [float(x) for x in parts[5:]]
                if len(values) == len(freqs):
                    dates.append(datetime(year, month, day, hour, minute))
                    data.append(values)
    
    return pd.DataFrame(data, index=dates, columns=freqs)

def get_season(month):
    return {12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'}[month]

def calculate_averages(df):
    # Convert all columns to numeric, coercing errors to NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    # Add season column back
    numeric_df['season'] = df.index.month.map(get_season)
    
    # Calculate monthly average
    monthly_avg = numeric_df.drop('season', axis=1).groupby(numeric_df.index.month).mean()
    # Calculate seasonal average
    seasonal_avg = numeric_df.drop('season', axis=1).groupby(numeric_df['season']).mean()
    # Calculate annual average
    annual_avg = numeric_df.drop('season', axis=1).mean()
    
    return monthly_avg, seasonal_avg, annual_avg