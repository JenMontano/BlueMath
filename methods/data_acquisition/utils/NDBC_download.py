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
import logging

def download_bulk_parameters(buoy_id, year):
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
                df = pd.read_csv(io.StringIO('\n'.join(data)), sep=r'\s+',
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

def process_buoy(buoy_id, base_dir, start_year=1960, end_year=None):
    """
    Combine all available data for a buoy between start_year and end_year
    """
    if end_year is None:
        end_year = datetime.now().year
        
    all_data = []

    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory for specific buoy if it doesn't exist
    buoy_dir = os.path.join(base_dir, buoy_id)
    os.makedirs(buoy_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        df = download_bulk_parameters(buoy_id, year)
        
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
        output_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_bulk_parameters.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        return combined_df
    else:
        print(f"No data found for buoy {buoy_id}")
        return None
    
def download_wave_spectra(buoy_id, base_dir, start_year=1970, end_year=None):
    """
    Download wave spectra data for a specific buoy, saving each year separately due to the discrepancy in the data format.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directory for specific buoy if it doesn't exist
    buoy_dir = os.path.join(base_dir, buoy_id, 'wave_spectra')
    os.makedirs(buoy_dir, exist_ok=True)
    
    if end_year is None:
        end_year = datetime.now().year
    
    for year in range(start_year, end_year + 1):
        url = f"https://www.ndbc.noaa.gov/data/historical/swden/{buoy_id}w{year}.txt.gz"
        
        try:
            # Read the data
            df = pd.read_csv(url, compression='gzip', sep=r'\s+',
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

def download_directional_spectra(buoy_id, base_dir, start_year, end_year=None):
    """
    Download directional wave spectra data for specified buoy and years.
    
    Parameters:
    -----------
    buoy_id : str
        Buoy identifier (e.g., '41001')
    years : list
        List of years to download data for
    base_dir : str or Path
        Base directory for buoy data 
    """
 
    # Convert base_dir to Path object if it's a string
    base_dir = Path(base_dir)
    buoy_id = str(buoy_id)
    
    # Coefficients to reconstruct the spectrum and their corresponding file identifiers and URLs
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
    
    if end_year is None:
        end_year = datetime.now().year
    
    for year in range(start_year, end_year + 1):
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
        D = D / np.trapezoid(D, angles)
        
        # Ensure non-negative values
        D[D < 0] = 0
        
        # Calculate full directional spectrum
        E[i, :] = C11[i] * D
    
    return E.T, freq_mesh.T, angle_mesh.T

def read_directional_file(file_path):
    """Read directional spectra file and return DataFrame with datetime index and frequency columns"""
    logging.info(f"Reading file: {file_path}")
    try:
        with gzip.open(file_path, 'rt') as f:
            # Read header lines until we find the frequencies
            header_lines = []
            while True:
                line = f.readline().strip()
                if not line.startswith('#'):
                    break
                header_lines.append(line)
            
            # Combine header lines and parse frequencies
            header = ' '.join(header_lines)
            try:
                # Skip the first 5 columns (YY MM DD hh mm)
                freqs = [float(x) for x in header.split()[5:]]
                logging.info(f"Found {len(freqs)} frequencies")
            except (ValueError, IndexError) as e:
                logging.error(f"Error parsing frequencies: {e}")
                return pd.DataFrame()
            
            # Read data
            data = []
            dates = []
            # Process the first line (which was read in the header loop)
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    year, month, day, hour, minute = map(int, parts[:5])
                    values = [float(x) for x in parts[5:]]
                    if len(values) == len(freqs):
                        dates.append(datetime(year, month, day, hour, minute))
                        data.append(values)
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing line: {e}")
            
            # Read remaining lines
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        values = [float(x) for x in parts[5:]]
                        if len(values) == len(freqs):
                            dates.append(datetime(year, month, day, hour, minute))
                            data.append(values)
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing line: {e}")
                        continue
            
            if not data:
                logging.warning("No valid data points found in file")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, index=dates, columns=freqs)
            logging.info(f"Created DataFrame with shape: {df.shape}")
            return df
            
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()

def read_spectra_file(file_path):
    """Read the C11 spectra file"""
    logging.info(f"Reading file: {file_path}")
    try:
        with gzip.open(file_path, 'rt') as f:
            # Read header lines until we find the frequencies
            header_lines = []
            while True:
                line = f.readline().strip()
                if not line.startswith('#'):
                    break
                header_lines.append(line)
            
            # Combine header lines and parse frequencies
            header = ' '.join(header_lines)
            try:
                # Skip the first 5 columns (YY MM DD hh mm)
                freqs = [float(x) for x in header.split()[5:]]
                logging.info(f"Found {len(freqs)} frequencies")
            except (ValueError, IndexError) as e:
                logging.error(f"Error parsing frequencies: {e}")
                return pd.DataFrame()
            
            # Read data
            data = []
            dates = []
            # Process the first line (which was read in the header loop)
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    year, month, day, hour, minute = map(int, parts[:5])
                    values = [float(x) for x in parts[5:]]
                    if len(values) == len(freqs):
                        dates.append(datetime(year, month, day, hour, minute))
                        data.append(values)
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing line: {e}")
            
            # Read remaining lines
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year, month, day, hour, minute = map(int, parts[:5])
                        values = [float(x) for x in parts[5:]]
                        if len(values) == len(freqs):
                            dates.append(datetime(year, month, day, hour, minute))
                            data.append(values)
                    except (ValueError, IndexError) as e:
                        logging.error(f"Error parsing line: {e}")
                        continue
            
            if not data:
                logging.warning("No valid data points found in file")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, index=dates, columns=freqs)
            logging.info(f"Created DataFrame with shape: {df.shape}")
            return df
            
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()

def get_season(month):
    """Return season based on month"""
    return {12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'}[month]



def plot_monthly_directional_spectra(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot with 12 subplots showing monthly directional spectra
    
    Parameters:
    -----------
    alpha1_df, alpha2_df, r1_df, r2_df, c11_df : pandas.DataFrame
        DataFrames containing the processed data
    buoy_id : str
        Buoy identifier
    start_date : str
        Start date in format 'YYYY-MM-DD HH:MM'
    end_date : str
        End date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create output directory
    output_dir = base_dir / buoy_id / 'processed' / f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process monthly data
    plt.figure(figsize=(20, 15))
    for month in range(1, 13):
        try:
            # Get monthly averages
            r1_monthly = r1_df.groupby(r1_df.index.month).mean().loc[month].values
            r2_monthly = r2_df.groupby(r2_df.index.month).mean().loc[month].values
            alpha1_monthly = alpha1_df.groupby(alpha1_df.index.month).mean().loc[month].values
            alpha2_monthly = alpha2_df.groupby(alpha2_df.index.month).mean().loc[month].values
            c11_monthly = c11_df.groupby(c11_df.index.month).mean().loc[month].values
            
            # Calculate directional spectrum
            E, freq_mesh, angle_mesh = calculate_directional_spectrum(
                c11_monthly, freqs, alpha1_monthly, alpha2_monthly, r1_monthly, r2_monthly
            )
            
            # Plot
            ax = plt.subplot(4, 3, month, projection='polar')
            pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap='magma')
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            plt.title(f'Month {month}')
            
        except KeyError:
            logging.warning(f"No data available for month {month}")
            continue
    
    plt.suptitle(f'Monthly Directional Spectra - {buoy_id} ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    plt.tight_layout()
    plt.savefig(output_dir / 'monthly_directional_spectra.png')
    plt.close()

def plot_seasonal_directional_spectra(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot with 4 subplots showing seasonal directional spectra
    
    Parameters:
    -----------
    alpha1_df, alpha2_df, r1_df, r2_df, c11_df : pandas.DataFrame
        DataFrames containing the processed data
    buoy_id : str
        Buoy identifier
    start_date : str
        Start date in format 'YYYY-MM-DD HH:MM'
    end_date : str
        End date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create output directory
    output_dir = base_dir / buoy_id / 'processed' / f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process seasonal data
    plt.figure(figsize=(20, 15))
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    for i, season in enumerate(seasons, 1):
        try:
            # Get seasonal averages
            r1_seasonal = r1_df.groupby(r1_df.index.month.map(get_season)).mean().loc[season].values
            r2_seasonal = r2_df.groupby(r2_df.index.month.map(get_season)).mean().loc[season].values
            alpha1_seasonal = alpha1_df.groupby(alpha1_df.index.month.map(get_season)).mean().loc[season].values
            alpha2_seasonal = alpha2_df.groupby(alpha2_df.index.month.map(get_season)).mean().loc[season].values
            c11_seasonal = c11_df.groupby(c11_df.index.month.map(get_season)).mean().loc[season].values
            
            # Calculate directional spectrum
            E, freq_mesh, angle_mesh = calculate_directional_spectrum(
                c11_seasonal, freqs, alpha1_seasonal, alpha2_seasonal, r1_seasonal, r2_seasonal
            )
            
            # Plot
            ax = plt.subplot(2, 2, i, projection='polar')
            pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap='magma')
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location('N')
            plt.title(f'Season {season}')
            
        except KeyError:
            logging.warning(f"No data available for season {season}")
            continue
    
    plt.suptitle(f'Seasonal Directional Spectra - {buoy_id} ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_directional_spectra.png')
    plt.close()

def plot_annual_directional_spectrum(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot showing annual directional spectrum
    
    Parameters:
    -----------
    alpha1_df, alpha2_df, r1_df, r2_df, c11_df : pandas.DataFrame
        DataFrames containing the processed data
    buoy_id : str
        Buoy identifier
    start_date : str
        Start date in format 'YYYY-MM-DD HH:MM'
    end_date : str
        End date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create output directory
    output_dir = base_dir / buoy_id / 'processed' / f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process annual data
    r1_annual = r1_df.mean().values
    r2_annual = r2_df.mean().values
    alpha1_annual = alpha1_df.mean().values
    alpha2_annual = alpha2_df.mean().values
    c11_annual = c11_df.mean().values
    
    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11_annual, freqs, alpha1_annual, alpha2_annual, r1_annual, r2_annual
    )
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap='magma')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.colorbar(pcm, label='Energy Density (m²/Hz/rad)')
    plt.title(f'Annual Directional Spectrum - {buoy_id} ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    plt.savefig(output_dir / 'annual_directional_spectrum.png')
    plt.close()

def plot_specific_date_directional_spectrum(alpha1, alpha2, r1, r2, c11, freqs, buoy_id, date_str, base_dir):
    """
    Create a single plot showing directional spectrum for a specific date
    
    Parameters:
    -----------
    alpha1, alpha2, r1, r2, c11 : numpy.ndarray
        Arrays containing the processed data
    freqs : numpy.ndarray
        Array of frequencies
    buoy_id : str
        Buoy identifier
    date_str : str
        Date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert date to datetime
    target_date = pd.to_datetime(date_str)
    
    # Calculate directional spectrum
    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11, freqs, alpha1, alpha2, r1, r2
    )
    
    # Create output directory
    output_dir = base_dir / buoy_id / 'processed' / 'specific_dates'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap='magma')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.colorbar(pcm, label='Energy Density (m²/Hz/rad)')
    plt.title(f'Directional Spectrum - {buoy_id} at {date_str}')
    plt.savefig(output_dir / f'directional_spectrum_{target_date.strftime("%Y%m%d_%H%M")}.png')
    plt.close()

def read_and_process_data(buoy_id, start_date, end_date, base_dir):
    """
    Read and process data for a date range
    
    Parameters:
    -----------
    buoy_id : str
        Buoy identifier
    start_date : str
        Start date in format 'YYYY-MM-DD HH:MM'
    end_date : str
        End date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    
    Returns:
    --------
    tuple
        (alpha1_df, alpha2_df, r1_df, r2_df, c11_df) if successful, None if failed
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get all years in the range
    years = range(start_dt.year, end_dt.year + 1)
    
    # Read all data files
    all_alpha1 = []
    all_alpha2 = []
    all_r1 = []
    all_r2 = []
    all_c11 = []
    
    for year in years:
        dir_spectra_dir = base_dir / buoy_id / 'directional_spectra'
        alpha1_path = dir_spectra_dir / f'{buoy_id}d{year}.txt.gz'
        alpha2_path = dir_spectra_dir / f'{buoy_id}i{year}.txt.gz'
        r1_path = dir_spectra_dir / f'{buoy_id}j{year}.txt.gz'
        r2_path = dir_spectra_dir / f'{buoy_id}k{year}.txt.gz'
        c11_path = dir_spectra_dir / f'{buoy_id}w{year}.txt.gz'
        
        # Read files
        alpha1_df = read_directional_file(alpha1_path)
        alpha2_df = read_directional_file(alpha2_path)
        r1_df = read_directional_file(r1_path)
        r2_df = read_directional_file(r2_path)
        c11_df = read_spectra_file(c11_path)
        
        if not any(df.empty for df in [alpha1_df, alpha2_df, r1_df, r2_df, c11_df]):
            all_alpha1.append(alpha1_df)
            all_alpha2.append(alpha2_df)
            all_r1.append(r1_df)
            all_r2.append(r2_df)
            all_c11.append(c11_df)
    
    if not all_alpha1:
        logging.error("No valid data found for the specified date range")
        return None
    
    # Combine all years
    alpha1_df = pd.concat(all_alpha1)
    alpha2_df = pd.concat(all_alpha2)
    r1_df = pd.concat(all_r1)
    r2_df = pd.concat(all_r2)
    c11_df = pd.concat(all_c11)
    
    # Filter by date range
    mask = (alpha1_df.index >= start_dt) & (alpha1_df.index <= end_dt)
    alpha1_df = alpha1_df[mask]
    alpha2_df = alpha2_df[mask]
    r1_df = r1_df[mask]
    r2_df = r2_df[mask]
    c11_df = c11_df[mask]
    
    # Convert all dataframes to numeric
    r1_df = r1_df.apply(pd.to_numeric, errors='coerce')
    r2_df = r2_df.apply(pd.to_numeric, errors='coerce')
    alpha1_df = alpha1_df.apply(pd.to_numeric, errors='coerce')
    alpha2_df = alpha2_df.apply(pd.to_numeric, errors='coerce')
    c11_df = c11_df.apply(pd.to_numeric, errors='coerce')
    
    return alpha1_df, alpha2_df, r1_df, r2_df, c11_df

# def read_and_process_specific_date(buoy_id, date_str, base_dir):
    """
    Read and process data for a specific date
    
    Parameters:
    -----------
    buoy_id : str
        Buoy identifier
    date_str : str
        Date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory for data storage
    
    Returns:
    --------
    tuple
        (alpha1, alpha2, r1, r2, c11, freqs) if successful, None if failed
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert date to datetime
    target_date = pd.to_datetime(date_str)
    
    # Create paths for files
    dir_spectra_dir = base_dir / buoy_id / 'directional_spectra'
    alpha1_path = dir_spectra_dir / f'{buoy_id}d{target_date.year}.txt.gz'
    alpha2_path = dir_spectra_dir / f'{buoy_id}i{target_date.year}.txt.gz'
    r1_path = dir_spectra_dir / f'{buoy_id}j{target_date.year}.txt.gz'
    r2_path = dir_spectra_dir / f'{buoy_id}k{target_date.year}.txt.gz'
    c11_path = dir_spectra_dir / f'{buoy_id}w{target_date.year}.txt.gz'
    
    # Read files
    alpha1_df = read_directional_file(alpha1_path)
    alpha2_df = read_directional_file(alpha2_path)
    r1_df = read_directional_file(r1_path)
    r2_df = read_directional_file(r2_path)
    c11_df = read_spectra_file(c11_path)
    
    if any(df.empty for df in [alpha1_df, alpha2_df, r1_df, r2_df, c11_df]):
        logging.error(f"Failed to read one or more data files for {buoy_id} {target_date.year}")
        return None
    
    # Get data for the specific date
    try:
        alpha1 = alpha1_df.loc[target_date].values
        alpha2 = alpha2_df.loc[target_date].values
        r1 = r1_df.loc[target_date].values
        r2 = r2_df.loc[target_date].values
        c11 = c11_df.loc[target_date].values
        freqs = np.array([float(col) for col in r1_df.columns])
    except KeyError:
        logging.error(f"No data available for {buoy_id} at {date_str}")
        return None
    
    return alpha1, alpha2, r1, r2, c11, freqs

def read_specific_date_data(buoy_id, date_str, base_dir):
    """
    Read data for a specific date from existing downloaded files
    
    Parameters:
    -----------
    buoy_id : str
        Buoy identifier
    date_str : str
        Date in format 'YYYY-MM-DD HH:MM'
    base_dir : str or Path
        Base directory where data files are stored
    
    Returns:
    --------
    tuple
        (alpha1, alpha2, r1, r2, c11, freqs) if successful, None if failed
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert date to datetime
    target_date = pd.to_datetime(date_str)
    
    # Create paths for files
    dir_spectra_dir = base_dir / buoy_id / 'directional_spectra'
    alpha1_path = dir_spectra_dir / f'{buoy_id}d{target_date.year}.txt.gz'
    alpha2_path = dir_spectra_dir / f'{buoy_id}i{target_date.year}.txt.gz'
    r1_path = dir_spectra_dir / f'{buoy_id}j{target_date.year}.txt.gz'
    r2_path = dir_spectra_dir / f'{buoy_id}k{target_date.year}.txt.gz'
    c11_path = dir_spectra_dir / f'{buoy_id}w{target_date.year}.txt.gz'
    
    # Read files
    alpha1_df = read_directional_file(alpha1_path)
    alpha2_df = read_directional_file(alpha2_path)
    r1_df = read_directional_file(r1_path)
    r2_df = read_directional_file(r2_path)
    c11_df = read_spectra_file(c11_path)
    
    if any(df.empty for df in [alpha1_df, alpha2_df, r1_df, r2_df, c11_df]):
        logging.error(f"Failed to read one or more data files for {buoy_id} {target_date.year}")
        return None
    
    # Get data for the specific date
    try:
        alpha1 = alpha1_df.loc[target_date].values
        alpha2 = alpha2_df.loc[target_date].values
        r1 = r1_df.loc[target_date].values
        r2 = r2_df.loc[target_date].values
        c11 = c11_df.loc[target_date].values
        freqs = np.array([float(col) for col in r1_df.columns])
    except KeyError:
        logging.error(f"No data available for {buoy_id} at {date_str}")
        return None
    
    return alpha1, alpha2, r1, r2, c11, freqs