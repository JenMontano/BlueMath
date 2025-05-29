import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils.NDBC_download import calculate_directional_spectrum
# Set font sizes
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 12
TEXT_SIZE = 14

def plot_bulk_timeseries(buoy_id, base_dir):
    """
    Create an enhanced time series plot of wave parameters with three subplots using dots
    """
    colors = ['plum']
    buoy_dir = os.path.join(base_dir, buoy_id)
    data_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_bulk_parameters.csv")
    
    # Read the combined CSV file
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"No data file found for buoy {buoy_id} at {data_file}")
        return
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(
        df['YYYY'].astype(str) + '-' +
        df['MM'].astype(str).str.zfill(2) + '-' +
        df['DD'].astype(str).str.zfill(2) + ' ' +
        df['hh'].astype(str).str.zfill(2) + ':' +
        df['mm'].astype(str).str.zfill(2),
        format='%Y-%m-%d %H:%M'
    )

    # for col in ['WVHT', 'DPD', 'APD', 'MWD']:
    #     # Replace missing value codes with NaN
    #     df[col] = df[col].replace([99.0, 999.0], np.nan)
        
    #     # Wave height and periods should not be 0
    #     if col in ['WVHT', 'DPD', 'APD']:
    #         df[col] = df[col].where(df[col] > 0, np.nan)
        
    #     # Periods should not be greater than 30 seconds
    #     if col in ['DPD', 'APD']:
    #         df[col] = df[col].where(df[col] <= 30, np.nan)
    
    # Create the plot 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    # Plot 1: Wave Height
    valid_wvht = ~pd.isna(df['WVHT'])
    ax1.plot(df.loc[valid_wvht, 'datetime'], df.loc[valid_wvht, 'WVHT'], 
            color= colors[0],label='Wave Height')
    ax1.set_ylabel('Wave Height (m)', fontsize=AXIS_LABEL_SIZE)
    ax1.set_title(f'Buoy {buoy_id}', fontsize= TITLE_SIZE, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)

    # Plot 2: Wave Periods
    valid_dpd = ~pd.isna(df['DPD'])
    valid_apd = ~pd.isna(df['APD'])
    ax2.plot(df.loc[valid_apd, 'datetime'], df.loc[valid_apd, 'APD'], 
            color= colors[0], markersize=1, label='Average Period')
    ax2.set_ylabel('Wave Period (s)', fontsize=AXIS_LABEL_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)

    # Plot 3: Wave Direction
    valid_direction = ~pd.isna(df['MWD'])
    ax3.plot(df.loc[valid_direction, 'datetime'], df.loc[valid_direction, 'MWD'], 
             '.', color= colors[0], markersize=1, label='Mean Wave Direction')
    ax3.set_ylabel('Wave Direction (°)', fontsize=AXIS_LABEL_SIZE)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    # Set y-axis limits
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax3.set_ylim(0, 360)

    # Align x-axes
    date_min = df['datetime'].min()
    date_max = df['datetime'].max()
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(date_min, date_max)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Add statistics text box
    stats_textHs = (
                 f"Mean Hs: {df['WVHT'].mean():.2f} m\n"
                 f"Max Hs: {df['WVHT'].max():.2f} m\n"
                 f"Min Hs: {df['WVHT'].min():.2f} m\n"
                 )
    stats_textTm = (
                 f"Mean Tm: {df['APD'].mean():.2f} s\n"
                 f"Max Tm: {df['APD'].max():.2f} s\n"
                 f"Min Tm: {df['APD'].min():.2f} s\n"
                 )
    stats_textDm = (
                 f"Mean Dirm: {df['APD'].mean():.2f} °\n"
                 )
    
    ax1.text(0.02, 0.98, stats_textHs, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top', fontsize=TEXT_SIZE)
    ax2.text(0.02, 0.98, stats_textTm, transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top', fontsize=TEXT_SIZE)
    ax3.text(0.02, 0.98, stats_textDm, transform=ax3.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top', fontsize=TEXT_SIZE)

    # Adjust layout
    plt.tight_layout()
    
    # Create Figures directory and save the plot
    output_dir = Path(base_dir) / buoy_id / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"buoy_{buoy_id}_wave_parameters.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_yearly_averages(buoy_id, base_dir, start_year, end_year=None):
    """
    Plot yearly wave spectra averages with overall average
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)
    
    # Keep track of all yearly averages
    yearly_averages = []
    valid_years = []
    
    if end_year is None:
        end_year = datetime.now().year
    
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(base_dir, buoy_id, 'wave_spectra', f"buoy_{buoy_id}_spectra_{year}.csv")
        
        try:
            # Read data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check if index is actually a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index for {year} is not a DatetimeIndex. Converting...")
                df.index = pd.to_datetime(df.index)
            
            # Convert column names to numeric frequencies
            orig_freqs = pd.to_numeric(df.columns)
            
            # Calculate yearly average
            try:
                avg = df.mean()
                
                # Interpolate to common frequency grid using numpy
                interp_avg = np.interp(common_freqs, orig_freqs, avg.values, 
                                      left=0, right=0)
                
                # Plot this year's average
                ax.plot(common_freqs, interp_avg, 
                       color='lightblue', alpha=0.4, linewidth=2)
                
                # Store for overall average
                yearly_averages.append(interp_avg)
                valid_years.append(year)
                
            except AttributeError as e:
                continue
            
        except (FileNotFoundError, ValueError) as e:
            continue
    
    # Plot overall average if we have data
    if yearly_averages:
        # Calculate overall average
        overall_avg = np.mean(yearly_averages, axis=0)
        
        # Plot overall average
        ax.plot(common_freqs, overall_avg,
               color='navy', linewidth=3,
               label=f'Overall Average ({len(yearly_averages)} years)')
        
        # Add year range to title
        if valid_years:
            year_range = f"{min(valid_years)}-{max(valid_years)}"
        else:
            year_range = "No valid years"
        
        # Customize plot
        ax.set_title(f'Yearly Wave Spectra - Buoy {buoy_id} ({year_range})', fontsize=TITLE_SIZE)
        ax.set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel('Spectral Density (m²/Hz)', fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=12)
    
    plt.tight_layout()

    # Create Figures directory and save the plot
    output_dir = Path(base_dir) / buoy_id / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"yearly_spectra_{buoy_id}_{start_year}_{end_year}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def plot_seasonal_averages(buoy_id, base_dir, start_year, end_year=None):
    """
    Plot seasonal averages with interpolation to handle different frequency grids
    Using only numpy for interpolation
    """
    # Create subplots for each season
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    seasons = {
        'DJF': {'months': [12, 1, 2], 'color': 'blue', 'ax': axes[0]},
        'MAM': {'months': [3, 4, 5], 'color': 'green', 'ax': axes[1]},
        'JJA': {'months': [6, 7, 8], 'color': 'red', 'ax': axes[2]},
        'SON': {'months': [9, 10, 11], 'color': 'orange', 'ax': axes[3]}
    }
    
    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)
    
    # Keep track of all interpolated averages for each season
    all_averages = {season: [] for season in seasons.keys()}
    
    if end_year is None:
        end_year = datetime.now().year
    
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(base_dir, buoy_id,'wave_spectra', f"buoy_{buoy_id}_spectra_{year}.csv")
        
        try:
            # Read data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check if index is actually a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index for {year} is not a DatetimeIndex. Converting...")
                df.index = pd.to_datetime(df.index)
            
            # Convert column names to numeric frequencies
            orig_freqs = pd.to_numeric(df.columns)
            
            # Process each season
            for season, info in seasons.items():
                try:
                    season_data = df[df.index.month.isin(info['months'])]
                    if not season_data.empty:
                        # Calculate seasonal average
                        avg = season_data.mean()
                        
                        # Interpolate to common frequency grid using numpy
                        interp_avg = np.interp(common_freqs, orig_freqs, avg.values, 
                                              left=0, right=0)
                        
                        # Plot this year's seasonal average
                        info['ax'].plot(common_freqs, interp_avg, 
                                      color=info['color'], alpha=0.2, linewidth=2)
                        
                        # Store for overall average
                        all_averages[season].append(interp_avg)
                except AttributeError as e:
                    print(f"Error processing season {season} for year {year}")
                    print(f"Index type: {type(df.index)}")
                    continue
            
        except (FileNotFoundError, ValueError) as e:
            continue
    
    # Plot overall averages for each season
    for season, info in seasons.items():
        if all_averages[season]:
            # Calculate overall average
            season_avg = np.mean(all_averages[season], axis=0)
            
            # Plot overall average
            info['ax'].plot(common_freqs, season_avg,
                          color=info['color'], linewidth=3,
                          label=f'Overall Average ({len(all_averages[season])} years)')
            
            # Customize plot
            info['ax'].set_title(season, fontsize=13)
            info['ax'].set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)
            info['ax'].set_ylabel('Spectral Density (m²/Hz)', fontsize=AXIS_LABEL_SIZE)
            info['ax'].grid(True, alpha=0.2)
            info['ax'].legend(fontsize=LEGEND_SIZE)
            info['ax'].tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.suptitle(f'Seasonal Wave Spectra - Buoy {buoy_id} ({start_year} to {end_year})', fontsize=TITLE_SIZE, y=1.02)
    plt.tight_layout()
    
    # Create Figures directory and save the plot
    output_dir = Path(base_dir) / buoy_id / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"seasonal_spectra_{buoy_id}_{start_year}_{end_year}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def plot_monthly_averages(buoy_id, base_dir, start_year, end_year=None):
    """
    Plot monthly wave spectra averages across all years
    """
    # Create a 4x3 grid for all 12 months
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    # Define month names and colors
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Use a colormap to get distinct colors for each month
    colors = plt.cm.hsv(np.linspace(0, 1, 12))
    
    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)
    
    # Keep track of all interpolated averages for each month
    all_averages = {month+1: [] for month in range(12)}
    
    if end_year is None:
        end_year = datetime.now().year
    
    # Process each year
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(base_dir, buoy_id,'wave_spectra', f"buoy_{buoy_id}_spectra_{year}.csv")
        
        try:
            # Read data
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check if index is actually a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"Warning: Index for {year} is not a DatetimeIndex. Converting...")
                df.index = pd.to_datetime(df.index)
            
            # Convert column names to numeric frequencies
            orig_freqs = pd.to_numeric(df.columns)
            
            # Process each month
            for month in range(1, 13):
                try:
                    month_data = df[df.index.month == month]
                    if not month_data.empty:
                        # Calculate monthly average
                        avg = month_data.mean()
                        
                        # Interpolate to common frequency grid using numpy
                        interp_avg = np.interp(common_freqs, orig_freqs, avg.values, 
                                              left=0, right=0)
                        
                        # Plot this year's monthly average
                        axes[month-1].plot(common_freqs, interp_avg, 
                                         color=colors[month-1], alpha=0.2, linewidth=2)
                        
                        # Store for overall average
                        all_averages[month].append(interp_avg)
                except AttributeError as e:
                    print(f"Error processing month {month} for year {year}: {e}")
                    print(f"Index type: {type(df.index)}")
                    continue
            
        except (FileNotFoundError, ValueError) as e:
            continue
    
    # Find the global maximum value for consistent y-axis
    max_value = 0
    for month in range(1, 13):
        if all_averages[month]:
            month_avg = np.mean(all_averages[month], axis=0)
            max_value = max(max_value, np.max(month_avg))
    
    # Add a small buffer to the max value
    y_max = max_value * 1.5
    
    # Plot overall averages for each month
    for month in range(1, 13):
        if all_averages[month]:
            # Calculate overall average
            month_avg = np.mean(all_averages[month], axis=0)
            
            # Plot overall average
            axes[month-1].plot(common_freqs, month_avg,
                             color=colors[month-1], linewidth=3,
                             label=f'Average ({len(all_averages[month])} years)')
            
            # Customize plot
            axes[month-1].set_title(month_names[month-1], fontsize=TITLE_SIZE)
            axes[month-1].set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)
            axes[month-1].set_ylabel('Spectral Density (m²/Hz)', fontsize=AXIS_LABEL_SIZE)
            axes[month-1].grid(True, alpha=0.2)
            axes[month-1].legend(fontsize=LEGEND_SIZE)
            axes[month-1].tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
            
            # Set consistent y-axis limits
            axes[month-1].set_ylim(0, y_max)
    
    plt.suptitle(f'Monthly Wave Spectra Averages - Buoy {buoy_id} ({start_year} to {end_year})', y=1.02, fontsize=TITLE_SIZE)
    plt.tight_layout()
    
    # Create Figures directory and save the plot
    output_dir = Path(base_dir) / buoy_id / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"monthly_spectra_{buoy_id}_{start_year}_{end_year}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

# def calculate_directional_spectrum(C11, freq, alpha1, alpha2, r1, r2):
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

def get_season(month):
    """Return season based on month"""
    return {12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'}[month]

def plot_specific_date_directional_spectrum(alpha1, alpha2, r1, r2, c11, freqs, buoy_id, date_str, base_dir):
    """
    Create a single plot showing directional spectrum for a specific date
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert date to datetime
    target_date = pd.to_datetime(date_str)
    
    # Calculate directional spectrum
    E, freq_mesh, angle_mesh = calculate_directional_spectrum(
        c11, freqs, alpha1, alpha2, r1, r2
    )
    
    # Create Figures directory
    output_dir = base_dir / buoy_id / 'Figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='polar')
    pcm = ax.pcolormesh(angle_mesh, freq_mesh, E, cmap='magma')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.colorbar(pcm, label='Energy Density (m²/Hz/rad)')
    plt.title(f'Directional Spectrum - {buoy_id} at {date_str}')
    
    # Save the plot with date in filename
    plt.savefig(output_dir / f'directional_spectrum_{target_date.strftime("%Y%m%d_%H%M")}.png')
    plt.show()

def plot_monthly_directional_spectra(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot with 12 subplots showing monthly directional spectra
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create Figures directory
    output_dir = base_dir / buoy_id / 'Figures'
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
    
    # Save the plot with date range in filename
    plt.savefig(output_dir / f'monthly_directional_spectra_{start_dt.strftime("%Y%m%d")}_{end_dt.strftime("%Y%m%d")}.png')
    plt.show()

def plot_seasonal_directional_spectra(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot with 4 subplots showing seasonal directional spectra
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create Figures directory
    output_dir = base_dir / buoy_id / 'Figures'
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
    
    # Save the plot with date range in filename
    plt.savefig(output_dir / f'seasonal_directional_spectra_{start_dt.strftime("%Y%m%d")}_{end_dt.strftime("%Y%m%d")}.png')
    plt.show()

def plot_average_directional_spectrum(alpha1_df, alpha2_df, r1_df, r2_df, c11_df, buoy_id, start_date, end_date, base_dir):
    """
    Create a single plot showing annual directional spectrum
    """
    # Convert base_dir to Path if it's a string
    base_dir = Path(base_dir)
    
    # Convert dates to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Get frequencies from the columns
    freqs = np.array([float(col) for col in r1_df.columns])
    
    # Create Figures directory
    output_dir = base_dir / buoy_id / 'Figures'
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
    plt.title(f'Average Directional Spectrum - {buoy_id} ({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})')
    
    # Save the plot with date range in filename
    plt.savefig(output_dir / f'average_directional_spectrum_{start_dt.strftime("%Y%m%d")}_{end_dt.strftime("%Y%m%d")}.png')
    plt.show()