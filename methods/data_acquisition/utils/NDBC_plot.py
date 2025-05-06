%matplotlib inline
import requests
import gzip
import io
import pandas as pd
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
# Set font sizes
TITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 12
TEXT_SIZE = 14

def plot_wvht_timeseries(buoy_id, base_dir,colors):
    """
    Create an enhanced time series plot of wave parameters with three subplots using dots
    """
    
    buoy_dir = os.path.join(base_dir, buoy_id)
    data_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_combined.csv")
    
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

    for col in ['WVHT', 'DPD', 'APD', 'MWD']:
        # Replace missing value codes with NaN
        df[col] = df[col].replace([99.0, 999.0], np.nan)
        
        # Wave height and periods should not be 0
        if col in ['WVHT', 'DPD', 'APD']:
            df[col] = df[col].where(df[col] > 0, np.nan)
        
        # Periods should not be greater than 30 seconds
        if col in ['DPD', 'APD']:
            df[col] = df[col].where(df[col] <= 30, np.nan)
    
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
    # ax2.plot(df.loc[valid_dpd, 'datetime'], df.loc[valid_dpd, 'DPD'], 
    #          'r.', markersize=1, label='Dominant Period')
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
    
    # Save the plot
    plot_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_wave_parameters.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    # plt.close(
    
def plot_multiple_buoys_wvht(buoy_ids, colors=None):
    """
    Create three subplots comparing wave parameters from multiple buoys
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    colors = ['slateblue','lightseagreen','plum','hotpink']
    
    # Dictionary to store DataFrames for each buoy
    buoy_data = {}
    
    # First, load all the data
    for buoy_id in buoy_ids:
        base_dir = "/Users/jenmontano/Duke/data/buoys"
        buoy_dir = os.path.join(base_dir, buoy_id)
        data_file = os.path.join(buoy_dir, f"buoy_{buoy_id}_combined.csv")
        
        try:
            df = pd.read_csv(data_file)
            # Create datetime column
            df['datetime'] = pd.to_datetime(
                df['YYYY'].astype(str) + '-' +
                df['MM'].astype(str).str.zfill(2) + '-' +
                df['DD'].astype(str).str.zfill(2) + ' ' +
                df['hh'].astype(str).str.zfill(2) + ':' +
                df['mm'].astype(str).str.zfill(2),
                format='%Y-%m-%d %H:%M'
            )
            
            # Clean data
            for col in ['WVHT', 'DPD', 'APD', 'MWD']:
                # Replace missing value codes with NaN
                df[col] = df[col].replace([99.0, 999.0], np.nan)
                
                # Wave height and periods should not be 0
                if col in ['WVHT', 'DPD', 'APD']:
                    df[col] = df[col].where(df[col] > 0, np.nan)
                
                # Periods should not be greater than 30 seconds
                if col in ['DPD', 'APD']:
                    df[col] = df[col].where(df[col] <= 30, np.nan)
            
            df.set_index('datetime', inplace=True)
            
            # Resample to hourly data
            df = df.resample('h')[['WVHT', 'APD', 'MWD']].mean()
            
            # Store in dictionary
            buoy_data[buoy_id] = df
            
        except FileNotFoundError:
            print(f"No data file found for buoy {buoy_id}")
            continue

    # Set font sizes
    TITLE_SIZE = 20
    AXIS_LABEL_SIZE = 18
    TICK_LABEL_SIZE = 16
    LEGEND_SIZE = 18

    # Store line objects for the legend
    legend_lines = []
    legend_labels = []
    
    # Plot data for each buoy
    for i, buoy_id in enumerate(buoy_ids):
        if buoy_id not in buoy_data:
            continue
            
        df = buoy_data[buoy_id]
        
        # Wave Height
        valid_wvht = ~pd.isna(df['WVHT'])
        line1, = ax1.plot(df.loc[valid_wvht].index, df.loc[valid_wvht, 'WVHT'], 
                         color=colors[i], linewidth=0.5)
        
        # Average Period
        valid_apd = ~pd.isna(df['APD'])
        ax2.plot(df.loc[valid_apd].index, df.loc[valid_apd, 'APD'], 
                color=colors[i], linewidth=0.5)
        
    # Wave Direction
        valid_direction = ~pd.isna(df['MWD'])
        ax3.plot(df.loc[valid_direction].index, df.loc[valid_direction, 'MWD'], 
         '.', color=colors[i], linestyle='None', markersize=1, label=f'Buoy {buoy_id}')
        
        # Store line and label for legend
        legend_lines.append(line1)
        legend_labels.append(f'Buoy {buoy_id}')
 
        # Calculate horizontal position for each buoy's stats
        x_pos = 0.02 + (i * 0.25)  
        
        # Wave Height stats
        stats_text_hs = (f"Buoy {buoy_id}:\n"
                        f"Mean Hs: {df['WVHT'].mean():.2f} m\n"
                        f"Max Hs: {df['WVHT'].max():.2f} m\n"
                        f"Min Hs: {df['WVHT'].min():.2f} m")
        
        ax1.text(x_pos, 0.98, stats_text_hs, transform=ax1.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top', fontsize=TICK_LABEL_SIZE)
        
        # Period stats
        stats_text_tm = (f"Buoy {buoy_id}:\n"
                        f"Mean Tm: {df['APD'].mean():.2f} s\n"
                        f"Max Tm: {df['APD'].max():.2f} s\n"
                        f"Min Tm: {df['APD'].min():.2f} s")
        
        ax2.text(x_pos, 0.98, stats_text_tm, transform=ax2.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top', fontsize=TICK_LABEL_SIZE)
        
        # Direction stats
        stats_text_dm = (f"Buoy {buoy_id}:\n"
                        f"Mean Dir: {df['MWD'].mean():.2f}°")
        
        ax3.text(x_pos, 0.98, stats_text_dm, transform=ax3.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                verticalalignment='top', fontsize=TICK_LABEL_SIZE)

    ax1.set_ylabel('Wave Height (m)', fontsize=AXIS_LABEL_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    ax2.set_ylabel('Average Period (s)', fontsize=AXIS_LABEL_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    ax3.set_ylabel('Wave Direction (°)', fontsize=AXIS_LABEL_SIZE)
    ax3.set_xlabel('Date', fontsize=AXIS_LABEL_SIZE)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax3.set_ylim(0, 360)
    
    # Create legend outside the first subplot
    legend = fig.legend(legend_lines, legend_labels,
                       loc='upper center', 
                       bbox_to_anchor=(0.5, 1.0),
                       ncol=len(buoy_ids),  # Place legend items horizontally
                       fontsize=LEGEND_SIZE)
    
    # Make legend lines thicker
    for legline in legend.get_lines():
        legline.set_linewidth(2.0)  # Increased line width in legend
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for legend at top

        # Synchronize x-axesf
    date_min = min(df.index.min() for df in buoy_data.values())
    date_max = max(df.index.max() for df in buoy_data.values())
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(date_min, date_max)
    
    # Save the plot
    plot_file = os.path.join(base_dir, f"buoys_wave_parameters_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')

def plot_yearly_averages(buoy_id, years, base_dir):
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
    
    # Process each year
    for year in years:
        file_path = os.path.join(base_dir, buoy_id, f"buoy_{buoy_id}_spectra_{year}.csv")
        
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
                # print(f"Processed year {year}")
                
            except AttributeError as e:
                # print(f"Error processing year {year}")
                # print(f"Index type: {type(df.index)}")
                continue
            
        except (FileNotFoundError, ValueError) as e:
            # print(f"Error with year {year}")
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
    return fig

def plot_seasonal_averages(buoy_id, years, base_dir):
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
    
    # Process each year
    for year in years:
        file_path = os.path.join(base_dir, buoy_id, f"buoy_{buoy_id}_spectra_{year}.csv")
        
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
            
            # print(f"Processed year {year}")
            
        except (FileNotFoundError, ValueError) as e:
            # print(f"Error with year {year}")
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
            info['ax'].set_ylabel('Spectral Density (m²/Hz)', fontsize=AXIS_LABEL_SIZE      )
            info['ax'].grid(True, alpha=0.2)
            info['ax'].legend(fontsize=LEGEND_SIZE)
            info['ax'].tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.suptitle(f'Seasonal Wave Spectra - Buoy {buoy_id}', fontsize=TITLE_SIZE, y=1.02)
    plt.tight_layout()
    
    return fig


def plot_monthly_averages(buoy_id, years, base_dir):
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
    
    # Process each year
    for year in years:
        file_path = os.path.join(base_dir, buoy_id, f"buoy_{buoy_id}_spectra_{year}.csv")
        
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
            
            # print(f"Processed year {year}")
            
        except (FileNotFoundError, ValueError) as e:
            # print(f"Error with year {year}: {e}")
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
    
    plt.suptitle(f'Monthly Wave Spectra Averages - Buoy {buoy_id}', y=1.02, fontsize=TITLE_SIZE)
    plt.tight_layout()
    
    return fig

def plot_monthly_spectra_comparison(buoy_ids, years, base_dir):
    """
    Plot monthly wave spectra averages for multiple buoys on the same plot for comparison
    """
    # Create a 4x3 grid for all 12 months
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()
    
    # Define month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)
    
    # Use a colormap to get distinct colors for each buoy
    colors = plt.cm.tab10(np.linspace(0, 1, len(buoy_ids)))
    
    # Initialize global max value for consistent y-axis across all plots
    global_max_value = 0
    
    # Process each buoy
    for i, buoy_id in enumerate(buoy_ids):
        # Keep track of all monthly averages for this buoy
        buoy_averages = {month+1: [] for month in range(12)}
        
        # Process each year for this buoy
        for year in years:
            file_path = os.path.join(base_dir, buoy_id, f"buoy_{buoy_id}_spectra_{year}.csv")
            
            try:
                # Read data
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Check if index is actually a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(f"Warning: Index for {buoy_id} - {year} is not a DatetimeIndex. Converting...")
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
                            
                            # Plot this year's monthly average with the buoy's color but lighter
                            axes[month-1].plot(common_freqs, interp_avg, 
                                             color=colors[i], alpha=0.1, linewidth=2)
                            
                            # Store for overall average
                            buoy_averages[month].append(interp_avg)
                    except AttributeError as e:
                        # print(f"Error processing month {month} for {buoy_id} - {year}: {e}")
                        continue
                
            except (FileNotFoundError, ValueError) as e:
                # Silently continue if file not found
                continue
        
        # Calculate and update the global maximum value
        for month in range(1, 13):
            if buoy_averages[month]:
                month_avg = np.mean(buoy_averages[month], axis=0)
                global_max_value = max(global_max_value, np.max(month_avg))
        
        # Plot overall averages for each month for this buoy
        for month in range(1, 13):
            if buoy_averages[month]:
                # Calculate overall average
                month_avg = np.mean(buoy_averages[month], axis=0)
                
                # Plot overall average
                axes[month-1].plot(common_freqs, month_avg,
                                 color=colors[i], linewidth=3,
                                 label=f'Buoy {buoy_id} ({len(buoy_averages[month])} years)')
        
        print(f"Processed buoy {buoy_id}")
    
    # Add a small buffer to the max value and set consistent y-axis
    y_max = global_max_value * 1.5
    
    # Customize each subplot
    for month in range(1, 13):
        axes[month-1].set_title(month_names[month-1], fontsize=12)
        axes[month-1].set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)
        axes[month-1].set_ylabel('Spectral Density (m²/Hz)', fontsize= AXIS_LABEL_SIZE)
        axes[month-1].grid(True, alpha=0.2)
        axes[month-1].legend(fontsize=LEGEND_SIZE)
        # Set consistent y-axis limits
        axes[month-1].set_ylim(0, y_max)
    
    plt.suptitle(f'Monthly Wave Spectra Comparison Between {buoy_ids}', y=1.02, fontsize=TITLE_SIZE)
    plt.tight_layout()
    
    return fig

def plot_spectra_comparison(buoy_ids, years, base_dir):
    """
    Plot wave spectra averages for multiple buoys on the same plot for comparison
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create a common frequency grid (0.01 to 0.5 Hz with 100 points)
    common_freqs = np.linspace(0.01, 0.5, 100)
    
    # Use a colormap to get distinct colors for each buoy
    colors = plt.cm.tab10(np.linspace(0, 1, len(buoy_ids)))
    
    # Process each buoy
    for i, buoy_id in enumerate(buoy_ids):
        # Keep track of all yearly averages for this buoy
        yearly_averages = []
        valid_years = []
        
        # Process each year for this buoy
        for year in years:
            file_path = os.path.join(base_dir, buoy_id, f"buoy_{buoy_id}_spectra_{year}.csv")
            
            try:
                # Read data
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Check if index is actually a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(f"Warning: Index for {buoy_id} - {year} is not a DatetimeIndex. Converting...")
                    df.index = pd.to_datetime(df.index)
                
                # Convert column names to numeric frequencies
                orig_freqs = pd.to_numeric(df.columns)
                
                # Calculate yearly average
                try:
                    avg = df.mean()
                    
                    # Interpolate to common frequency grid using numpy
                    interp_avg = np.interp(common_freqs, orig_freqs, avg.values, 
                                          left=0, right=0)
                    
                    # Plot this year's average with the buoy's color but lighter
                    ax.plot(common_freqs, interp_avg, 
                           color=colors[i], alpha=0.2, linewidth=2)
                    
                    # Store for overall average
                    yearly_averages.append(interp_avg)
                    valid_years.append(year)
                    
                except AttributeError as e:
                    print(f"Error processing {buoy_id} - {year}: {e}")
                    continue
                
            except (FileNotFoundError, ValueError) as e:
                # Silently continue if file not found
                continue
        
        # Plot overall average for this buoy if we have data
        if yearly_averages:
            # Calculate overall average
            overall_avg = np.mean(yearly_averages, axis=0)
            
            # Plot overall average
            ax.plot(common_freqs, overall_avg,
                   color=colors[i], linewidth=3,
                   label=f'Buoy {buoy_id} ({len(yearly_averages)} years)')
            
            print(f"Processed buoy {buoy_id} with {len(yearly_averages)} valid years")
    
    # Customize plot
    ax.set_title(f'Wave Spectra Comparison Between {buoy_ids}', fontsize=TITLE_SIZE)
    ax.set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('Spectral Density (m²/Hz)', fontsize=AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    # Optional: Set y-axis to log scale for better visualization of spectral shape
    # ax.set_yscale('log')
    
    plt.tight_layout()
    return fig