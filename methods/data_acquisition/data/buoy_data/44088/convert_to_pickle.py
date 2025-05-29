import pandas as pd
import numpy as np

# Read the CSV file
input_file = 'buoy_44088_bulk_parameters.csv'
output_file = 'buoy_44088_bulk_parameters.pkl'

# Read the CSV file
df = pd.read_csv(input_file)

# Create datetime index from the time columns
df['datetime'] = pd.to_datetime({
    'year': df['YYYY'],
    'month': df['MM'],
    'day': df['DD'],
    'hour': df['hh'],
    'minute': df['mm']
})

# Check for duplicate timestamps
duplicates = df[df.duplicated('datetime', keep=False)]
if not duplicates.empty:
    print(f"\nFound {len(duplicates)} duplicate timestamps:")
    print(duplicates.sort_values('datetime').head())
    
    # Keep the first occurrence of each timestamp
    df = df.drop_duplicates('datetime', keep='first')
    print(f"\nRemoved duplicates. New shape: {df.shape}")

df = df.set_index('datetime')

# Create new dataframe with the required columns
new_df = pd.DataFrame()

# Convert values, replacing 999.0 with NaN (if present)
new_df['Hs_Buoy'] = df['WVHT'].replace(999.0, np.nan)  # Wave height
new_df['Tm_Buoy'] = df['APD'].replace(999.0, np.nan)   # Average period
new_df['Tp_Buoy'] = df['DPD'].replace(999.0, np.nan)   # Dominant period
new_df['Dir_Buoy'] = df['MWD'].replace(999, np.nan)    # Mean wave direction (int)
new_df['Spr_Buoy'] = np.nan     # Wave spread (filled with NaN)

new_df.index = df.index

# Verify index is unique
print("\nVerifying index is unique:", new_df.index.is_unique)

# Save to pickle
new_df.to_pickle(output_file)

print(f"\nConversion complete. File saved as {output_file}")
print("\nFirst few rows of the converted data:")
print(new_df.head())
print("\nLast 5 rows of the converted data:")
print(new_df.tail()) 