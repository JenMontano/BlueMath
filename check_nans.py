import pickle
import pandas as pd
import numpy as np

# Read the pickle file
file_path = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/outputs/buoy_41025_bulk_parameters.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Print total number of rows
print("Total number of rows:", len(data))

# Check NaN values in each column
print("\nNumber of NaN values in each column:")
print(data.isna().sum())

# Check for infinite values
print("\nNumber of infinite values in each column:")
print(np.isinf(data.select_dtypes(include=np.number)).sum())

# Print percentage of valid data in each column
print("\nPercentage of valid data in each column:")
print(100 * (1 - data.isna().sum() / len(data)))

# Print time range of data
print("\nTime range of data:")
print("Start:", data.index.min())
print("End:", data.index.max())

# Print a few examples of rows with NaN values
print("\nExample rows with NaN values:")
print(data[data.isna().any(axis=1)].head()) 