import pickle
import pandas as pd

# Read the pickle file
file_path = '/lustre/geocean/WORK/users/jen/BlueMath/methods/hybrid_downscaling/additive/BinWaves/outputs/buoy_41025_bulk_parameters.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Data type:", type(data))
print("\nVariables in the dataset:")
if hasattr(data, 'columns'):
    print(data.columns)
elif isinstance(data, dict):
    print(list(data.keys()))
else:
    print("Data structure:", data.shape if hasattr(data, 'shape') else "unclear")

print("\nFirst few entries:")
if hasattr(data, 'head'):
    print(data.head())
elif isinstance(data, dict):
    for key, value in list(data.items())[:5]:
        print(f"{key}: {value}")
else:
    print(data[:5] if hasattr(data, '__getitem__') else data) 