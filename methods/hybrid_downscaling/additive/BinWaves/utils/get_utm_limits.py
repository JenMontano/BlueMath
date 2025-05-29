from pyproj import Transformer

# Create transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# WGS84 limits
min_lon, max_lon = 76.8000,-75.2000
min_lat, max_lat = 34.4000, 35.6000

# Convert to UTM
min_x, min_y = transformer.transform(min_lon, min_lat)
max_x, max_y = transformer.transform(max_lon, max_lat)

print(f"UTM limits:")
print(f"X: {min_x:.6f} to {max_x:.6f}")
print(f"Y: {min_y:.6f} to {max_y:.6f}") 