from pyproj import Transformer

# Create transformer from geographic to UTM coordinates (UTM zone 18N for the Carolinas)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Define the corners of your region
corners = [
    (-76.5, 35),  # Southwest
    (-74.5, 35),  # Southeast
    (-76.5, 37),  # Northwest
    (-74.5, 37)   # Northeast
]

# Convert each corner to UTM coordinates
utm_corners = [transformer.transform(lon, lat) for lon, lat in corners]

# Print the results
print("UTM coordinates for your region:")
print(f"X (East-West) range: {min(x for x, y in utm_corners):.2f} to {max(x for x, y in utm_corners):.2f}")
print(f"Y (North-South) range: {min(y for x, y in utm_corners):.2f} to {max(y for x, y in utm_corners):.2f}")

# Convert buoy location
buoy_lon, buoy_lat = -74.8390, 36.6120
buoy_x, buoy_y = transformer.transform(buoy_lon, buoy_lat)
print("\nBuoy location in UTM coordinates:")
print(f"X: {buoy_x:.2f}")
print(f"Y: {buoy_y:.2f}") 