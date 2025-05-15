from pyproj import Transformer

# Create transformer from WGS84 to UTM Zone 18N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# List of buoy coordinates and names
coordinates = [
    # Lon, Lat, Name
    [-74.8390, 36.6120, "44100 (already converted)"],
    [-75.5930, 36.2580, "44100"],
    [-75.4210, 36.0010, "44086"],
    [-75.7140, 36.2000, "44056"],
    [-75.3300, 35.7500, "44095"],
    [-75.2850, 35.2580, "44120"]
]

print("WGS84 to UTM Zone 18N conversions:")
print("=" * 60)
print(f"{'Buoy':<10} {'WGS84 Coordinates':<30} {'UTM Coordinates (Zone 18N)'}")
print("-" * 60)

for lon, lat, name in coordinates:
    utm_x, utm_y = transformer.transform(lon, lat)
    print(f"{name:<10} {lat:.4f}°N, {lon:.4f}°W {utm_x:.2f}, {utm_y:.2f}")

print("\nCode for wrapper.py:")
print("=" * 60)
for lon, lat, name in coordinates[1:]:  # Skip the first one which is already in code
    utm_x, utm_y = transformer.transform(lon, lat)
    print(f"#{lat:.4f}, {lon:.4f} (WGS84) - buoy {name}")
    print(f"self.locations = np.vstack((self.locations, [{utm_x:.2f}, {utm_y:.2f}]))") 