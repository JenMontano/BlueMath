import numpy as np
import json

def mesh_to_geojson(locations_x, locations_y, output_file):
    """
    Convert mesh coordinates to GeoJSON format.
    
    Args:
        locations_x: 2D array of longitude coordinates (WGS84)
        locations_y: 2D array of latitude coordinates (WGS84)
        output_file: Path to output GeoJSON file
    """
    features = []
    
    # Create features directly from the geographic coordinates
    for i, (lon, lat) in enumerate(zip(locations_x.flatten(), locations_y.flatten())):
        feature = {
            "type": "Feature",
            "properties": {
                "id": f"{i:04d}",  # Format ID as 4-digit string
                "type": "BinWaves"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            }
        }
        features.append(feature)
    
    # Create the GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)

# if __name__ == "__main__":
#     # Create transformer from WGS84 to UTM
#     wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    
#     # Buoy coordinates in WGS84
#     lon = -75.2850  # longitude
#     lat = 35.2580   # latitude
    
#     # Convert WGS84 to UTM
#     utm_x, utm_y = wgs84_to_utm.transform(lon, lat)
    
#     print(f"Buoy 41120 coordinates:")
#     print(f"WGS84: {lon}°E, {lat}°N")
#     print(f"UTM: X={utm_x:.3f}, Y={utm_y:.3f}") 