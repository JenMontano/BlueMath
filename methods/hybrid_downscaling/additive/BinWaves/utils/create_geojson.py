import numpy as np
import json
from pyproj import Transformer

def utm_to_wgs84(x, y):
    """Convert UTM coordinates to WGS84 (longitude/latitude)"""
    transformer = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat

def create_geojson(locations_file, output_file):
    # Read the locations file
    locations = np.loadtxt(locations_file)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Convert each point and add to features
    for i, (x, y) in enumerate(locations):
        # Convert UTM to WGS84
        lon, lat = utm_to_wgs84(x, y)
        
        # Create feature
        feature = {
            "type": "Feature",
            "properties": {
                "id": f"{i:04d}",
                "type": "BinWaves"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
        geojson["features"].append(feature)
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)

