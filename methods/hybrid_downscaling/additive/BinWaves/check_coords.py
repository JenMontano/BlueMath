#!/usr/bin/env python3
import netCDF4 as nc
import sys

def check_coordinate_system(nc_file):
    """
    Check if a NetCDF file is in UTM or degrees coordinates.
    """
    try:
        # Open the NetCDF file
        with nc.Dataset(nc_file, 'r') as ds:
            print(f"\nExamining file: {nc_file}")
            print("\nAvailable variables:", list(ds.variables.keys()))
            
            # Check for coordinate variables
            coord_vars = {}
            for var_name in ['lon', 'lat', 'longitude', 'latitude', 'x', 'y']:
                if var_name in ds.variables:
                    var = ds.variables[var_name]
                    units = getattr(var, 'units', 'no units')
                    coord_vars[var_name] = units
                    print(f"\n{var_name}:")
                    print(f"  Units: {units}")
                    if hasattr(var, 'standard_name'):
                        print(f"  Standard name: {var.standard_name}")
                    if hasattr(var, 'axis'):
                        print(f"  Axis: {var.axis}")
            
            # Check for projection information
            if hasattr(ds, 'crs'):
                print("\nCRS information found:")
                for attr in ds.crs.ncattrs():
                    print(f"  {attr}: {getattr(ds.crs, attr)}")
            
            # Determine coordinate system
            print("\nCoordinate system determination:")
            if any('degree' in str(units).lower() for units in coord_vars.values()):
                print("  This file appears to be in geographic coordinates (degrees)")
            elif any('meter' in str(units).lower() for units in coord_vars.values()):
                print("  This file appears to be in UTM or another projected coordinate system (meters)")
            else:
                print("  Could not definitively determine coordinate system from units")
                print("  Please check the coordinate variable names and values manually")
    except Exception as e:
        print(f"Error examining file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_coords.py <path_to_netcdf_file>")
        sys.exit(1)
    
    nc_file = sys.argv[1]
    check_coordinate_system(nc_file) 