#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import os.path as op

import xarray as xr
import numpy as np


def fix_dir(base_dirs):
    '''
    fix csiro direction for wavespectra (from -> to)'
    '''

    new_dirs = base_dirs + 180
    new_dirs[np.where(new_dirs >= 360)] = new_dirs[np.where(new_dirs >= 360)] - 360

    return new_dirs

from typing import Dict, Tuple

def superpoint_calculation(
    stations_data: xr.DataArray,
    stations_dimension_name: str,
    sectors_for_each_station: Dict[str, Tuple[float, float]],
    deg_sup: float = 0.0,
) -> xr.DataArray:
    """
    Join multiple station spectral data for each directional sector using linear weights in overlap.
    """
    superpoint_dataarray = xr.zeros_like(
        stations_data.isel({stations_dimension_name: 0})
    )
    weight_sum = xr.zeros_like(stations_data["dir"], dtype=float)

    for station_id, (dir_min, dir_max) in sectors_for_each_station.items():
        # Calculate sector center, handle wrap-around
        if dir_min < dir_max:
            center = (dir_min + dir_max) / 2
            mask = (stations_data["dir"] >= dir_min - deg_sup) & (stations_data["dir"] <= dir_max + deg_sup)
            width = (dir_max - dir_min) / 2 + deg_sup
        else:
            # Wrap-around sector
            center = ((dir_min + dir_max + 360) / 2) % 360
            mask = (stations_data["dir"] >= dir_min - deg_sup) | (stations_data["dir"] <= dir_max + deg_sup)
            width = ((360 - dir_min) + dir_max) / 2 + deg_sup

        # Compute distance to center, handle wrap-around
        dir_vals = stations_data["dir"].values
        dist = np.abs((dir_vals - center + 180) % 360 - 180)
        # Linear weight: 1 at center, 0 at edge of overlap
        weights = np.clip(1 - dist / width, 0, 1)
        weights = xr.DataArray(weights, dims=["dir"])

        # Only apply weights within the mask
        weights = weights.where(mask, 0.0)

        station_data = stations_data.sel({stations_dimension_name: station_id}).where(mask, 0.0)
        superpoint_dataarray += station_data * weights
        weight_sum += weights

    # Normalize by total weight
    superpoint_dataarray = superpoint_dataarray / weight_sum.where(weight_sum > 0, 1.0)

    # Debug: print uncovered direction bins
    uncovered = (weight_sum == 0)
    if np.any(uncovered):
        print("Warning: Some direction bins are not covered by any sector:", stations_data["dir"].values[uncovered])

    return superpoint_dataarray

# def superpoint_calculation_auto_sectors(
#     stations_data: xr.DataArray,
#     stations_dimension_name: str,
#     sector_width: float = 7.5,
#     deg_sup: float = 0.0,
#     average: bool = True,
#     verbose: bool = True,
# ) -> xr.DataArray:
#     """
#     Automatically determine which station to use for each directional sector based on 
#     which station has the highest average energy, instead of manually defining sectors.
    
#     Parameters
#     ----------
#     stations_data : xr.DataArray
#         DataArray containing spectral data for multiple stations.
#     stations_dimension_name : str
#         Name of the dimension representing different stations in the DataArray.
#     sector_width : float, optional
#         Width of each directional sector in degrees. Default is 7.5 degrees.
#     deg_sup : float, optional
#         Degrees of superposition to create overlap between sectors. Default is 0.0.
#     average : bool, optional
#         Whether to use time-averaged energy for station selection. Default is True.
#     verbose : bool, optional
#         Whether to print progress messages. Default is True.
        
#     Returns
#     -------
#     xr.DataArray
#         A new DataArray where each point is the spectral data from the station with 
#         highest average energy for that directional sector.
#     """
#     import numpy as np
    
#     # Get direction values
#     dir_vals = stations_data["dir"].values
    
#     if verbose:
#         print(f"Available direction values: {dir_vals}")
#         print(f"Direction range: {dir_vals.min()}° to {dir_vals.max()}°")
#         print(f"Number of direction bins: {len(dir_vals)}")
    
#     # Create sectors based on sector_width
#     sectors = []
#     for i in range(0, 360, int(sector_width)):
#         start_angle = i
#         end_angle = (i + sector_width) % 360
#         sectors.append((start_angle, end_angle))
    
#     if verbose:
#         print(f"Created {len(sectors)} sectors of {sector_width}° width")
#         print(f"Processing sectors: {sectors}")
    
#     # Initialize result array
#     superpoint_dataarray = xr.zeros_like(
#         stations_data.isel({stations_dimension_name: 0})
#     )
    
#     # For each sector, find the station with highest average energy
#     for sector_idx, (dir_min, dir_max) in enumerate(sectors):
#         if verbose:
#             print(f"\nProcessing sector {sector_idx + 1}/{len(sectors)}: {dir_min}° to {dir_max}°")
        
#         # Create mask for this sector - handle wrap-around properly
#         if dir_min < dir_max:
#             # Normal sector (e.g., 0° to 45°)
#             mask = (stations_data["dir"] >= dir_min - deg_sup) & (stations_data["dir"] < dir_max + deg_sup)
#         else:
#             # Wrap-around sector (e.g., 315° to 0°)
#             mask = (stations_data["dir"] >= dir_min - deg_sup) | (stations_data["dir"] < dir_max + deg_sup)
        
#         if verbose:
#             # Check how many direction bins are in this sector
#             sector_dirs = stations_data["dir"].where(mask, drop=True)
#             print(f"  Direction bins in sector: {list(sector_dirs.values)}")
#             print(f"  Number of direction bins in sector: {len(sector_dirs)}")
        
#         # Calculate average energy for each station in this sector
#         station_energies = {}
#         for station_id in stations_data[stations_dimension_name].values:
#             station_data = stations_data.sel({stations_dimension_name: station_id})
            
#             # Apply sector mask
#             sector_data = station_data.where(mask, 0.0)
            
#             if average:
#                 # Use time-averaged energy (similar to Plot_spectrum with average=True)
#                 avg_energy = np.nanmean(sector_data.values, axis=0)
#                 # Sum over all directions and frequencies for total energy
#                 total_energy = np.nansum(avg_energy)
#             else:
#                 # Use instantaneous energy (first time step)
#                 instant_energy = sector_data.values[0, :, :]
#                 total_energy = np.nansum(instant_energy)
            
#             station_energies[station_id] = total_energy
            
#             if verbose and sector_idx == 0:  # Only show for first sector to avoid too much output
#                 print(f"    Station {station_id}: raw energy sum = {total_energy:.6f}")
        
#         # Find station with highest energy
#         best_station = max(station_energies, key=station_energies.get)
#         best_energy = station_energies[best_station]
        
#         if verbose:
#             print(f"  Station energies: {station_energies}")
#             print(f"  Selected station: {best_station} (energy: {best_energy:.6f})")
        
#         # Use the best station for this sector (no linear weighting, just direct assignment)
#         station_data = stations_data.sel({stations_dimension_name: best_station}).where(mask, 0.0)
#         superpoint_dataarray += station_data
    
#     if verbose:
#         print(f"\nCompleted processing all {len(sectors)} sectors")
    
#     return superpoint_dataarray

def superpoint_calculation_auto_sectors(
    stations_data: xr.DataArray,
    stations_dimension_name: str,
    sector_width: float = 7.5,
    deg_sup: float = 0.0,
    average: bool = True,
    verbose: bool = True,
) -> xr.DataArray:
    """
    Automatically determine which station to use for each directional sector based on 
    which station has the highest average energy, instead of manually defining sectors.
    """
    import numpy as np
    
    # Convert to numpy arrays if they're dask arrays
    if hasattr(stations_data.values, 'compute'):
        if verbose:
            print("Converting dask arrays to numpy arrays...")
        stations_data = stations_data.compute()
    
    # Get direction values
    dir_vals = stations_data["dir"].values
    
    if verbose:
        print(f"Available direction values: {dir_vals}")
        print(f"Direction range: {dir_vals.min()}° to {dir_vals.max()}°")
        print(f"Number of direction bins: {len(dir_vals)}")
    
    # Create sectors based on sector_width
    sectors = []
    for i in range(0, 360, int(sector_width)):
        start_angle = i
        end_angle = (i + sector_width) % 360
        sectors.append((start_angle, end_angle))
    
    if verbose:
        print(f"Created {len(sectors)} sectors of {sector_width}° width")
        print(f"Processing sectors: {sectors}")
    
    # Initialize result array
    superpoint_dataarray = xr.zeros_like(
        stations_data.isel({stations_dimension_name: 0})
    )
    
    # For each sector, find the station with highest average energy
    for sector_idx, (dir_min, dir_max) in enumerate(sectors):
        if verbose:
            print(f"\nProcessing sector {sector_idx + 1}/{len(sectors)}: {dir_min}° to {dir_max}°")
        
        # Create mask for this sector - handle wrap-around properly
        if dir_min < dir_max:
            # Normal sector (e.g., 0° to 45°)
            mask = (stations_data["dir"] >= dir_min - deg_sup) & (stations_data["dir"] < dir_max + deg_sup)
        else:
            # Wrap-around sector (e.g., 315° to 0°)
            mask = (stations_data["dir"] >= dir_min - deg_sup) | (stations_data["dir"] < dir_max + deg_sup)
        
        if verbose:
            # Check how many direction bins are in this sector
            sector_dirs = stations_data["dir"].where(mask, drop=True)
            print(f"  Direction bins in sector: {list(sector_dirs.values)}")
            print(f"  Number of direction bins in sector: {len(sector_dirs)}")
        
        # Calculate average energy for each station in this sector
        station_energies = {}
        for station_id in stations_data[stations_dimension_name].values:
            station_data = stations_data.sel({stations_dimension_name: station_id})
            
            # Apply sector mask
            sector_data = station_data.where(mask, 0.0)
            
            # Debug: Check what sector_data actually is
            if verbose and sector_idx == 0:
                print(f"    Debug - sector_data type: {type(sector_data)}")
                print(f"    Debug - sector_data keys: {list(sector_data.keys()) if hasattr(sector_data, 'keys') else 'no keys'}")
                if hasattr(sector_data, 'efth'):
                    print(f"    Debug - sector_data.efth type: {type(sector_data.efth)}")
                    print(f"    Debug - sector_data.efth.values type: {type(sector_data.efth.values)}")
            
            # Get the actual data - handle both DataArray and Dataset cases
            if hasattr(sector_data, 'efth'):
                # It's a Dataset, get the 'efth' variable
                data_values = sector_data.efth.values
            else:
                # It's a DataArray, get values directly
                data_values = sector_data.values
            
            # Ensure it's a numpy array
            if hasattr(data_values, 'compute'):
                data_values = data_values.compute()
            
            if average:
                # Use time-averaged energy (similar to Plot_spectrum with average=True)
                avg_energy = np.nanmean(data_values, axis=0)
                # Sum over all directions and frequencies for total energy
                total_energy = np.nansum(avg_energy)
            else:
                # Use instantaneous energy (first time step)
                instant_energy = data_values[0, :, :]
                total_energy = np.nansum(instant_energy)
            
            station_energies[station_id] = total_energy
            
            if verbose and sector_idx == 0:  # Only show for first sector to avoid too much output
                print(f"    Station {station_id}: raw energy sum = {total_energy:.6f}")
        
        # Find station with highest energy
        best_station = max(station_energies, key=station_energies.get)
        best_energy = station_energies[best_station]
        
        if verbose:
            print(f"  Station energies: {station_energies}")
            print(f"  Selected station: {best_station} (energy: {best_energy:.6f})")
        
        # Use the best station for this sector (no linear weighting, just direct assignment)
        station_data = stations_data.sel({stations_dimension_name: best_station}).where(mask, 0.0)
        superpoint_dataarray += station_data
    
    if verbose:
        print(f"\nCompleted processing all {len(sectors)} sectors")
    
    return superpoint_dataarray

def stations_superposition(p_stations, stations_id, sectors, deg_sup, st_wind_id,
                           fix_dir_bool = True, efth_to_rad = True,
                           freq_n = 'frequency', dir_n='direction', efth_n='efth',
                           wspeed_n = 'u10m', wdir_n = 'udir'):
    '''
    Join station spectral data for each sector

    p_stations   - path to stations database (stations_XXXX.nc files)
    stations_id  - list of stations ID
    sectors      - list of tuples: directional sector for each station
    deg_sup      - degrees of superposition
    st_wind_id   - station ID for wind data

    fix_dir_bool - fix csiro directions
    efth_to_rad  - transform efth to radians

    freq_n       - name of frequency dimension
    dir_n        - name of direction dimension
    efth_n       - name of efth variable
    wspeed_n     - name of wind speed variable (default: 'u10m')
    wdir_n       - name of wind direction variable (default: 'udir')
    '''
    


    # generate empty efth_all and cont variables from station dimensions
    first_st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(stations_id[0])))
    efth_all = np.full([len(first_st.time), len(first_st[freq_n]), len(first_st[dir_n]), len(stations_id)], 0.0)
    cont = np.full([len(first_st[dir_n])], 0)

    # Initialize wind and depth variables
    wsp = np.zeros(len(first_st.time))
    wdir = np.zeros(len(first_st.time))
    depth = np.zeros(len(first_st.time))
    
    # Save coordinates for later use
    time_values = first_st.time.values
    dir_values = first_st[dir_n].values
    freq_values = first_st[freq_n].values
    
    del first_st
    wind_station_found = False

    # read stations
    for s_ix, s_id in enumerate(stations_id):

        st = xr.open_dataset(op.join(p_stations, 'station_{0}.nc'.format(s_id)))


        if fix_dir_bool:
            st[dir_n] = fix_dir(st[dir_n])  # fix direction dimension

        # find station data indexes inside sector (and superposition degrees)
        if (sectors[s_ix][1] - sectors[s_ix][0]) < 0:
            d = np.where((st[dir_n].values > sectors[s_ix][0] - deg_sup) |
                         (st[dir_n].values <= sectors[s_ix][1] + deg_sup))[0]
        else:
            d = np.where((st[dir_n].values > sectors[s_ix][0] - deg_sup) &
                         (st[dir_n].values <= sectors[s_ix][1] + deg_sup))[0]

        cont[d] += 1
        efth_all[:, :, d, s_ix] = st[efth_n][:, :, d]

        # get wind data from choosen wind station
        if s_id == st_wind_id:
       
            wsp = st[wspeed_n].values
            wdir = st[wdir_n].values
            depth = np.full([len(st.time.values)], st.depth)
            wind_station_found = True

        # Clean up memory
        st.close()
        del st

    if not wind_station_found:
        raise ValueError(f"Wind station ID {st_wind_id} not found in the provided stations list: {stations_id}")

    # promediate superimposed station data (using data counter)
    efth_all = (np.sum(efth_all, axis = 3) / cont)
    if efth_to_rad:
        efth_all = efth_all * (np.pi / 180)

    # mount superpoint dataset
    super_point = xr.Dataset(
        {
            'efth': (['time','freq','dir'], efth_all),
            'Wspeed': (['time'], wsp),
            'Wdir': (['time'], wdir),
            'Depth': (['time'], depth),
        },
        coords = {
            'time': time_values,
            'dir': dir_values,
            'freq': freq_values
        }
    )

    # round time to hour
    super_point['time'] = super_point['time'].dt.round('H').values

    return super_point

def bulkparams_partitions(p_store, sp, chunks=3, wcut=0.333, msw=5, agef=1.7):
    '''
    Calculates superpoint spectra statistics and bulk parameters using wavespectra library.

    p_store - path for storage chunk datasets
    sp      - superpoint Dataset
    chunks  - split process in N chunks (split by time dimension to prevent memory issues)

    wcut    - wavespectra: wind cut
    msw     - wavespectra: max number of swells
    agef    - wavespectra: age factor
    '''

    # ensure storage folder exists
    if not op.isdir(p_store):
        os.makedirs(p_store)

    # this function needs wavespectra==3.5 
    import wavespectra

    # get split position
    pos = np.int64(len(sp.time) / chunks)

    # solve each chunk
    for p in range(chunks):

        # select current chunk  superpoint data
        if p == 0:
            sp1 = sp.isel(time = np.arange(0, pos))

        elif p == (chunks - 1):
            sp1 = sp.isel(time = np.arange(p*pos, len(sp.time)))

        else:
            sp1 = sp.isel(time = np.arange(p*pos, (p+1)*pos))

        # use wavespectra to calculate spectra partitions
        ds_part1 = sp1.spec.partition.ptm1(
            sp1.Wspeed, sp1.Wdir, sp1.Depth,
            wscut=wcut, swells=msw, agefac=agef
        )

        # clean memory
        del sp1
        gc.collect()

        # ensure time dimension is ok
        u, i = np.unique(ds_part1.time, return_index=True)
        ds_part1 = ds_part1.isel(time=i)

        # store solved chunk spectra
        nf = 'partitions_spectra_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        ds_part1 = ds_part1.to_dataset()
        store_xarray_max_compression(ds_part1, op.join(p_store, nf))

        # calculate spectral stats
        stats_part1 = ds_part1.spec.stats(['hs','tp','tm02','dpm','dspr'])

        # store spectral stats
        nf = 'partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        stats_part1.to_netcdf(op.join(p_store, nf))

        print('chunk {0}/{1} done.'.format(p+1, chunks))

        # clean memory
        del ds_part1, stats_part1
        gc.collect()


    # load processed chunks spectra stats and merge it 
    nfs = ['partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut) for p in range(chunks)]
    stats_part = xr.open_mfdataset([op.join(p_store, f) for f in nfs])

    # calculate superpoint bulk parameters
    bulk_params = sp.spec.stats(['hs','tp','tm02','dpm','dm','dspr'])

    return bulk_params, stats_part

