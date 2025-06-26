import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def axplot_spectrum(ax, x, y, z, vmax=None, ylim=0.49, cmap='magma', plot_center=False):
    '''
    Plots spectrum in given polar axes

    ax - input axes (polar)
    x  - spectrum directions
    y  - spectrum frequency
    z  - spectrum energy
    '''

    # fix coordinates for pcolormesh
    x1 = np.append(x, x[0])
    if plot_center:
        y1 = np.append(0, y)
    else:
        y1 = np.append(y, y[-1])
    z1 = z

    # If vmax is not provided, calculate it from the data for an optimal color scale
    if vmax is None:
        vmax = np.nanmax(np.sqrt(z1))
        if vmax == 0:  # Handle cases with no energy
            vmax = 0.1

    # polar pcolormesh
    p1 = ax.pcolormesh(
        x1, y1, np.sqrt(z1),
        vmin = 0, vmax = vmax,
    )

    # polar axes configuration
    p1.set_cmap(cmap)
    ax.set_theta_zero_location('N', offset = 0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)

    return p1


def Plot_spectrum(sp, time_ix=0, average=False, ylim=0.49, figsize = [8, 8], title=''):
    '''
    Plots superpoint spectrum at a time index or time average

    sp      - spectrum dataset
    time_ix - time index, instant to plot
    average - True to plot energy average
    title  - title of the plot
    '''

    # Detect direction coordinate name
    dir_coord_name = None
    if 'dir' in sp.coords:
        dir_coord_name = 'dir'
    elif 'direction' in sp.coords:
        dir_coord_name = 'direction'
    else:
        raise ValueError(f"Direction coordinate not found. Available coordinates: {list(sp.coords.keys())}")

    # Detect frequency coordinate name
    freq_coord_name = None
    if 'freq' in sp.coords:
        freq_coord_name = 'freq'
    elif 'frequency' in sp.coords:
        freq_coord_name = 'frequency'
    else:
        raise ValueError(f"Frequency coordinate not found. Available coordinates: {list(sp.coords.keys())}")

    # Handle both Dataset and DataArray inputs for 'sp' to make the function more flexible
    if isinstance(sp, xr.Dataset):
        # If it's a dataset, assume the data is in the 'efth' variable
        data_values = sp.efth.values
    elif isinstance(sp, xr.DataArray):
        # If it's a data array, use its values directly
        data_values = sp.values
    else:
        raise TypeError(f"Input 'sp' must be an xarray Dataset or DataArray, not {type(sp)}")

    # superpoint spectrum energy (time index or time average)
    if not average:
        z = data_values[time_ix, :, :]
        ttl = title + ' - time: {0}'.format(sp.time[time_ix].values)

    else:
        # time average
        z = np.nanmean(data_values, axis=0)
        ttl = title

    # generate figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1,1,1, projection = 'polar')
    #TODO: Check why the super point isn't doing the angle correction amd if the efth also need to be corrected
    # sp[dir_coord_name] = fix_dir(sp[dir_coord_name])
    # plot spectrum
    axplot_spectrum(ax, np.deg2rad(sp[dir_coord_name].values), sp[freq_coord_name].values, z, ylim=ylim)
    ax.set_title(ttl, fontsize=14);

    return fig