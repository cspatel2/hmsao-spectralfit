
# %%
from datetime import datetime
from pytz import timezone, UTC
from itertools import chain
import os
from glob import glob
import numpy as np
import xarray as xr
from sza import solar_zenith_angle



import matplotlib.pyplot as plt
from time import perf_counter_ns
import sys
# %%
# inputs
plot = False
rootdir = 'test_data/l1a/202501'
wavelength = '5577'
sweden = {'lon': 20.41, 'lat': 67.84, 'elev': 420}  # deg, #deg, #m
wlslice = slice(557.25, 558.15)  # 1st wl slice
# wl bounds for 5577 and background. source:test_intensity_by_sum_5577.py
feature_bounds = {
    f'{wavelength}': slice(557.555, 557.715)
}
bg_bounds = {
    f'{wavelength}_bg1': slice(557.37, 557.53),
    f'{wavelength}_bg2': slice(557.87, 558.03),
}

# asset that all the bounds are the same size in nm
boundsize = None
for key, bound in chain(feature_bounds.items(), bg_bounds.items()):
    if boundsize is None:
        boundsize = np.ceil(bound.stop - bound.start)
    else:
        if boundsize != np.ceil(bound.stop - bound.start):
            raise ValueError(
                f'Bounds {key} = {bound} is not the same size as feature size {boundsize}')


# calibrated zenith angle slice
za0, dza = 0, 10
zaslice = slice(za0-dza, za0+dza)  # deg
# %% Functions
def get_date_from_fn(fn: str): return os.path.basename(fn).split('_')[1]


# %%
all_fnames = glob(os.path.join(rootdir, f'*{wavelength}*.nc'))
dates = [get_date_from_fn(f) for f in all_fnames]
dates = np.unique(dates)
print(f'Dates: {dates}')

# for each wavelength, for each day
date = dates[0]
# 0. Get file names
fnames = glob(os.path.join(rootdir, f'*{date}*{wavelength}*.nc'))
print(f'Date: {date} with {len(fnames)} files')

# 1. open files
# data
ds = xr.open_mfdataset(fnames)
# ds = ds.assign(noise=ds.noise/ds.exposure)  # convert from photons -> photons/s
# calibration data
calibfn = glob(f'photometric_calib/calib-data/*calib*{wavelength}*.nc')
calibds = xr.open_dataset(calibfn[0])

nds = ds.copy()
wl = int(wavelength)/10  # nm

# 2. protometric calib, convert from photons/s -> Rayleighs
noise = nds.intensity*(np.sqrt((calibds.conversion_error/calibds.conversion_factor)
                ** 2 + (nds.noise/nds.intensity)**2))
nds = nds.assign(intensity=ds.intensity*calibds.conversion_factor)
nds = nds.assign(noise=noise)


if plot:
    plt.figure()
    z = 0
    tds = nds.isel(tstamp=1).sel(za=z, method='nearest')
    plt.errorbar(tds.wavelength.values, tds.intensity.values, tds.noise.values)
    plt.plot(tds.wavelength.values, tds.intensity.values, color='red')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity [Rayleighs]')
    plt.title(f'PLOT 1: {wavelength} A line Brightness at za = {z} deg')

# 3. calc solar zenith angle (deg)
nds['sza'] = ('tstamp', [solar_zenith_angle(
    t, lat=sweden['lat'], lon=sweden['lon'], elevation=sweden['elev']) for t in ds.tstamp.values]
)
nds.sza.attrs = {'units': 'deg', 'long_name': 'Solar Zenith Angle'}

sza_astrodown = 90 +18  # astronomincal dawn is 18deg below horizon
# TODO: Daytime
# daysza = slice(None, sza_astrodown)  # daytime, skip for now

# nightime
# nds = nds.where(nds.sza >= sza_astrodown, drop=True)
sza = nds.sza
exposure = nds.exposure
ccdtemp = nds.ccdtemp
# remove unnecessary variables
nds = nds.drop_vars(['exposure', 'ccdtemp', 'sza'])

# 4. bin za
nds = nds.sel(wavelength=wlslice).sel(za=zaslice)


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))


zabin = int(np.ceil(1/np.mean(np.diff(nds.za.values))))
print(zabin)
coarsen = nds.coarsen(za=zabin, boundary='pad')
nds = nds.assign(noise=coarsen.reduce(rms_func).noise)  # noise is rms(noise)
nds = coarsen.sum()  # intensity is summed


if plot:
    plt.figure()
    tds = nds.isel(tstamp=1).sel(za=2, method='nearest')
    plt.errorbar(tds.wavelength.values, tds.intensity.values, tds.noise.values)
    plt.plot(tds.wavelength.values, tds.intensity.values, color='red')
    plt.title('PLOT 2')

#%%
# 5. background intensities and error by sum
# should be the dataset for average background intensity of shape (tstamp, za)
bck_ = []  # list of ds
bckds = None
for key, bound in bg_bounds.items():
    # sum all intensities in the wl slice for each za
    bds = nds.sel(wavelength=bound).sum(dim='wavelength')
    bds = bds.assign(noise=nds.sel(wavelength=bound).reduce(
        rms_func, dim='wavelength').noise)  # noise is rms(noise)
    bck_.append(bds)
#%%
bckds = xr.concat(bck_, dim='idx')
bckds = bckds.assign(noise = bckds['noise'].reduce(rms_func, dim='idx')/bckds.idx.size)
bckds = bckds.assign(intensity = bckds['intensity'].mean(dim='idx'))

#%%
# 6. line intensities and error by sum
line_ = []  # should be the dataset for line intensity of shape (tstamp, za)
for key, bound in feature_bounds.items():
    print(key)
    # sum all intensities in the wl slice for each za
    bds = nds.sel(wavelength=bound).sum(dim='wavelength')
    bds = bds.assign(noise=nds.sel(wavelength=bound).reduce(
        rms_func, dim='wavelength').noise)  # noise is rms(noise)
    # error propagation for line intensity
    bds.noise
    noise = np.sqrt(bds.noise**2 + bckds.noise**2)
    bds -= bckds.intensity  # subtract the average background
    bds = bds.assign(noise=noise)  # add the error to the line intensity
    bds = bds.rename({'intensity': key, 'noise': f'{key}_err'})
    bds[key].attrs = {'units': 'Rayleighs', 'long_name': f'{key} Brightness'}
    bds[f'{key}_err'].attrs = {'units': 'Rayleighs',
                               'long_name': f'{key} Brightness Error'}
    line_.append(bds)

bckds = bckds.rename({'intensity': 'background', 'noise': 'background_err'})
#%%
bckds['background'].attrs = {'units': 'Rayleighs',
                             'long_name': 'Mean Background Brightness'}
bckds['background_err'].attrs = {
    'units': 'Rayleighs', 'long_name': 'Mean Background Brightness Error'}

#%%
line_.append(bckds)
# line_ += bck_
# 7. combine datasets
saveds = xr.merge(line_)

saveds = saveds.assign_coords(dict(
    sza=('tstamp', sza.values),
    ccdtemp=('tstamp', ccdtemp.values),
    exposure=('tstamp', exposure.values),
))
saveds.ccdtemp.attrs = ccdtemp.attrs
saveds.exposure.attrs = exposure.attrs
saveds.sza.attrs = sza.attrs

# save dataset
savedir = f'test_data/l1b'
os.makedirs(savedir, exist_ok=True)
sub_outfname = f'hms-aorigin_{date}_{wavelength}.nc'
sub_outfpath = os.path.join(savedir, sub_outfname)
encoding = {var: {'zlib': True}
            for var in (*saveds.data_vars.keys(), *saveds.coords.keys())}
print('Saving %s...\t' % (sub_outfname), end='')
sys.stdout.flush()
tstart = perf_counter_ns()
saveds.to_netcdf(sub_outfpath, encoding=encoding)
tend = perf_counter_ns()
print(f'Done. [{(tend-tstart)*1e-9:.3f} s]')
# %%
plot = True
if plot:
    za = -5
    x = saveds.tstamp.values
    x = [datetime.fromtimestamp(t,tz= UTC) for t in x]
    y = saveds.sel(za=za, method='nearest')
    plt.errorbar(x, y[f'{wavelength}'].values,
                 y[f'{wavelength}_err'].values, label=f'{wavelength}', color='red')
   
    plt.legend(loc='best')
    plt.xlabel('Time [UTC]')
    plt.ylabel('Intensity [Rayleighs]')
    plt.title(f'Line intensities at {za} deg')


# %%
d = saveds.isel(tstamp = 2)
key = ['5577', 'background']
for k in key:
    plt.figure()
    x = d[k].values
    dx = d[f'{k}_err'].values
    y = d.za.values

    plt.errorbar(x,y,xerr=dx, yerr=None, fmt='-', label='5577', ecolor = 'red')


# %%
d['5577_err'].plot(y = 'za')
# %%
d['5577'].plot(y = 'za')
# d['background'].plot(y = 'za')

# %%
ds.noise.isel(tstamp=1).plot(y = 'za', x = 'wavelength')

# %%
