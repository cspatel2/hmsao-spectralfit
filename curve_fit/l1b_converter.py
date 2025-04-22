# %%
import os
from glob import glob
import numpy as np
import xarray as xr
from sza import solar_zenith_angle
from test_curvefit import spectral_fit
from tqdm import tqdm

from uncertainties import unumpy as un
from uncertainties import ufloat, UFloat
import matplotlib.pyplot as plt
# %%


def get_date_from_fn(fn: str):
    return os.path.basename(fn).split('_')[1]


# %%
wl = '6300'
fdir = 'test_data'
all_fnames = glob(os.path.join(fdir, f'*{wl}*.nc'))
dates = [get_date_from_fn(f) for f in all_fnames]
dates = np.unique(dates)
print(len(dates))

# for each wavelength, for each day
date = dates[0]
# TODO: 0. Get file names
fnames = glob(os.path.join(fdir, f'*{date}*{wl}*.nc'))
len(fnames)

# TODO: 1. open file
ds = xr.open_mfdataset(fnames)

calibfn = glob(f'photometric_calib/calib-data/*calib*{wl}*.nc')
calibds = xr.open_dataset(calibfn[0])


nds = ds.copy()
wl = int(wl)/10
print(wl)
dwl = .75  # nm
wlslice = slice(wl-dwl, wl+dwl)
za = 0
dza = 10  # deg
zaslice = slice(za - dza, za+dza)
# for each tstamp
out = []

#TODO: add timer
# t_binsize = 2 # bins
# nds = nds.coarsen(tstamp=t_binsize, boundary='pad').sum()
tstamps = nds.tstamp.values
for tidx, ts in enumerate(tstamps):
    
    imgda = nds.isel(tstamp=tidx).intensity

    #2. protometric calib
    imgda.values *= calibds.conversion_factor.values  # countrate -> rayleights

    #3. Calc SZA (only fit the nighttime data)
    longitude, latitude, elevation = 20.41, 67.84, 420  # sweden deg,deg,m
    sza = solar_zenith_angle(
        ts, lat=latitude, lon=longitude, elevation=elevation)
    sza_astrodown = 108  # astronomincal dawn is 18deg below horizon
    if sza <= sza_astrodown:
        continue  # daytime, skip for now
        # TODO: its day time, so scale and subtract before fitting
    else:
        pass  # nighttime, so let it process
    
    #4. bin za
    # select useable data
    imgda= imgda.sel(wavelength=wlslice).sel(za=zaslice)
    # bin along za
    za_binsize = 10  # bins
    

    imgda = imgda.coarsen(za=za_binsize, boundary='pad').sum()
    
    #5. fit each za bin
    out_ = []
    zaarr = imgda.za.values
    pbar = tqdm(zaarr, desc= f'{tidx:02}/{len(tstamps):02}')
    for zidx, za in enumerate(pbar):
        lineds = imgda.isel(za=zidx)
    
        # drop all the zeros
        lineds = lineds.where(lineds > 20, drop=True)

        # fit
        x = lineds.wavelength.values
        y = lineds.values
        del lineds
        #p = [x0,a0,a1,a3,c,a,w]
        p0 = [np.mean(x), np.mean(y), (y[-1] - y[0]) / (x[-1] - x[0]), 0,
            wl, np.max(y) - np.min(y), .05]
        p_low = [0, 0, -np.inf, -np.inf, wl - 0.02, 0, 0]
        p_high = [np.inf, np.inf, np.inf, np.inf, wl+0.02, np.inf, dwl/2]
        #res = [feature, feature_std, bg, residual, cwl, cwl_std]
        res = spectral_fit(x=x, y=y, p0=p0, plow=p_low, phigh=p_high, plot=False, calc = True) #line_intensity, cwl, bg, res
        out_.append(list(res))
    out.append(out_)

# TODO: 5. Save dataset

out = np.array(out)
saveds = xr.Dataset(
    data_vars=dict(
        intensity = (('tstamp','za'),np.asarray(out[:,:,0])),
        int_stdev = (('tstamp','za'),np.asarray(out[:,:,1])),
        bg = (('tstamp','za'),np.asarray(out[:,:,2])),
        res = (('tstamp','za'),np.asarray(out[:,:,3])),
        cwl = (('tstamp','za'),np.asarray(out[:,:,4])),
        cwl_stdev = (('tstamp','za'),np.asarray(out[:,:,5])),
    ),
    coords=dict(
        tstamp = (('tstamp'), tstamps),
        za = (('za'),zaarr)
    )
)
# TODO: add attrs for each variable and the ds
wl = int(wl*10)
saveds.to_netcdf(f'l1b_{date}_{wl}.nc')
#TODO: add encoding
# %%
tds = xr.open_dataset('l1b_20250122_6300.nc')
# %%
# tds = saveds
# # %%
plt.errorbar(tds.za.values, tds.intensity.isel(tstamp=0).values, yerr=tds.int_stdev.isel(tstamp=0).values)
# %%
tds.intensity.plot(x='tstamp',vmax = 500)
# %%
from datetime import datetime 
datetime.fromtimestamp(tds.tstamp.values[70])

# %%
