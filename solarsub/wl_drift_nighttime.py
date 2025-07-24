#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from glob import glob
import os
from datetime import datetime
from pytz import UTC
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.optimize import curve_fit
# %%
fdir = '/run/user/1000/gvfs/smb-share:server=locsststor.local,share=processed/hmsao_1a'
infodict = {'6300': slice(629.9, 630.2),
            '5577': slice(557.6, 557.9),} 

#calc drift for each window, for nighttime data
for win,bounds in infodict.items():
    fnames = glob(os.path.join(fdir, f'*/*{win}*.nc'))
    dates = np.unique([os.path.basename(fn).split('_')[1] for fn in fnames])
    dates.sort()
    
    dslist = []
    dmask = np.arange(0,len(dates))
    dt = []
    for date in tqdm(dates[dmask]):
        #get data
        skyfn = glob(os.path.join(fdir, f'*/*{date}*{win}*.nc'))
        skyfn.sort()
        skyds = xr.open_mfdataset(skyfn)

        #pick time range: last hour of the day
        date = datetime.strptime(date, '%Y%m%d')
        date = date.astimezone(UTC)
        dt.append(float(date.timestamp()))
        start = date.replace(hour=23)
        end = start.replace(minute=59, second=59)
        
        #slice for time, za and wavelength
        skyds = skyds.intensity.sel(tstamp=slice(start.timestamp(), end.timestamp()))
        skyds = skyds.sel(za=slice(-10, 10), wavelength=bounds)
        img = skyds.mean(dim='tstamp')

        drift = img.idxmax(dim='wavelength') # max wavelength at each za
        dslist.append(drift)

        # l = img.sel(za=10, method='nearest')
        # peakwl = l.idxmax(dim='wavelength').values
        # l.plot()
        # plt.axvline(peakwl, color='red', linestyle='--', label='Peak Wavelength')
        # plt.legennd()

    ds = xr.concat(dslist, dim='tstamp')
    ds['tstamp'] = dt
    ds['tstamp'].attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    ds = ds.rename('peak_wl')
    ds.attrs['units'] = 'nm'
    ds.attrs['long_name'] = 'Peak Wavelength'
    ds.attrs['description'] = 'Peak wavelength for the last hour of the day'
    #save dataset
    print(f'Saving dataset...')
    ds.to_netcdf(f'wldrift_{win}.nc', mode='w', format='NETCDF4', engine='netcdf4')
    print(f'Saved to wldrift_{win}.nc')
###################################################################################
#%%
ds = xr.open_dataset('wldrift_6300.nc')

# %%
## PLOT RED AND GREEN WAVELENGTH DRIFT
infodict = {'6300': 'red', '5577': 'green'}
fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=300,
                        tight_layout=True, squeeze=True)
for win, color in infodict.items():
    ds = xr.open_dataset(f'wldrift_{win}.nc')
    ds['peak_wl'] = ds.peak_wl.isel(tstamp = 3).mean('za').values- ds['peak_wl'] 
    ds.peak_wl.mean(dim= 'za').plot(label=f'PeakWL - {int(win)/10:0.1f}', color=color, ax = ax)

ax2 = ax.twinx()
df = pd.read_csv('kiruna_historical_temps.csv')
ds = df.to_xarray()
ds['datetime'].values = np.array(ds['datetime'], dtype='datetime64')

ax2.plot(ds['datetime'], ds['temp'], color = 'blue', ls = '--', lw = 0.5)
ax2.set_ylabel('Temperature [°C]', color='blue')
ax2.tick_params(axis='y', colors='blue')
# lines = ax.get_lines() + ax2.get_lines()
# ax.legend(lines, [line.get_label() for line in lines], loc='best')
ax.legend(loc = 'best')
ax.set_xlim(np.datetime64('2025-01-15'), np.datetime64('2025-05-30'))
ax.set_xlabel('Time')
ax.set_ylabel('Δλ [nm]')

# %% 
# TEMP DATA FROM KIRUNA, SWEDEN DOWNLOADED FROM VISUALCROSSING.COM
df = pd.read_csv('kiruna_historical_temps.csv')
df = df.set_index('datetime')
tempds = df.to_xarray()
tempds['datetime'] = np.array(tempds['datetime'], dtype='datetime64')

# %% 
# FITTING WAVELENGTH DRIFT 
win = '6300'
COLOR = 'red'
ds = xr.open_dataset(f'wldrift_{win}.nc')
ds['peak_wl'] =ds['peak_wl'] = ds.peak_wl.isel(tstamp = -2).mean('za').values- ds['peak_wl'] 

# nds = ds.sel(tstamp=slice('2025-01-20', '2025-03-15')) #FIRST HALF
nds = ds.sel(tstamp=slice('2025-03-15','2025-06-01' )) #SECOND HALF
# nds = ds #NO SELECTION

#get data
x = nds['tstamp'].values
y = nds['peak_wl'].sel(za = slice(-10,10)).mean(dim='za').values
# y = nds['peak_wl'].sel(za=-10, method='nearest').values
xx = np.arange(len(y))

#define linear function for fitting
def linear(x, m, b):
    return m * x + b    
plate_scale = 0.0038 # nm/pix
# popt, pcov = curve_fit(linear, xx, y,sigma = np.full_like(y, 3.9e-3), nan_policy='omit')

#fit the data to linear function
popt, pcov = curve_fit(linear, xx, y,sigma = np.nanstd(np.array(y)) ,nan_policy='omit')
err = np.sqrt(np.diag(pcov))

#plot fitted line, data, and uncertainty
plt.figure(figsize=(6.4, 4.8), dpi=300)
nds['peak_wl'].sel(za = slice(-10,10)).mean(dim='za').plot.scatter(label='Peak Wavelength', color=COLOR, s=1)
plt.plot(x, linear(xx, *popt), label=f'Fitted Line \n m = {popt[0]:.0e} ± {err[0]:.0e} nm/day \n b = {popt[1]:.0e} ± {err[1]:.0e} nm \n total drift: {len(y)*popt[0]:0.2f} nm ({len(y)*popt[0]/plate_scale:0.0f} pix) in {len(y)} days ', color='blue')
u = linear(xx, *(popt+err))
l = linear(xx, *(popt-err))
plt.fill_between(x, l, u, color='blue', alpha=0.1)
plt.legend()
plt.ylabel('Δλ [nm]')
# %%
