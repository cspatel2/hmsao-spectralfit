# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import Iterable, List
from typing import SupportsFloat as Numeric
from glob import glob
import os
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from datetime import datetime 
from pytz import UTC
import matplotlib.dates as mdates
# %%
def normalize_intensity(x: np.ndarray) -> np.ndarray:
    """Normalize the intensity of a spectrum to the range [0, 1]."""
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def Residual_wl_drift(wlshift, wlsol, wlblue):
    return wlblue - wlsol + wlshift


def calc_wldrift_core(sol: np.ndarray, sky: np.ndarray, solwl:np.ndarray, skywl:np.ndarray, bounds:tuple[Numeric,Numeric]) -> Numeric:
    """Calculate the wavelength drift between solar and sky spectra.
    This function finds the peaks in both spectra, sorts them, and uses least squares to find the wavelength shift.
    """
    sol = normalize_intensity(sol)
    sky = normalize_intensity(sky)
    dist = 10 # distance between peaks
    solmask = np.where((solwl >= bounds[0]) & (solwl <= bounds[1]))
    skymask = np.where((skywl >= bounds[0]) & (skywl <= bounds[1]))
    brights = [sol, sky] #original input spectra 
    masks = [solmask, skymask] #masks for the bounds for each spectrum
    mbrights = [sol[solmask], sky[skymask]] #masked spectra
    peaks = [find_peaks(-mb, distance = dist)[0] for mb  in mbrights] #troughs in the masked spectra
    solpeak,skypeak = peaks[0], peaks[1]
    min_idx = np.argmin([len(i) for i in peaks]) # pick spectrum with min number of peaks
    mask = [0,1]


    if len(peaks[min_idx]) < 1:
        print('im here')
        return np.nan
    else:
        try:
            m = peaks[min_idx]
            b = mbrights[min_idx][m]
            db = np.diff(b)
            if len(peaks[min_idx]) == 1  or ((len(peaks[min_idx]) >=2) and (np.min(np.abs(db)) >0.2)):
                # If only one peak is found, smoothen the spectra and find peaks again
                print(f'{len(peaks[min_idx])} im here')
                sol = gaussian_filter(sol, 10)
                sky = gaussian_filter(sky, 10)
                solpeak,_ = find_peaks(-sol, distance=dist)
                skypeak,_ = find_peaks(-sky, distance=dist)
                print(len(solpeak), len(skypeak))
                mask = [0]
            
            if len(solpeak) < 1 or len(skypeak) < 1: return np.nan
            solsort = sorted(zip(sol[solpeak], solwl[solpeak]))
            skysort = sorted(zip(sky[skypeak], skywl[skypeak]))

            _, sollam = zip(*[solsort[i] for i in mask])
            _, skylam = zip(*[skysort[i] for i in mask])

            result = least_squares(Residual_wl_drift, 0, args=(
                np.array(sollam), np.array(skylam)))
        except: 
            plt.figure()
            plt.plot(solwl, sol, color='orange', label='solar')
            plt.plot(solwl[solpeak], sol[solpeak], 'o', color='orange')
            plt.plot(skywl, sky, color='tab:blue', label='sky')
            plt.plot(skywl[skypeak], sky[skypeak], 'o', color='tab:blue')
    return result.x[0]
    
    


def avg_solarspec(solspec: xr.Dataset) -> xr.DataArray:
    """Average the solar spectrum."""
    t = solspec['exposure'].sum().values
    br = solspec['intensity'].sum(dim='tstamp') / t
    return br

# %%
skyfn = '/home/charmi/locsststor/proc/hmsao_1a/202502/hmsao-l1a_20250220_6300[1].nc'

solfn = '/home/charmi/locsststor/proc/hmsao_solspec/202506/sol-l1a_20250612_6300[1].nc'
#%%
skyds = xr.open_dataset(skyfn)
ZABINS = int(1/np.mean(np.diff(skyds.za.values)))
skyds = skyds.coarsen(za =ZABINS, boundary='trim').mean()

solds = xr.open_dataset(solfn) 
solds = solds.coarsen(za = ZABINS, boundary='trim').mean()
sol = avg_solarspec(solds) #intensity (za:1809, wavelength: 793)
#%%
sol= sol.expand_dims(tstamp=skyds.tstamp.values)


# %%
ts = slice(0,-1)
bounds = slice(630.00, 630.3)


a = xr.apply_ufunc(
    calc_wldrift_core,
    sol.isel(tstamp=ts).sel(wavelength=bounds),
    skyds["intensity"].isel(tstamp=ts).sel(wavelength=bounds),
    sol.isel(tstamp=ts).sel(wavelength=bounds)['wavelength'],
    skyds.isel(tstamp=ts).sel(wavelength=bounds)["wavelength"],
    input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"],["wavelength"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    kwargs={'bounds': (bounds.start, bounds.stop)},
    on_missing_core_dim="copy",
    dask_gufunc_kwargs={'allow_rechunk': True}

)
# %%
a['tstamp'] = [datetime.fromtimestamp(t, UTC) for t in a.tstamp.values  ]
#%%
c = skyds.intensity.isel(tstamp = 30)
plt.figure()
c.plot()
plt.title(datetime.fromtimestamp(int(c.tstamp.values), UTC))

fig,ax = plt.subplots()
myFmt = mdates.DateFormatter("%H:%M")  
ax.xaxis.set_major_formatter(myFmt)
a.plot(y = 'za', ax = ax)
ax.axvline(datetime.fromtimestamp(int(c.tstamp.values), UTC), color='red', linestyle='--', label='sky exposure')



# %%
bounds = slice(630, 630.3)
za = -0.5786846230604041
sk = skyds.intensity.isel(tstamp=0).sel(za = za, method = 'nearest').sel(wavelength = bounds)
sk.values = normalize_intensity(sk.values)
so = sol.isel(tstamp=0).sel(za = za, method = 'nearest').sel(wavelength = bounds)
so.values = normalize_intensity(so.values)

drift  = a.isel(tstamp=0).sel(za = za, method = 'nearest')
print(f'Drift: {drift.values:.3f} nm')
# %%
sk.plot(color = 'tab:blue')
so.plot(color = 'orange')
# %%
so['wavelength'] = so['wavelength'] + drift.values
# %%
sk.plot(color = 'tab:blue')
so.plot(color = 'orange')
#%%
dist = 15
peaksk,_ = find_peaks(-sk.values, distance=dist)
peaksso,_ = find_peaks(-so.values, distance=dist)
sortedsk = sorted(zip(sk.values[peaksk], sk.wavelength.values[peaksk]))
sortedso = sorted(zip(so.values[peaksso], so.wavelength.values[peaksso]))
isk, skylam = zip(*sortedsk)
iso, sollam = zip(*sortedso)
#%%
sk.plot(color = 'tab:blue')
so.plot(color = 'orange')
plt.scatter(skylam,isk, color = 'tab:blue', label = 'sky peaks')
plt.scatter(sollam,iso, color = 'orange', label = 'solar peaks')
# %%
from scipy.ndimage import gaussian_filter
#%%

k = gaussian_filter(sk.values,10) 
o = gaussian_filter(so.values,10)
# %%
plt.plot(sk.wavelength.values, k, color = 'tab:blue', label = 'sky')
plt.plot(so.wavelength.values, o, color = 'orange', label = 'solar')

# %%
peaksk,_ = find_peaks(-k, distance=dist)
peaksso,_ = find_peaks(-o, distance=dist)

# %%
sortsk = sorted(zip(k[peaksk], sk.wavelength.values[peaksk]))
sortso = sorted(zip(o[peaksso], so.wavelength.values[peaksso]))

# %%
mask = [0]
isk, skylam = zip(*[sortsk[i] for i in mask])
iso, sollam = zip(*[sortso[i] for i in mask])
# %%
plt.plot(sk.wavelength.values, k, color = 'tab:blue', label = 'sky')
plt.scatter(skylam,isk, color = 'tab:blue')
plt.plot(so.wavelength.values, o, color = 'orange', label = 'solar')
plt.scatter(sollam,iso, color = 'orange')

# %%
dif = np.array(skylam)-np.array(sollam)
# %%
plt.plot(sk.wavelength.values, k, color = 'tab:blue', label = 'sky')
plt.plot(so.wavelength.values+dif[0], o, color = 'orange', label = 'solar')
# %%
plt.plot(sk.wavelength.values, sk.values, color = 'tab:blue', label = 'sky')
plt.plot(so.wavelength.values+dif[0], so.values, color = 'orange', label = 'solar')
# %%
