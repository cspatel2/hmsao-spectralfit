#%%
from typing import Iterable
from typing import SupportsFloat as Numeric
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime
from pytz import UTC
from scipy.optimize import least_squares
from scipy.signal import correlate
#%%
def align_spectra_core(sol: np.ndarray, sky: np.ndarray, wavelength: np.ndarray, offset: float = None) -> np.ndarray:
    sol_ = np.nan_to_num(sol)
    sky_ = np.nan_to_num(sky)

    if offset is None: #automatically find a lag by using scipy.signal.correlate
        sky_norm = (sky_ - np.mean(sky_)) / np.std(sky_)
        sol_norm = (sol_ - np.mean(sol_)) / np.std(sol_)
        corr = correlate(sky_norm, sol_norm, mode='full')
        lag = np.arange(-len(sol_) + 1, len(sky_))[np.argmax(corr)]
        offset = -lag * np.mean(np.diff(wavelength))

    shifted_wl = wavelength - offset
    interp_sol = np.interp(wavelength, shifted_wl, sol_, left=np.nan, right=np.nan)
    return interp_sol

def apply_alignment(sol_ds: xr.DataArray, sky_ds: xr.DataArray, offset:float = None) -> xr.DataArray:
    wl = sky_ds["wavelength"]
    # sky_ds,sol_ds = xr.broadcast(sky_ds, sol_ds)

    aligned = xr.apply_ufunc(
        align_spectra_core,
        sol_ds['intensity'],
        sky_ds['intensity'],
        wl,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"offset": offset},
        on_missing_core_dim="copy",
    )
    return aligned

def Residual(sf:int,skyspec:Iterable[float],solspec:Iterable[float])-> Iterable[float]:return solspec - sf*skyspec # type: ignore


def solar_subtraction_core(sky:np.ndarray, sol:np.ndarray)->np.ndarray:
    # finite = np.isfinite(sky) & np.isfinite(sol)
    # res = least_squares(Residual, 1, args = (sky[finite],sol[finite]), loss= 'soft_l1')
    # sf = res.x[0]
    mask = np.isfinite(sky) & np.isfinite(sol)
    if not np.any(mask):
        return np.full_like(sky, np.nan)

    sky_masked = sky[mask]
    sol_masked = sol[mask]

    denom = np.dot(sky_masked, sky_masked)
    if denom == 0:
        sf = 0
    else:
        sf = np.dot(sky_masked, sol_masked) / denom

    return sol - sf * sky


# def apply_solar_subtraction(sol_ds: xr.Dataset, sky_ds: xr.Dataset):
#     sol_ds['tstamp'] = sky_ds.tstamp
#     # _,sol_ds = xr.broadcast(sky_ds, sol_ds)

#     subtracted = xr.apply_ufunc(
#         solar_subtraction_core,
#         sol_ds['intensity'],
#         sky_ds['intensity'],
#         input_core_dims=[["wavelength"], ["wavelength"]],
#         output_core_dims=[["wavelength"]],
#         vectorize=True,
#         dask="parallelized",
#         on_missing_core_dim="copy",
#     )

#     return subtracted

def apply_solar_subtraction(sol_ds: xr.Dataset, sky_ds: xr.Dataset):
    # _,sol_ds = xr.broadcast(sky_ds, sol_ds)
    if isinstance(sol_ds, xr.Dataset):
        sol_ds = sol_ds['intensity']
    
    if isinstance(sky_ds, xr.DataArray):
        sky_ds = sky_ds['intensity'] 

    if 'tstamp' in list(sol.sizes.keys()):
        sol_ds['tstamp'] = sky_ds.tstamp
    else:
        sol_ds = sol_ds.expand_dims(tstamp = sky_ds.tstamp.values)
    
    

    subtracted = xr.apply_ufunc(
        solar_subtraction_core,
        sol_ds,
        sky_ds,
        input_core_dims=[["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        on_missing_core_dim="copy",
    )

    return subtracted

#%%
fdir = '../../data/l1a'
date = '20250321'
win = '6300'
fnames = glob(os.path.join(fdir,f'*/*{date}*{win}*.nc'))
print(len(fnames))
sky = xr.open_dataset(fnames[3])
sky['wavelength'] = sky['wavelength'].values

fdir = 'hmsao_solspec'
fnames = glob(os.path.join(fdir,f'*/*{win}*.nc'))
print(len(fnames))
solds = xr.open_dataset(fnames[1])


#%%

soltest = solds.isel(tstamp = slice(0,50))
counts = soltest.intensity.sum('tstamp',skipna = True)
exp = soltest.exposure.sum('tstamp')
soltest['intensity'] = counts/exp
del counts, exp
#%%
skytest = sky.isel(tstamp = slice(0,50))
counts = skytest.intensity.sum('tstamp',skipna = True) 
exp = skytest.exposure.sum('tstamp')
skytest['intensity'] = counts/exp
#%%

a = apply_alignment(soltest,skytest,  -0.021)
#%%
a = a.expand_dims(tstamp=solds.tstamp.values)
#%%
sol = solds.assign(intensity = a)
#%%
sol.intensity.sel(za = 0, method = 'nearest').plot(lw = 0.5)
sky.intensity.isel(tstamp=50).sel(za = 0, method = 'nearest').plot(lw = 0.5)
# %%
sol.intensity.sel(za = 0, method = 'nearest').sel(wavelength = slice(629.5, 630.5)).plot(lw = 0.5)
sky.intensity.isel(tstamp=50).sel(za = 0, method = 'nearest').sel(wavelength = slice(629.5, 630.5)).plot(lw = 0.5)
# %%

# %%

# %%
wl = int(win)/10
dwl = 0.6
# sol['wavelength'] = sol.wavelength.values +0.02

c = apply_solar_subtraction(
    sky.sel(wavelength = slice(wl-dwl, wl+dwl)),
    sol.sel(wavelength = slice(wl-dwl, wl+dwl))
                            )
# %%

c.intensity.sel(za= 0, method = 'nearest').plot(vmin = 0)


# %%
bounds = slice(629.91, 630.07)
c.sel(za= 0, method = 'nearest').sel(wavelength = bounds).plot(vmin = 0 )
# %%

d = c.intensity.clip(min = 0)
# %%
d.sel(za= 0, method = 'nearest').plot(vmin = 0)

# %%
