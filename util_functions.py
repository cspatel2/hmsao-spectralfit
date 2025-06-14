#%%
import csv
import numpy as np
import xarray as xr
from typing import Iterable



def get_bounds_from_csv(csv_path: str, wavelength: str):
    """
    Get the bounds from the csv file.
    """
    with open(csv_path) as f:
        bg_dict = {}
        feature_dict = {}
        za_dict = {}
        for row in csv.reader(f):
            key = row[0]
            if key.startswith('#'):
                continue
            elif key.startswith(wavelength) and 'bg' in key:
                bg_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength) and 'za' in key:
                za_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength):
                feature_dict[key] = slice(float(row[1]), float(row[2]))
        if len(bg_dict) == 0 and len(feature_dict) == 0:
            raise ValueError(f'No bounds found for {wavelength} in {csv_path}')
    return feature_dict, bg_dict, za_dict


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))

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
    # sky_ds = sky_ds.drop_vars('tstamp')
    # sol_ds = sol_ds.drop_vars('tstamp')

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
        dask_gufunc_kwargs = {'allow_rechunk':True}
        

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


def apply_solar_subtraction(sol_ds: xr.Dataset, sky_ds: xr.Dataset):

    if isinstance(sol_ds, xr.Dataset):
        sol_ds = sol_ds['intensity']
    
    if isinstance(sky_ds, xr.Dataset):
        sky_ds = sky_ds['intensity']
 
    # if 'tstamp' in list(sol_ds.sizes.keys()):
    #     sol_ds['tstamp'] = sky_ds.tstamp
    # else:
    sol_ds = sol_ds.expand_dims(tstamp = sky_ds.tstamp.values)
    # sol_ds['tstamp'] = sky_ds.tstamp
    
    

    subtracted = xr.apply_ufunc(
        solar_subtraction_core,
        sol_ds,
        sky_ds,
        input_core_dims=[["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        on_missing_core_dim="copy",
        dask_gufunc_kwargs = {'allow_rechunk':True}
    )

    return subtracted