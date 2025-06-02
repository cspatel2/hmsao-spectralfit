# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import readsav
import xarray as xr
# %%
df = readsav('lightbox_calib_curve.sav')

# %%
keys = list(df.keys())
# %%
wavelength = np.asarray(df[keys[0]], dtype = float)
brightness = np.asarray(df[keys[1]], dtype = float)
# %%
plt.plot(wavelength, brightness, 'o-')
# %%
ds = xr.Dataset(
    data_vars={
        'brightness': (['wavelength'], brightness)
    },
    coords={
        'wavelength': (['wavelength'], wavelength)
    }
)

ds['wavelength'].attrs.update({'units': '\u212B', 'full_units': 'Angstrom'})
ds['brightness'].attrs.update({'units': 'R/\u212B', 'full_units':'Rayleighs/Angstrom','description': 'brightness of lightbox'})

ds.attrs.update({'description': 'LBS calibration curve',
                 'calibration_location': 'Boston University',
                 'calibration_time':' April 2017',
                'source_file': 'lightbox_calib_curve.sav',
                'history': 'created by sav2nc.py',
               'file_creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S EDT')})

ds.to_netcdf('lightbox_calib_curve.nc')
# %%
ds = xr.open_dataset('lightbox_calib_curve.nc')
