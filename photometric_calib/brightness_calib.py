#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from astropy.io import fits
from glob import glob
from datetime import datetime
import os
# %%

#%%
def main(wl:str, calib_curve_fname: str, lamp_hmsimg_fname:str):
    '''
    returns a calibration factor dataset that converts from instrument dependent units (counts) to instrument independent units (Rayleighs). It uses the LBS lightbox calibration curve and caliblab_l1a data. '''

    if wl not in ['6563','4861','5577','6300','7774','4278']:
        raise ValueError('Wavelength not supported')


    #get dataset that has l1a images of lamp 
    ds = xr.open_mfdataset(lamp_hmsimg_fname)
    countsds = ds.intensity.mean(dim='idx') #countrate (counts/s)
    noise = ds.noise.mean(dim = 'idx') #countrate (counts/s)
    wlarray = countsds.wavelength.values #nm

    #interp brightness for wl array using calib curve
    calibds = xr.open_dataset(calib_curve_fname)
    x = wlarray * 10 #nm -> A
    xp = calibds['wavelength'].values #A
    yp = calibds['brightness'].values #Rayleigh/A
    brightness = np.interp(x, xp, yp) #Rayleigh/A
    # plt.plot(xp,yp)
    # plt.plot(x,brightness)
    # plt.show()
    brightness *= np.mean(np.diff(x)) #Rayleigh/A -> Rayleigh

    #image is straightened, so each row is same wavelength
    height,_ = countsds.shape
    brightness_grid = np.tile(brightness, (height,1)) #Rayleigh 

    

    conversion_factor = brightness_grid/countsds.values #Rayleigh/countrate
    conversion_error = conversion_factor * (noise.values/countsds.values) #rayleigh/countrate
    rateds = xr.Dataset(
        data_vars=dict(
            conversion_factor=(['za','wavelength'], conversion_factor),
            conversion_error=(['za','wavelength'], conversion_error),
                       ),
        coords=dict(za=('za',countsds.za.values),
                     wavelength=('wavelength', countsds.wavelength.values))
    )
    rateds.conversion_factor.attrs['units'] = 'Rayleigh/ CountRate'
    rateds.conversion_factor.attrs['long_name'] = 'Conversion Factor'
    rateds.conversion_error.attrs['units'] = 'Rayleigh/ CountRate'
    rateds.conversion_error.attrs['long_name'] = 'Error in Conversion Factor'
    rateds.wavelength.attrs['units'] = 'nm'
    rateds.za.attrs['units'] = 'deg'
    rateds.za.attrs['long_name'] = 'Zenith Angle'

    rateds = rateds.assign_attrs({
        'description': 'Conversion factor of instrument dependent units (counts) to instrument independent units (Rayleighs) using LBS lightbox from Boston Univeristy.',
        'calib_curve_fname': calib_curve_fname,
        'date_created': str(datetime.now()),
        'Note': 'OG ' + ds.attrs['Note'].split('\n')[0].strip(' ')
    })
    outfname = f'hmsao_photometric_calib_{wl}.nc'
    print(f'saving {outfname}...')
    rateds.to_netcdf(outfname)
    print('Done.')
    del rateds
    # return rateds

# %%
if __name__ == '__main__':
    curvefn = 'lightbox_calib_curve.nc'
    datadir = 'calib-data/l1a'
    for wl in ['6563','4861','5577','6300','7774','4278']:
        fnames = glob(os.path.join(datadir,f'*{wl}.nc'))
        main(wl = wl, calib_curve_fname=curvefn, lamp_hmsimg_fname= fnames)

# %%
