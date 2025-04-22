#%%
from __future__ import annotations
from glob import glob
import os
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
from misdesigner import *
import xarray as xr
from PIL import Image
from typing import Dict, Iterable, List, SupportsFloat as Numeric
from astropy.io import fits as fits
import matplotlib as mpl
from skimage import transform
from tqdm import tqdm
usetex = False
from datetime import datetime
import pytz

if not usetex:
    # computer modern math text
    mpl.rcParams.update({'mathtext.fontset': 'cm'})
mpl.rc('font', **{'family': 'serif',
       'serif': ['Times' if usetex else 'Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=usetex)

# %%

def main(modelpath:str, fitsfname: str, wl: str):
    model = MisInstrumentModel.load(modelpath)
    predictor = MisCurveRemover(model)

    
    def get_tstamp_from_hdu(hdu) -> Numeric:
        if 'TIMESTAMP_S' in hdu.header:
            time_s = hdu.header['TIMESTAMP_S']
            if 'TIMESTAMP_NS' in hdu.header:
                time_ns = hdu.header['TIMESTAMP_NS']
            else:
                time_ns = 0
            time = time_s + 1e-9*time_ns
            return time
        else:
            raise ValueError('Invalid file')

    def get_exposure_from_hdu(hdu) -> Numeric:
        if 'EXPOSURE_S' in hdu.header:
            exp_s = hdu.header['EXPOSURE_S']
            if 'EXPOSURE_NS' in hdu.header:
                exp_ns = hdu.header['EXPOSURE_NS']
            else:
                exp_ns = 0
            exp = exp_s + 1e-9*exp_ns
            return exp
        else:
            raise ValueError('Invalid File')

    def zenith_angle(gamma_mm: Numeric | Iterable[Numeric], f1: Numeric = 30, f2: Numeric = 30, D: Numeric = 24, yoffset: Numeric = 12.7) -> Numeric:
        """Calculates the zenith angle in degrees from the gamma(mm) in slit coordinates.

        Args:
            gamma_mm (Numeric | Iterable[Numeric]): gamma (mm) in slit (instrument coordinate system) coordinates.
            f1 (Numeric, optional): focal length (mm) of the 1st lens in the telecentric foreoptic. Defaults to 30 mm.
            f2 (Numeric, optional): focal length (mm) of the 2nd lens in the telecentric foreoptic. Defaults to 30 mm.
            D (Numeric, optional): Distance (mm) between the two lens. Defaults to 24 mm.
            yoffset (Numeric, optional): the distance between the optic axis of the telescope to the x-axis of the instrument coordinate system. Defaults to 12.7 mm.

        Returns:
            Numeric: the zenith angle in degrees.
                    Note: result is non linear b/c of arctan()

        """
        if isinstance(gamma_mm, (int, float)):
            return [zenith_angle(x) for x in gamma_mm]
        if np.min(gamma_mm) < 0:
            sign = -1
        else:
            sign = 1
        num = -(gamma_mm-(sign*yoffset))*(f1+f2-D)
        den = f1*f2
        return np.rad2deg(np.arctan(num/den))


    def convert_gamma_to_zenithangle(ds: xr.Dataset, plot: bool = False, returnboth: bool = False):
        """converts gamma(mm) in slit coordinate to zenith angle (degrees) in a straightened dataset.

        Args:
            ds (xr.Dataset): straightened dataset.
            plot (bool, optional): if True, left plot is raw zenith angle and right plot is linearized zenith angle. Defaults to False.
            returnboth (bool, optional): if True, returns both datasets i.e. with raw (non linear) zenith angle and second with linear zenith angles. If false, only returns dataset with linear zenith angles. Defaults to False.

        Returns:
            _type_: dataset with gamma(mm) replaced with zenith angle (deg)
                    Note: calculated zenith angles are non-linear b/c of arctann(). This is corrected using ndimage.transform.warp() to a linearized zenith angles.
        """
        # initilize the new dataset with linear za
        nds = ds.copy()

        # gamma -> zenith angle
        angles = zenith_angle(ds.gamma.values)

        # coordinate map in the input image
        mxi, myi = np.meshgrid(ds.wavelength.values, angles)
        imin, imax = np.nanmin(myi), np.nanmax(myi)
        myi -= imin  # shift to 0
        myi /= (imax - imin)  # normalize to 1
        myi *= (len(angles))  # adjust

        # coordinate map in the output image
        if np.nanmin(angles) < 0:
            sign = 1
        else:
            sign = -1
        linangles = np.linspace(np.min(angles), np.max(angles), len(
            angles), endpoint=True)[::sign]  # array of linear zenith angles
        mxo, myo = np.meshgrid(ds.wavelength.values, linangles)
        omin, omax = np.nanmin(mxo), np.nanmax(mxo)
        mxo -= omin  # shift to 0
        mxo /= (omax - omin)  # normalize to 1
        mxo *= (len(ds.wavelength.values))  # adjust

        # inverse map
        imap = np.zeros((2, *(ds.shape)), dtype=float)
        imap[0, :, :] = myi  # input image map
        imap[1, :, :] = mxo  # output image map

        # nonlinear za -> linear za
        timg = transform.warp(ds.values, imap, order=1, cval=np.nan)

        # replace gamma to raw za values
        ds['gamma'] = angles
        ds['gamma'] = ds['gamma'].assign_attrs(
            {'unit': 'deg', 'long_name': 'Zenith Angle'})
        ds = ds.rename({'gamma': 'za'})
        # replace gamma to linear za values
        nds.values = timg
        nds['gamma'] = linangles
        nds['gamma'] = nds['gamma'].assign_attrs(
            {'unit': 'deg', 'long_name': 'Zenith Angle'})
        nds = nds.rename({'gamma': 'za'})
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
            fig.tight_layout()

            vmin = np.nanpercentile(ds.values, 1)
            vmax = np.nanpercentile(ds.values, 99)
            ds.plot(ax=ax1, vmin=vmin, vmax=vmax)
            ax1.set_title('Zenith Angle (NL)')

            vmin = np.nanpercentile(timg, 1)
            vmax = np.nanpercentile(timg, 99)
            nds.plot(ax=ax2, vmin=vmin, vmax=vmax)
            ax2.set_title('Zenith Angle (Warped Linear)')

        if returnboth:
            return nds, ds
        else:
            return nds



    with fits.open(fitsfname) as hdul:
        hdu = hdul['IMAGE']
        header = hdu.header
        tstamp = get_tstamp_from_hdu(hdu)  # s
        ststamp = datetime.fromtimestamp(tstamp, tz=pytz.utc)
        exposure = get_exposure_from_hdu(hdu)  # s
        temp = header['CCD-TEMP']  # C
        imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
        # 1. get img
        data = np.asarray(hdu.data, dtype=float)  # counts
        # # 2. dark/bias correction
        # if darkds is not None:
        #     dark = np.asarray(
        #         darkds['darkrate'].values, dtype=float)
        #     bias = np.asarray(darkds['bias'].values, dtype=float)
        #     data -= bias + dark * exposure  # counts
        # 3. total counts -> counts.sec
        data = data/exposure  # counts/sec
        # 4. Crop and resize image
        data = Image.fromarray(data)
        data = data.rotate(-.311,
                            resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
        data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = Image.new('F', imgsize, color=np.nan)
        image.paste(data, (110, 410))
        data = np.asarray(image).copy()
        del image
        # array -> DataArray
        data_ = xr.DataArray(
            data,
            dims=['gamma', 'beta'],
            coords={
                'gamma': predictor.gamma_grid,
                'beta': predictor.beta_grid
            },
            attrs={'unit': 'ADU/s'}
        )
        # 5. straighten img
        imaps = predictor._imaps
    
        for _, window, xform, coords, res in imaps:
            if window.name != wl:
                continue
            xran = (coords['beta'].max(), coords['beta'].min())
            yran = (coords['gamma'].min(), coords['gamma'].max())
            rawdata = data_.sel(gamma=slice(*yran), beta=slice(*xran))
            # rawdata /= res
            gamma = ('gamma', model._gamma_to_slit(model._gamma_from_image(model._gamma_from_mosaic(coords['gamma']))),
                            {
                    'coodinate': 'Slit',
                    'unit': 'mm',
                    'description': 'Height in the instrument coordinate.'
                })


        data = predictor.straighten_image(
            data_, wl, coord='Slit')
        data = convert_gamma_to_zenithangle(data)
        # 6. Save
        data = data.expand_dims(
            dim={'tstamp': (tstamp,)}).to_dataset(name='intensity', promote_attrs=True)
        data['exposure'] = xr.Variable(
            dims='tstamp', data=[exposure], attrs={'unit': 's'}
        )
        data['ccdtemp'] = xr.Variable(
            dims='tstamp', data=[temp], attrs={'unit': 'C'}
        )

    fig, [ax1,ax2] = plt.subplots(1,2,figsize = (10,6) )
    rawdata.isel(beta = slice(None,None,1)).plot(ax = ax1, vmax = 20)
    ax1.set_title('Raw')
    data.intensity.isel(tstamp = 0).plot(ax = ax2, vmax = 200)
    ax2.set_title('Straightened')
    w = int(wl)/10
    fig.suptitle(f'Window {w:0.1f} at {datetime.fromtimestamp(tstamp)}')
    plt.show()
# %%
if __name__ == '__main__':
    fn = 'test_data/20250118141143.641.fits'
    w = '6300' 
    modelpath = '../l1a_converter/hmsa_origin_ship.json'
    main(modelpath=modelpath, fitsfname=fn, wl=w)


# %%
