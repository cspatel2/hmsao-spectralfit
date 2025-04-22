# %%
# %%
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

if not usetex:
    # computer modern math text
    mpl.rcParams.update({'mathtext.fontset': 'cm'})
mpl.rc('font', **{'family': 'serif',
       'serif': ['Times' if usetex else 'Times New Roman']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=usetex)
# %%


def get_exposure_from_fn(fn: str) -> float:
    return float(fn.strip('.png').split('_')[-1])

# the following functions are copied from hmsao_l1a_converter.py. Make sure all the files here undergo the same processing as the l1a converter


def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # because we ignored the first row and first column
    hot_pixels = np.array(hot_pixels) + 1

    # This is the image with the hot pixels removed
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges == True:
        height, width = np.shape(data)

        ### Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height-1):
            # left side:
            med = np.median(data[index-1:index+2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index-1:index+2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width-1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width-1):
            # bottom:
            med = np.median(data[0:2, index-1:index+2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index-1:index+2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height-1], [index]]))
                fixed_image[-1, index] = med
        ### Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width-1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [width-1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image


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


# %%
def main(fnames: List[str], modelpath: str, prefix: str, darkds: str = None,READNOISE = None, plot: bool = False):
    model = MisInstrumentModel.load(modelpath)
    predictor = MisCurveRemover(model)
    imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
    
    # readnoise option
    readnoise_ = None
    rn_computed = {}

    if darkds is not None:
        darkds = xr.open_dataset(darkds)
        is_dark_subtracted = 'is'
    else:
        is_dark_subtracted = 'is not'

    file = fnames[0]
    output = {k: [] for k in predictor.windows}
    for fidx, file in enumerate(tqdm(fnames)):
        # get image and exposure
        data = Image.open(file)
        exp = get_exposure_from_fn(file)
        data = np.array(data)
        # hot pixel correction
        _, data = find_outlier_pixels(data)
        # dark subtraction

        if darkds is not None:
            dark = np.asarray(darkds['darkrate'].values, dtype=float)
            bias = np.asarray(darkds['bias'].values, dtype=float)
            data -= bias + dark * exp
        elif readnoise_ is None and READNOISE is not None:
            readnoise = np.full(
                (imgsize[-1],imgsize[0]), READNOISE, dtype=float)
            readnoise = Image.fromarray(readnoise)
            readnoise = readnoise.rotate(-.311,
                                            resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
            readnoise = readnoise.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT)
            image = Image.new('F', imgsize, color=np.nan)
            image.paste(readnoise, (110, 410))
            readnoise = np.asarray(image).copy()
            del image
            readnoise = xr.DataArray(
                readnoise,
                dims=['gamma', 'beta'],
                coords={
                    'gamma': predictor.gamma_grid,
                    'beta': predictor.beta_grid
                },
                attrs={'unit': 'ADU'}
            )
            for window in predictor.windows:
                rn = predictor.straighten_image(
                    readnoise, window, coord='Slit')
                rn_computed[window] = convert_gamma_to_zenithangle(rn)
        # Counts -> counts/sec
        data = data/exp
        # rotate and flip
        data = Image.fromarray(data)
        data = data.convert('F')
        data = data.rotate(-.311, resample=Image.Resampling.BILINEAR,
                           fillcolor=np.nan)
        data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = Image.new('F', imgsize, color=np.nan)
        image.paste(data, (110, 410))
        data = np.asarray(image).copy()
        del image
        # image to dataset to straighten
        data_ = xr.DataArray(
            data,
            dims=['gamma', 'beta'],
            coords={
                'gamma': predictor.gamma_grid,
                'beta': predictor.beta_grid
            },
            attrs={'unit': 'ADU/s'}
        )
        # straighten
        for window in predictor.windows:
            data = predictor.straighten_image(data_, window, coord='Slit')
            data = convert_gamma_to_zenithangle(data)
            data = data.expand_dims(
                dim={'idx': (fidx,)}).to_dataset(name='intensity', promote_attrs=True)
            data['exposure'] = xr.Variable(
                dims='idx', data=[exp], attrs={'unit': 's'})
            if plot:
                fig, ax = plt.subplots()
                data.plot(ax=ax)
                wl = int(window)/10
                ax.axvline(wl, color='red', lw=0.2, ls='--')
                plt.show()
                plt.close(fig)
            output[window].append(data)

    # save to netcdf
    for window in predictor.windows:
        sub_outfpath = f'{prefix}_l1a_{window}.nc'
        ds: xr.Dataset = xr.concat(output[window], dim='idx')
        ds['intensity'].attrs['unit'] = 'ADU/s'
        ds.attrs.update(
            dict(Description=" HMSA-O Straighted Image of Calibration Lamp",
                 ROI=f'{str(window)} nm',
                 DataProcessingLevel='1A',
                 # FileCreationDate=tnow,
                 ObservationLocation='LoCSST | Lowell, MA',
                 Note=f'data {is_dark_subtracted} dark corrected. \n Lamp calibration curve can be found in lightbox_calib_curve.nc',
                 ))
        if readnoise is not None:
            ds['noise'] = rn_computed[window] # readnoise]
        ds['intensity'].attrs['unit'] = 'ADU/s'
        encoding = {var: {'zlib': True}
                    for var in (*ds.data_vars.keys(), *ds.coords.keys())}
        ds.to_netcdf(sub_outfpath, encoding=encoding)

    del output


# %%
# %%
if __name__ == '__main__':
    darkds = None
    modelpath = '../../l1a_converter/hmsa_origin_ship.json'
    fnames = glob('calib-data/raw/*light*.png')
    prefix = 'caliblamp_'
    main(fnames=fnames, modelpath=modelpath, prefix=prefix,READNOISE=6, darkds=darkds)
# %%
# ncfiles = glob('caliblamp_*.nc')
# ds = xr.open_mfdataset(ncfiles[0])
# # %%
# ds.intensity.mean(dim = 'idx').plot(vmax = 20)
# # %%
# ds.intensity.isel(idx = 0).plot(vmax = 20)

# %%
# %%
