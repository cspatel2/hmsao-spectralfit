
# %%
import numpy as np
import matplotlib.pyplot as plt
from skmpython.GenericFit import GenericFitFunc, GenericFitManager
import xarray as xr
from glob import glob
import os
from typing import Callable, List, Iterable
import inspect
import uncertainties as un
import matplotlib
from sklearn.metrics import auc #area under curve

# %%
# polynomial


def background_fcn(x: float, x0: float, a0: float, a1: float, a2: float) -> float:
    return a0 + a1 * (x-x0) + a2*(x-x0)**2

# gaussian


def feature_fcn(x: float, c: float, a: float, w: float) -> float:
    return a*np.exp(-((x-c)/w)**2)


def spectral_fit(x: Iterable, y: Iterable, p0: List[float], plow: list[float] = None, phigh: List[float] = None, plot: bool = False, calc:bool = False):

    # initalize baseclass
    bfunc = GenericFitFunc(background_fcn=background_fcn,
                           num_background_params=4)
    bfunc.register_feature_fcn(fcn=feature_fcn, num_params=3)
    bfunc.finalize()

    # initialize manager
    gfit = GenericFitManager(
        x=x, y=y, p0=p0, baseclass=bfunc, plot=True, window_title='Test')
    print(gfit._baseclass)

    # fit
    if plow is None and phigh is None:
        popt, pcov = gfit.run(ioff=False, p0=p0)
    else:
        popt, pcov = gfit.run(ioff=False,p0=p0, bounds=(plow, phigh))
    print(popt)
    if plot:  # plot the final fit
        # gfit.plot()
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=300, tight_layout=True, squeeze=True)
        ax.plot(gfit._x, gfit._y, label = 'Data')
        ax.fill_between(gfit._x,gfit._y, alpha = 0.3)
        ax.plot(gfit._x, gfit.background(gfit._x, popt), label ='Fitted polynominal (bg)' )
        ax.fill_between(gfit._x, gfit.background(gfit._x, popt),alpha = 0.3 )
        ax.plot(gfit._x, gfit.feature(gfit._x, popt), label = 'Fitted Gaussian (signal)')
        ax.plot(gfit._x, gfit.full_field(gfit._x, popt), label = 'Gauss- poly fit')
        plt.axvline(x = popt[-3], label = f'Peak WL - {popt[-3]:0.2f} nm', color = 'violet')
        plt.legend(loc = 'best')
        ax.set_ylim(60,None)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Counts')
        ax.legend()
        plt.show()
        if not calc:
            return popt, pcov
        else:
            amp_v = popt[-2] #amplitude value
            amp_s = pcov[-2, -2] #amplitude std
            wid_v = popt[-1] #width value
            wid_s = pcov[-1, -1] #width std
            amp = un.ufloat(amp_v, amp_s)
            wid = un.ufloat(wid_v, wid_s)
            s_ct: un.UFloat = amp * wid * np.sqrt(np.pi) #area under curve of gaussian using fitted parameters
            fit_wl = un.ufloat(popt[-3],pcov[-3,-3]) #fitted peak wavelength
            # bg_out = quad(gfit.background, wlaxis[0], wlaxis[-1], args=(popt))#area under the background (polynomial) curve
            # bg_out = gfit.background(gfit._x,popt)
            bg_out = gfit.background(gfit._x, popt).mean()
            # bg_out = auc(gfit._x, gfit.background(gfit._x,popt))
            res = gfit._y - gfit._fit_func(gfit._x,*popt) #residual - measure of how good the fit is
            s = [r*r for r in res] # squared of residual

            res_auc = auc(gfit._x,s) # area under curve of residual
            res_out = np.sqrt(res_auc) # sqrt of AUC of residual

            ss = np.sum(s) # sum sqaured of residual
            RSS_out = np.sqrt(ss) # root squared sum of residual
            return s_ct, fit_wl, bg_out, RSS_out


# %%


def main(wl: str, ds: xr.Dataset, tidx: int):

    # select useable data
    wl = int(wl)/10
    dwl = 1  # nm
    wlslice = slice(wl-dwl, wl+dwl)
    za = 0
    dza = 20  # deg
    zaslice = slice(za - dza, za+dza)
    nds = ds.isel(tstamp=tidx).sel(wavelength=wlslice).sel(za=zaslice)

    # bin along za
    za_binsize = 4  # bins
    nds = nds.coarsen(za=za_binsize, boundary='pad').sum()

    # pick a single row
    zaidx = int(len(nds.za.values)/2) - 1  # center index closest to za = 0 deg
    lineds = nds.isel(za=zaidx)

    # drop all the zeros
    lineds = lineds.where(lineds > 50, drop=True)

    # fit
    x = lineds.wavelength.values
    y = lineds.intensity.values
    #p = [x0,a0,a1,a3,c,a,w]
    p0 = [np.mean(x), np.mean(y), (y[-1] - y[0]) / (x[-1] - x[0]), 0,
          wl, np.max(y) - np.min(y), .05]
    p_low = [0, 0, -np.inf, -np.inf, wl - 0.1, 0, 0]
    p_high = [np.inf, np.inf, np.inf, np.inf, wl+0.1, np.inf, dwl]
    feature, cwl, bg, res = spectral_fit(x=x, y=y, p0=p0, plow=p_low, phigh=p_high, plot=True, calc = True)
    return feature, cwl, bg, res



# %%
if __name__ == '__main__':
    wl = '6300'
    fnames = glob(f'test_data/*{wl}*.nc')
    ds = xr.open_dataset(fnames[0])
    feature, cwl, bg, rss= main(wl=wl, ds=ds, tidx=10)
    print(f'line strength: {feature}')
    print(f'Background strength: {bg}')
    print(f'Fitted Central Wl: {cwl}')
    print(f'RSS Residual: {rss}')



# %%
