# %%
# LEVEL 1 B CONVERTER: Level 1B (L1A) data is line brightness (calucalated by sum) with photmetric calibration.
# %%
import sys
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterable
from typing import SupportsFloat as Numeric
import os
import argparse

from time import perf_counter_ns

import csv
import numpy as np
import xarray as xr
from sza import solar_zenith_angle
from itertools import chain
from datetime import datetime
from pytz import timezone, UTC

from util_functions import *

# %%
parser = argparse.ArgumentParser(
    description='Convert L1A data to L1b Data with photometric calibration.')

parser.add_argument(
    'rootdir',
    metavar='rootdir',
    type=str,
    help='Root directory of L1A data'
)

parser.add_argument(
    'destdir',
    metavar='destdir',
    # required = False,
    type=str,
    default=os.getcwd(),
    nargs='?',
    help='Root directory where L1 data will be stored.'
)

parser.add_argument(
    'dest_prefix',
    metavar='dest_prefix',
    # required = False,
    type=str,
    default=None,
    nargs='?',
    help='Prefix of the saved L1 data finename.'
)

parser.add_argument(
    '--overwrite',
    required=False,
    type=str2bool,
    default=False,
    nargs='?',
    help='If you want to rewrite an existing file, then True. Defaults to False.'
)


def list_of_strings(arg: str) -> List[str]:
    return arg.split(',')


parser.add_argument(
    '--windows',
    # metavar = 'NAME',
    # action='append',
    required=False,
    type=list_of_strings,
    default=None,
    nargs='?',
    help='Window(s) to process (list of str i.e. "1235", "3456").'
)

parser.add_argument(
    '--bounds',
    # metavar = 'NAME',
    required=True,
    type=str,
    # default = AS REQUIRED,
    nargs='?',
    help='bounds CSV file path.'
)

parser.add_argument(
    '--pcalibdir',
    metavar='photometric_calibration_dir',
    required=True,
    type=str,
    # default = AS REQUIRED,
    nargs='?',
    help='Photometric Calibration (Conversion Factor) .nc directory path.'
)

parser.add_argument(
    '--zabinsize',
    type=float,
    default=None,
    nargs='?',
    help='the binsize of Zenith Angle (za | y-axis) in deg. Defaults to 1.5 deg.'
)

parser.add_argument(
   '--dates',
    required= False,
    type=list_of_strings,
    default = None,
    nargs = '?',
    help = 'Dates to process in the format YYYYMMDD  (list seperated by commas).'
)

parser.add_argument(
    '--soldir',
    required=True,
    type=str,
    nargs='?',
    help='Directory path for Solar Spectra 1la file (.nc).'
)
# %%


def main():
    # Initialize constants and inputs
    args = parser.parse_args()
    SWEDEN = {'lon': 20.41, 'lat': 67.84, 'elev': 420}  # deg, deg, m

    # Check solspec .nc file
    if not os.path.exists(args.soldir):
        print('solar spectrum l1a (.nc) file does not exist.')

    # Check Directories
    # Destination dir
    if os.path.exists(args.destdir):
        if os.path.isfile(args.destdir):
            print('Destination path provided is a file. Directory path required.')
            sys.exit()
    else:
        os.makedirs(args.destdir, exist_ok=True)

    # Root Directory
    if not os.path.exists(args.rootdir):  # doesnt exist
        print('Root directory does not exist.')
        sys.exit()
    # exists but is a file
    elif not os.path.isdir(args.rootdir) and os.path.isfile(args.rootdir):
        print('Root directory path provided is a file. Directory path required.')
        sys.exit()
    elif len(glob(os.path.join(args.rootdir, '*.nc'))) == 0 and len(os.listdir(args.rootdir)) == 0:
        print('Root Directory is empty.')  # exists but is empty
        sys.exit()
    elif len(glob(os.path.join(args.rootdir, '*.nc'))) == 0 and len(os.listdir(args.rootdir)) != 0:
        subdirs: Iterable = [f for f in os.listdir(
            args.rootdir) if os.path.isdir(os.path.join(args.rootdir, f))]
    else:  # .nc files are in this dir
        subdirs = ['']

    valid_windows: Iterable = [w for w in args.windows for subdir in subdirs if len(
        glob(os.path.join(args.rootdir, subdir, f'*{w}*.nc'))) > 0]

    # Sol Directory - valid window if data in both soldir and rootdir
    valid_windows = [w for w in valid_windows if len(
        glob(os.path.join(args.soldir, f'*{w}*.nc'))) > 0]

    if len(valid_windows) == 0:
        sys.exit('No valid windows found in the root directory.')
    print(f'Valid windows to process: {valid_windows}')

    args.windows = valid_windows

    # Destination prefix
    if args.dest_prefix is None:
        print('Destination prefix is not provided. Using default: hmsao-l1b')
        args.dest_prefix = 'hmsao-l1b'
    elif not args.dest_prefix and 'l1b' not in args.dest_prefix.lower():
        args.dest_prefix = f'{args.dest_prefix}-l1b'

    # Zenith Angle bin size
    if args.zabinsize is None:
        args.zabinsize = 1.5
    else:
        args.zabinsize = float(args.zabinsize)

    for win in args.windows:
        # BOUNDS
        # get bounds for the window
        feature_bounds, bg_bounds, za_bounds = get_bounds_from_csv(
            args.bounds, win)
        ZASLICE = za_bounds[f'{win}_za']
        # assert that all bounds are the same size in nm
        boundsize = None
        for key, bound in chain(feature_bounds.items(), bg_bounds.items()):
            if boundsize is None:
                boundsize = np.ceil(bound.stop - bound.start)
            else:
                if boundsize != np.ceil(bound.stop - bound.start):
                    raise ValueError(
                        f'Bounds {key} = {bound} is not the same size as feature size {boundsize}')

        # PHOTOMETRIC CALIBRATION
        if not os.path.exists(args.pcalibdir):
            print('Photometric calibration directory does not exist.')
            sys.exit()
        elif not os.path.isdir(args.pcalibdir) and os.path.isfile(args.pcalibdir):
            print(
                'Photometic calibration directory path provided is a file. Directory path required.')
            sys.exit()
        else:
            calibfn = glob(os.path.join(args.pcalibdir, f'*calib*{win}*.nc'))
            if len(calibfn) == 0:
                print(f'No photometric calibration file found for {win}.')
                sys.exit()
            calibds = xr.open_dataset(calibfn[0])


        # DATES
        for subdir in subdirs:
            dates: Iterable = [os.path.basename(f).split('_')[1] for f in glob(
                os.path.join(args.rootdir, subdir, f'*{win}*.nc'))]
            dates = np.unique(dates)

            dates.sort()  # type: ignore
        
        if args.dates is not None:
            valid_dates = [d for d in args.dates if d in dates]
            dates = valid_dates

            # LEVEL 1B CONVERSION FOR EACH DATE
            for yymmdd in tqdm(dates, desc=f'{win}'):  # type: ignore
                # OUTFILE
                yymm: str = datetime.strptime(
                    yymmdd, '%Y%m%d').strftime('%Y%m')
                outfn: str = os.path.join(
                    args.destdir, yymm, f'{args.dest_prefix}_{yymmdd}_{win}.nc')
                os.makedirs(os.path.dirname(outfn), exist_ok=True)
                # check overwrite
                # file exists and not overwriting
                if os.path.exists(outfn) and not args.overwrite:
                    print(f'File {outfn} already exists. Skipping.')
                    continue
                # file exists and overwriting
                elif os.path.exists(outfn) and args.overwrite:
                    os.remove(outfn)
                    print(f'{outfn} removed, overwriting...')
                else:  # file does not exist so overwrite does not matter
                    pass

                # 0. Get data
                fnames: Iterable = glob(os.path.join(
                    args.rootdir, subdir, f'*{yymmdd}*{win}*.nc'))
                ds: xr.Dataset = xr.open_mfdataset(fnames)
                nds: xr.Dataset = ds.copy()
                wl: float = int(win)/10  # nm
                dwl: float = 0.6  # nm

                # SOLAR DATA
                solds: xr.Dataset = xr.open_dataset(
                    glob(os.path.join(args.soldir, f'*{win}*.nc'))[0])
                solds = solds.drop_vars(['ccdtemp', 'za'])
                # align the solar spectra to the sky spectra
                sol = solds.isel(tstamp=slice(0, 50))
                counts = sol.intensity.sum('tstamp', skipna=True)
                exp = sol.exposure.sum('tstamp')
                # TODO: add noise here
                sol['intensity'] = counts/exp
                del counts, exp

                sky = nds.isel(tstamp=slice(0, 50))
                counts = sky.intensity.sum('tstamp', skipna=True)
                exp = sky.exposure.sum('tstamp')
                sky['intensity'] = counts/exp
                del counts, exp

                # type: ignore , here solds is da(za,wavelength)
                a = apply_alignment(sol, sky,  -0.022)
                # a = apply_alignment(sol, sky)

                solds = a

                nds = nds.sel(wavelength=slice(wl-dwl, wl+dwl))
                solds = solds.sel(wavelength=slice(wl-dwl, wl+dwl))

                # 1. Calc Solar Zenith Angle (sza)
                nds['sza'] = ('tstamp', [solar_zenith_angle(t, lat=SWEDEN['lat'], lon=SWEDEN['lon'], elevation=SWEDEN['elev']) for t in ds.tstamp.values]
                              )
                nds.sza.attrs = {'units': 'deg',
                                 'long_name': 'Solar Zenith Angle'}

                # # 2. protometric calib, convert from photons/s -> Rayleighs
                # noise: xr.DataArray = nds.intensity * (np.sqrt((calibds.conversion_error/calibds.conversion_factor)** 2 + (nds.noise/nds.intensity)**2))
                # nds = nds.assign(intensity=ds.intensity * calibds.conversion_factor)
                # nds = nds.assign(noise=noise)

                # 3. remove unnecessary variables, add back to the final ds later
                sza: xr.DataArray = nds.sza
                exposure: xr.DataArray = nds.exposure
                ccdtemp: xr.DataArray = nds.ccdtemp
                nds = nds.drop_vars(['exposure', 'ccdtemp'])

                # 4. Bin Along ZA (y-axis)
                # select relevant za slice
                nds = nds.sel(za=ZASLICE)
                # binsize
                ZABINSIZE: int = int(
                    np.ceil(args.zabinsize/np.mean(np.diff(nds.za.values))))
                # bin sky data
                coarsen = nds.coarsen(za=ZABINSIZE, boundary='trim')
                nds = coarsen.sum()  # type: ignore  intensity is summed
                nds = nds.assign(noise=coarsen.reduce(
                    rms_func).noise)  # noise is rms(noise)

                # bin daytime data
                solds = solds.sel(za=ZASLICE)
                solds = solds.coarsen(za=ZABINSIZE, boundary='trim').sum()

                # TODO: 5. handle daytime/nighttime data processing
                sza_twilight: float = 90 + 6  # civil dawn is 18deg below horizon

                # Daytime
                dayds = nds.where(nds.sza < sza_twilight, drop=True)
                dayds = dayds.drop_vars('sza')
                nightds = nds.where(nds.sza >= sza_twilight, drop=True)
                nightds = nightds.drop_vars('sza')

                if len(dayds.tstamp.values) > 0:
                    # print(f'dayds = {list(dayds.keys())}')
                    b = apply_solar_subtraction(
                        solds,
                        dayds
                    )
                    dayds['intensity'] = b.clip(min=0)
                    # dayds['intensity'] = b


                    # reconstruct the whole dataset
                    assert (nightds.za == dayds.za).all()
                    assert (nightds.wavelength == dayds.wavelength).all()

                    nds = xr.concat([nightds, dayds], dim="tstamp")
                    nds = nds.sortby('tstamp')

                    print(f'nds = {list(nds.variables)}')

                    # nds = nds.drop_vars(['sza'])

                # nightime
                # nds = nds.where(nds.sza >= sza_astrodown, drop=True)

                # 6A. background intensities (by sum)
                bck_: Iterable[xr.Dataset] = []  # list of ds
                for key, bound in bg_bounds.items():
                    # sum all intensities in the wl slice for each za
                    bds = nds.sel(wavelength=bound).sum(dim='wavelength')
                    bds = bds.assign(noise=nds.sel(wavelength=bound).reduce(
                        # noise is rms(noise)
                        rms_func, dim='wavelength').noise)
                    bck_.append(bds)
                bckds: xr.Dataset = xr.concat(bck_, dim='idx')
                del bck_
                bckds = bckds.assign(noise=bckds['noise'].reduce(
                    rms_func, dim='idx')/bckds.idx.size)
                bckds = bckds.assign(
                    intensity=bckds['intensity'].mean(dim='idx'))

                # 6B. line intensities (by sum) and background subtraction
                line_: Iterable[xr.Dataset] = []
                for key, bound in feature_bounds.items():
                    # sum all intensities in the wl slice for each za
                    bds = nds.sel(wavelength=bound).sum(dim='wavelength')
                    bds = bds.assign(noise=nds.sel(wavelength=bound).reduce(
                        # noise is rms(noise)
                        rms_func, dim='wavelength').noise)
                    # error propagation for line intensity
                    noise = np.sqrt(bds.noise**2 + bckds.noise**2)
                    bds -= bckds.intensity  # subtract the average background
                    # add the error to the line intensity
                    bds = bds.assign(noise=noise)
                    bds = bds.rename({'intensity': key, 'noise': f'{key}_err'})
                    bds[key].attrs = {'units': 'Counts/s',
                                      'long_name': f'{key} Å  Line Brightness'}
                    bds[f'{key}_err'].attrs = {'units': 'Counts/s',
                                               'long_name': f'{key} Å Line Brightness Error'}
                    line_.append(bds)

                # 7. prepare final dataset
                # rename varirables
                bckds = bckds.rename({'intensity': 'bg', 'noise': 'bg_err'})
                bckds['bg'].attrs = {'units': 'Counts/s',
                                     'long_name': 'Mean Background Brightness'}
                bckds['bg_err'].attrs = {
                    'units': 'Counts/s', 'long_name': 'Mean Background Brightness Error'}
                # merge line and background datasets
                line_.append(bckds)
                saveds: xr.Dataset = xr.merge(line_)
                saveds = saveds.assign_coords(dict(
                    sza=('tstamp', sza.values),
                    ccdtemp=('tstamp', ccdtemp.values),
                    exposure=('tstamp', exposure.values),
                ))
                saveds.ccdtemp.attrs = ccdtemp.attrs
                saveds.exposure.attrs = exposure.attrs
                saveds.sza.attrs = sza.attrs

                saveds = saveds.assign_coords(dict(
                    daybool=xr.where(saveds.sza > sza_twilight, 1, 0))
                )
                saveds.daybool.attrs = dict(unit='Bool',
                                            description='True(1) if its daytime, False(0) if its nighttime. \n Astronomincal dawn is when the sun is 18 deg (sza = 90+18=108) below horizon.')

                saveds.attrs.update(
                    dict(Description=" HMSA-O line brightness",
                         ROI=f'{wl} nm',
                         DataProcessingLevel='1B',
                         FileCreationDate=datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT"),
                         ObservationLocation='Swedish Institute of Space Physics/IRF (Kiruna, Sweden)',
                         Note=f'data background subtracted.',
                         )
                )

                # 8. save dataset
                encoding = {var: {'zlib': True}
                            for var in (*saveds.data_vars.keys(), *saveds.coords.keys())}
                print('Saving %s...\t' % (os.path.basename(outfn)), end='')
                sys.stdout.flush()
                tstart = perf_counter_ns()
                saveds.to_netcdf(outfn, encoding=encoding)
                tend = perf_counter_ns()
                print(f'Done. [{(tend-tstart)*1e-9:.3f} s]')


if __name__ == '__main__':
    main()
