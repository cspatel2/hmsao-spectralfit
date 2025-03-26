#Photometric Calibration
Calculates conversion factor for photometric calibration of HMS-AO data

### 1. sav2nc.py
converts .sav file of calibration curve (Brightness vs Wavelength) to a usuable .nc file that will be used in the following steps.

### 2. straighten_png.py
Images of the LBS calibration lamp were taken using hmsao. Those images were saved as .png. __straighten_png.py__ performs L1a processing (consistant with https://github.com/cspatel2/hmsao-l1a-converter.git) on png images. Returns .nc files 

### 3. brightness_calib.py
creates photometric_calib.nc files for each wavelength. It contains a conversion factor for each pixel to convert data from instrument dependent units (Counts/s) to instrument independent units (Rayleighs).

