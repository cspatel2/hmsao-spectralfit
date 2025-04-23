# %%

from datetime import datetime, timezone, timedelta
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
import matplotlib.pyplot as plt
import numpy as np



# %%

def solar_zenith_angle(tstamp:float, lat:float, lon:float, elevation:float)->float:
    """ Calcuates solar zenith angle using location and time.

    Args:
        tstamp (float): Seconds since UNIX epoch 1970-01-01 00:00:00 UTC
        lat (float): Latitude of observer in degrees. -90°(S) <= lat <= +90°(N) 
        lon (float): Longitude of observer in degrees.  Longitudes are measured increasing to the east, so west longitudes are negative.
        elevation (float): height in meters above sea level. 

    Returns:
        float: SZA range is  0° (Local Zenith) <= sza <= 180°
    """    
    # Create an EarthLocation object
    location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=elevation*u.m)

    #date and time in UTC
    # date = datetime.datetime(2025, 1, 18, 12, 45, 0)
    date = datetime.fromtimestamp(tstamp,tz = timezone.utc)
    time = Time(date, scale='utc')

    # Create an AltAz frame
    altaz = AltAz(obstime=time, location=location)

    # Get the Sun's position in AltAz coordinates
    sun = get_sun(time)
    sun_altaz = sun.transform_to(altaz)

    # Extract the zenith angle (90 - altitude)
    # zenith_angle = 90 * u.deg - sun_altaz.alt
    zenith = np.abs(sun_altaz.alt.deg -90)


    # print(f"{time}")
    # lt = date.astimezone(tz = pytz.timezone('Europe/Stockholm'))
    # print(f'{lt} -- {zenith_angle}')
    return zenith


if __name__ == '__main__':
    # Kiruna, Sweden coordinates
    longitude = 20.41
    latitude = 67.84 
    elevation = 420 # Approximate elevation

    test_date = datetime(2025, 1, 27, tzinfo=timezone.utc) 
    res = [test_date]
    for i in range(24*4):
        t = res[-1] + timedelta(minutes = 30)
        res.append(t)
    tstamps = [datetime.timestamp(r) for r in res]
    sza = [solar_zenith_angle(t, latitude,longitude,elevation) for t in tstamps]


    plt.plot(res,sza)
    plt.axhline(90, ls='-.')
    plt.gcf().autofmt_xdate()
    plt.xlabel('time (UTC)')
    plt.ylabel('Solar Zenith Angle (deg)')
    plt.title('Location: Kiruna, Sweden')
    plt.ylim(np.max(sza)+5, np.min(sza)-5)
    




# %%
