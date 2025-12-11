"""
Quick tool to check what's visible from Leiden at any given time.
This version uses lower resolution for faster computation.
"""

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from datetime import datetime

# Leiden Observatory coordinates
LEIDEN_LAT = 52.1676 * u.deg
LEIDEN_LON = 4.4576 * u.deg
LEIDEN_HEIGHT = 0 * u.m

leiden = EarthLocation(lat=LEIDEN_LAT, lon=LEIDEN_LON, height=LEIDEN_HEIGHT)


def check_visibility(obs_time_str, min_elevation=20):
    obs_time = Time(obs_time_str)

    print(f"Observable Sky from Leiden Observatory")
    print(f"Time: {obs_time.iso} UTC")
    print(f"Local Sidereal Time: {obs_time.sidereal_time('apparent', leiden)}")
    
    altaz_frame = AltAz(obstime=obs_time, location=leiden)
    zenith = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=altaz_frame)
    zenith_gal = zenith.galactic
    zenith_icrs = zenith.icrs
    
    print("ZENITH (straight overhead):")
    print(f"  Galactic: l={zenith_gal.l.deg:.2f}°, b={zenith_gal.b.deg:.2f}°")
    print(f"  RA/Dec:   α={zenith_icrs.ra.deg:.2f}°, δ={zenith_icrs.dec.deg:.2f}°")
    
    # Check cardinal directions at various elevations
    print("\nCARDINAL DIRECTIONS (at 45° elevation):")
    for direction, az in [("North", 0), ("East", 90), ("South", 180), ("West", 270)]:
        coord = SkyCoord(alt=45*u.deg, az=az*u.deg, frame=altaz_frame)
        gal = coord.galactic
        print(f"  {direction:6s}: Gal l={gal.l.deg:6.2f}°, b={gal.b.deg:6.2f}°")
    
    # Check interesting astronomical objects
    print("\nNOTABLE OBJECTS:")
    objects = {
        "Cygnus A (Cyg A)": SkyCoord(ra=299.8681*u.deg, dec=40.7339*u.deg, frame='icrs'),
        "Cas A": SkyCoord(ra=350.85*u.deg, dec=58.815*u.deg, frame='icrs'),
        "Virgo A (M87)": SkyCoord(ra=187.7059*u.deg, dec=12.3911*u.deg, frame='icrs'),
        "Orion Nebula": SkyCoord(ra=83.8221*u.deg, dec=-5.3911*u.deg, frame='icrs')
    }
    
    for name, coord in objects.items():
        altaz = coord.transform_to(altaz_frame)
        visible = "✓" if altaz.alt.deg > min_elevation else "✗"
        gal = coord.galactic
        print(f"  {visible} {name:20s}: Alt={altaz.alt.deg:6.2f}°, Az={altaz.az.deg:6.2f}°, "
              f"(l={gal.l.deg:6.2f}°, b={gal.b.deg:6.2f}°)")
    
    # Calculate observable fraction of sky
    # Sample the sky at regular intervals
    print("\nSKY COVERAGE:")
    alt_grid = np.linspace(min_elevation, 90, 20)
    az_grid = np.linspace(0, 360, 72)
    
    n_sampled = 0
    galactic_l_range = [360, 0]  # [min, max]
    galactic_b_range = [90, -90]  # [max, min]
    
    for alt in alt_grid:
        for az in az_grid:
            coord = SkyCoord(alt=alt*u.deg, az=az*u.deg, frame=altaz_frame)
            gal = coord.galactic
            n_sampled += 1
            
            galactic_l_range[0] = min(galactic_l_range[0], gal.l.deg)
            galactic_l_range[1] = max(galactic_l_range[1], gal.l.deg)
            galactic_b_range[0] = max(galactic_b_range[0], gal.b.deg)
            galactic_b_range[1] = min(galactic_b_range[1], gal.b.deg)
    
    # Approximate solid angle above horizon
    # For min_elevation > 0, solid angle = 2π(1 - sin(min_elevation))
    solid_angle_observable = 2 * np.pi * (1 - np.sin(np.radians(min_elevation)))
    solid_angle_total = 4 * np.pi
    percent_observable = 100 * solid_angle_observable / solid_angle_total
    
    print(f"  Minimum elevation: {min_elevation}°")
    print(f"  Observable sky: {percent_observable:.1f}% of full sphere")
    print(f"  Galactic longitude range: {galactic_l_range[0]:.1f}° to {galactic_l_range[1]:.1f}°")
    print(f"  Galactic latitude range: {galactic_b_range[1]:.1f}° to {galactic_b_range[0]:.1f}°")


if __name__ == "__main__":
    print("="*70)
    check_visibility("2025-11-21 16:00:00")
    print("="*70)
    