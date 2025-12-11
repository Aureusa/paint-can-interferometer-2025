"""
Calculate which region of the HI4PI all-sky map is observable from Leiden
at a specific time, and overlay it on the map.
"""

from astropy.io import fits
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, Galactic
from astropy.time import Time
import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
from datetime import datetime, timedelta

# Leiden Observatory coordinates
# You can adjust these for your specific location
LEIDEN_LAT = 52.1676 * u.deg
LEIDEN_LON = 4.4576 * u.deg
LEIDEN_HEIGHT = 0 * u.m  # Approximately sea level

leiden = EarthLocation(lat=LEIDEN_LAT, lon=LEIDEN_LON, height=LEIDEN_HEIGHT)

def get_observable_region(observer_location, obs_time, min_elevation=20*u.deg, nside=1024):
    # Generate all HEALPix pixel centers in galactic coordinates
    npix = hp.nside2npix(nside)
    
    # Get galactic coordinates for all pixels
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    
    # Convert to galactic longitude and latitude
    # HEALPix theta is colatitude (0 at north pole), phi is longitude
    gal_l = np.degrees(phi) * u.deg
    gal_b = (90 - np.degrees(theta)) * u.deg
    
    # Create SkyCoord in Galactic frame
    sky_coords = SkyCoord(l=gal_l, b=gal_b, frame='galactic')
    
    # Transform to AltAz frame for the observer
    altaz_frame = AltAz(obstime=obs_time, location=observer_location)
    altaz_coords = sky_coords.transform_to(altaz_frame)
    
    # Determine which pixels are above the minimum elevation
    observable_mask = altaz_coords.alt > min_elevation
    
    return observable_mask, altaz_coords


def export_observable_to_fits(healpix_map, observable_mask, obs_time, observer_location,
                               output_prefix="observable_sky", 
                               pixel_size_arcmin=1.0, 
                               image_size_deg=None,
                               center_coord=None):
    """
    Export the observable HI4PI region to FITS and PNG files with square projection.
    
    Parameters:
    -----------
    healpix_map : array
        HEALPix map data
    observable_mask : array
        Boolean mask of observable pixels
    obs_time : Time
        Observation time
    observer_location : EarthLocation
        Observer's location
    output_prefix : str
        Prefix for output filenames
    pixel_size_arcmin : float
        Pixel size in arcminutes (default 1.0)
    image_size_deg : float
        Size of the square image in degrees (default: auto-calculated from observable region)
    center_coord : SkyCoord
        Center coordinate for the image (default: zenith)
    
    Returns:
    --------
    output_fits : str
        Path to output FITS file
    output_png : str
        Path to output PNG file
    """
    from astropy.wcs import WCS
    from astropy.io import fits as astropy_fits
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    print("\n" + "="*70)
    print("EXPORTING OBSERVABLE REGION TO FITS AND PNG")
    print("="*70)
    
    # Determine center coordinate (default to zenith)
    if center_coord is None:
        zenith_altaz = SkyCoord(alt=90*u.deg, az=0*u.deg, 
                               frame=AltAz(obstime=obs_time, location=observer_location))
        center_coord = zenith_altaz.galactic
        print(f"Center: Zenith (Galactic l={center_coord.l.deg:.2f}°, b={center_coord.b.deg:.2f}°)")
    else:
        print(f"Center: Custom (Galactic l={center_coord.l.deg:.2f}°, b={center_coord.b.deg:.2f}°)")
    
    # Determine image size (default to cover observable region with some margin)
    if image_size_deg is None:
        # For Leiden at ~52° latitude, observable hemisphere extends roughly 90° from zenith
        # Use a conservative 120° to capture most observable region
        image_size_deg = 70.0
        print(f"Image size: {image_size_deg}° × {image_size_deg}° (auto-sized)")
    else:
        print(f"Image size: {image_size_deg}° × {image_size_deg}° (user-specified)")
    
    # Calculate number of pixels
    pixel_size_deg = pixel_size_arcmin / 60.0
    n_pixels = int(np.ceil(image_size_deg / pixel_size_deg))
    # Ensure even number for symmetry
    if n_pixels % 2 != 0:
        n_pixels += 1
    
    actual_size_deg = n_pixels * pixel_size_deg
    print(f"Pixel size: {pixel_size_arcmin:.2f} arcmin = {pixel_size_deg:.4f}°")
    print(f"Image dimensions: {n_pixels} × {n_pixels} pixels")
    print(f"Actual image size: {actual_size_deg:.2f}° × {actual_size_deg:.2f}°")
    
    # Create WCS (World Coordinate System) for the image
    # Using Galactic coordinates with tangent plane projection (TAN)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [n_pixels/2.0 + 0.5, n_pixels/2.0 + 0.5]  # Reference pixel (center)
    wcs.wcs.crval = [center_coord.l.deg, center_coord.b.deg]  # Reference coordinate
    wcs.wcs.cdelt = [-pixel_size_deg, pixel_size_deg]  # Pixel scale (negative for RA-like axis)
    wcs.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]  # Galactic coordinates, tangent projection
    wcs.wcs.cunit = ["deg", "deg"]
    
    print("\nProjecting HEALPix data to square grid...")
    
    # Create coordinate grid for the output image
    y_indices, x_indices = np.mgrid[0:n_pixels, 0:n_pixels]
    
    # Convert pixel coordinates to world coordinates
    world_coords = wcs.pixel_to_world(x_indices, y_indices)
    glon = world_coords.l.deg
    glat = world_coords.b.deg
    
    # Convert to HEALPix pixel indices
    nside = hp.npix2nside(len(healpix_map))
    theta = np.radians(90.0 - glat)  # Colatitude
    phi = np.radians(glon)            # Longitude
    healpix_indices = hp.ang2pix(nside, theta, phi)
    
    # Create the output image by sampling the HEALPix map
    output_image = healpix_map[healpix_indices]
    
    # Mask pixels that are not observable (set to NaN)
    is_observable = observable_mask[healpix_indices]
    output_image_masked = np.where(is_observable, output_image, np.nan)
    
    # Calculate statistics
    valid_pixels = ~np.isnan(output_image_masked)
    n_valid = np.sum(valid_pixels)
    percent_valid = 100 * n_valid / output_image_masked.size
    
    print(f"\nImage statistics:")
    print(f"  Valid (observable) pixels: {n_valid} ({percent_valid:.1f}%)")
    if n_valid > 0:
        valid_data = output_image_masked[valid_pixels]
        print(f"  N_H range: {np.nanmin(valid_data):.2e} to {np.nanmax(valid_data):.2e} cm^-2")
        print(f"  N_H median: {np.nanmedian(valid_data):.2e} cm^-2")
    
    # Create FITS file with proper header
    output_fits = f"{output_prefix}_{obs_time.isot.replace(':', '-')}.fits"
    
    # For simulation, use linear scale (not log)
    # Non-observable pixels set to 0.0 instead of NaN for CASA compatibility
    output_image_for_fits = np.where(is_observable, output_image, 0.0)
    
    header = wcs.to_header()
    header['OBJECT'] = 'HI4PI Observable Region'
    header['TELESCOP'] = 'Leiden Observatory (simulated)'
    header['OBSERVER'] = 'Observation Planning'
    header['DATE-OBS'] = obs_time.isot
    header['OBSTIME'] = obs_time.isot
    header['MJD-OBS'] = obs_time.mjd
    header['BUNIT'] = 'cm-2'  # Column density units
    header['BTYPE'] = 'Intensity'
    header['COMMENT'] = 'HI column density from HI4PI survey'
    header['COMMENT'] = f'Observable from lat={observer_location.lat.deg:.4f} lon={observer_location.lon.deg:.4f}'
    header['COMMENT'] = f'Minimum elevation: 20 degrees'
    header['COMMENT'] = f'Pixel size: {pixel_size_arcmin:.2f} arcmin'
    header['COMMENT'] = 'Non-observable pixels set to 0.0'
    header['COMMENT'] = 'Data in LINEAR scale (not log)'
    
    # Create HDU and write to file
    hdu = astropy_fits.PrimaryHDU(data=output_image_for_fits, header=header)
    hdu.writeto(output_fits, overwrite=True)
    print(f"\n✓ FITS file saved: {output_fits}")
    
    # Create PNG model image - clean image with just the data, no axes/labels
    output_png = f"{output_prefix}_{obs_time.isot.replace(':', '-')}.png"
    
    # Calculate vmax for linear scale
    vmax_linear = np.nanpercentile(output_image_masked[valid_pixels], 99)
    
    # Create figure with exact pixel dimensions (no margins, axes, or labels)
    dpi = 100  # DPI for output
    fig = plt.figure(figsize=(n_pixels/dpi, n_pixels/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    ax.axis('off')  # No axes
    
    # Display only the linear data (as in FITS file)
    ax.imshow(output_image_for_fits, origin='lower', cmap='viridis',
              vmin=0, vmax=vmax_linear, interpolation='nearest')
    
    # Save with no padding or extra whitespace
    plt.savefig(output_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✓ PNG model image saved: {output_png}")
    print(f"  (Clean image: {n_pixels}×{n_pixels} pixels, linear scale, no axes)")
    
    print("="*70 + "\n")
    
    return output_fits, output_png


def plot_observable_region(healpix_map, observable_mask, obs_time, observer_name="Leiden"):
    """
    Plot the HI4PI map with the observable region highlighted.
    """
    # Create masked version of the data (set non-observable to UNSEEN)
    masked_data = healpix_map.copy()
    masked_data[~observable_mask] = hp.UNSEEN
    
    # Define Cas A coordinates (RA, Dec) and convert to Galactic
    cas_a = SkyCoord(ra=350.85*u.deg, dec=58.815*u.deg, frame='icrs')
    cas_a_gal = cas_a.galactic
    
    fig = plt.figure(figsize=(18, 12))
    
    # Full sky map
    hp.mollview(healpix_map, title=f'HI4PI All-Sky Map (Full Sky)', 
                unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
                coord='G', fig=fig.number, sub=(2, 2, 1))
    hp.graticule()
    # Mark Cas A
    hp.projplot(cas_a_gal.l.deg, cas_a_gal.b.deg, 'r*', markersize=15, 
                coord='G', lonlat=True)
    hp.projtext(cas_a_gal.l.deg, cas_a_gal.b.deg + 5, 'Cas A', 
                coord='G', lonlat=True, color='red', fontsize=10, 
                ha='center', weight='bold')
    
    # Show only the observable region
    hp.mollview(masked_data, title=f'Observable from {observer_name}\n{obs_time.iso}', 
                unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
                coord='G', fig=fig.number, sub=(2, 2, 2))
    hp.graticule()
    # Mark Cas A
    hp.projplot(cas_a_gal.l.deg, cas_a_gal.b.deg, 'r*', markersize=15, 
                coord='G', lonlat=True)
    hp.projtext(cas_a_gal.l.deg, cas_a_gal.b.deg + 5, 'Cas A', 
                coord='G', lonlat=True, color='red', fontsize=10, 
                ha='center', weight='bold')
    
    # Cartesian view of observable region
    hp.cartview(masked_data, title=f'Observable Region (Cartesian)', 
                unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
                coord='G', fig=fig.number, sub=(2, 2, 3))
    hp.graticule()
    # Mark Cas A
    hp.projplot(cas_a_gal.l.deg, cas_a_gal.b.deg, 'r*', markersize=15, 
                coord='G', lonlat=True)
    hp.projtext(cas_a_gal.l.deg, cas_a_gal.b.deg + 5, 'Cas A', 
                coord='G', lonlat=True, color='red', fontsize=10, 
                ha='center', weight='bold')
    
    # Orthographic view centered on zenith
    # Calculate galactic coordinates of zenith
    zenith_altaz = SkyCoord(alt=90*u.deg, az=0*u.deg, 
                           frame=AltAz(obstime=obs_time, location=leiden))
    zenith_gal = zenith_altaz.galactic
    
    hp.orthview(masked_data, title=f'View from Zenith (Overhead)\nGal (l={zenith_gal.l.deg:.1f}°, b={zenith_gal.b.deg:.1f}°)', 
                unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
                coord='G', rot=(zenith_gal.l.deg, zenith_gal.b.deg, 0),
                fig=fig.number, sub=(2, 2, 4))
    hp.graticule()
    # Mark Cas A (in orthographic view)
    hp.projplot(cas_a_gal.l.deg, cas_a_gal.b.deg, 'r*', markersize=15, 
                coord='G', lonlat=True)
    hp.projtext(cas_a_gal.l.deg, cas_a_gal.b.deg + 3, 'Cas A', 
                coord='G', lonlat=True, color='red', fontsize=10, 
                ha='center', weight='bold')
    
    plt.suptitle(f'HI4PI Observable from Leiden at {obs_time.iso}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def main():
    # Load HI4PI data
    print("Loading HI4PI data...")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fits_image_filename = os.path.join(script_dir, 'HI4PI_vel_-100_100_N_Htot.fits')
    
    if not os.path.exists(fits_image_filename):
        print(f"Error: FITS file not found at {fits_image_filename}")
        print(f"Please ensure 'HI4PI_vel_-100_100_N_Htot.fits' is in {script_dir}")
        return
    
    hdul = fits.open(fits_image_filename)
    image = hdul[1].data
    data = image['N_H']
    healpix_map = data.flatten()
    nside = 1024  # From the FITS header
    print(f"Loaded {healpix_map.size} pixels with NSIDE={nside}")
    
    # Set observation time
    # Option 1: Use current time
    # obs_time = Time.now()
    
    # Option 2: Specify a future time
    # Example: November 15, 2025 at 22:00 UTC
    obs_time = Time('2025-11-21 16:00:00')
    
    # Option 3: Interactive - uncomment to ask user
    # time_str = input("Enter observation time (YYYY-MM-DD HH:MM:SS UTC): ")
    # obs_time = Time(time_str)
    
    print(f"\nCalculating observable region from Leiden at {obs_time.iso}")
    print(f"Leiden coordinates: {LEIDEN_LAT}, {LEIDEN_LON}")
    
    # Calculate observable region
    print("Computing which pixels are visible...")
    observable_mask, altaz_coords = get_observable_region(
        leiden, obs_time, min_elevation=20*u.deg, nside=nside
    )
    
    n_observable = np.sum(observable_mask)
    percent_observable = 100 * n_observable / len(observable_mask)
    print(f"Observable pixels: {n_observable} ({percent_observable:.1f}% of sky)")
    
    # Find the galactic coordinates of zenith
    zenith_altaz = SkyCoord(alt=90*u.deg, az=0*u.deg, 
                           frame=AltAz(obstime=obs_time, location=leiden))
    zenith_gal = zenith_altaz.galactic
    print(f"\nZenith (overhead) in Galactic coordinates:")
    print(f"  l = {zenith_gal.l.deg:.2f}°, b = {zenith_gal.b.deg:.2f}°")
    
    # Find some notable directions
    north_altaz = SkyCoord(alt=45*u.deg, az=0*u.deg, 
                          frame=AltAz(obstime=obs_time, location=leiden))
    north_gal = north_altaz.galactic
    print(f"North (45° elevation) in Galactic coordinates:")
    print(f"  l = {north_gal.l.deg:.2f}°, b = {north_gal.b.deg:.2f}°")
    
    # Export to FITS and PNG for simulation
    print("\nExporting observable region for simulation...")
    fits_file, png_file = export_observable_to_fits(
        healpix_map, observable_mask, obs_time, leiden,
        output_prefix="leiden_observable_80",
        pixel_size_arcmin=5.0,
        image_size_deg=80.0,
        center_coord=None  # None = zenith
    )
    
    # Plot the results
    print("\nGenerating HEALPix visualization...")
    plot_observable_region(healpix_map, observable_mask, obs_time)
    
    hdul.close()
    
    # Additional info: Show when notable objects are visible
    print("\nNotable Objects at this time:")
    
    # Galactic Center
    gal_center = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
    gal_center_altaz = gal_center.transform_to(AltAz(obstime=obs_time, location=leiden))
    print(f"\n  Galactic Center:")
    print(f"    Altitude: {gal_center_altaz.alt.deg:.2f}°")
    print(f"    Azimuth: {gal_center_altaz.az.deg:.2f}°")
    if gal_center_altaz.alt.deg > 20:
        print("    ✓ Visible!")
    else:
        print("    ✗ Below horizon or too low")
    
    # Cas A
    cas_a = SkyCoord(ra=350.85*u.deg, dec=58.815*u.deg, frame='icrs')
    cas_a_gal = cas_a.galactic
    cas_a_altaz = cas_a.transform_to(AltAz(obstime=obs_time, location=leiden))
    print(f"\n  Cas A (Cassiopeia A):")
    print(f"    Galactic: l={cas_a_gal.l.deg:.2f}°, b={cas_a_gal.b.deg:.2f}°")
    print(f"    Altitude: {cas_a_altaz.alt.deg:.2f}°")
    print(f"    Azimuth: {cas_a_altaz.az.deg:.2f}°")
    if cas_a_altaz.alt.deg > 20:
        print("    ✓ Visible! (Marked in red on plots)")
    else:
        print("    ✗ Below horizon or too low")


if __name__ == "__main__":
    main()
