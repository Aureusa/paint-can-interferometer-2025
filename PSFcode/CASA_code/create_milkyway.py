from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp

fits_image_filename = 'HI4PI_vel_-100_100_N_Htot.fits'

hdul = fits.open(fits_image_filename)
print("FITS structure:")
print(hdul.info())

# Extract HEALPix data
image = hdul[1].data
data = image['N_H']
print(f"\nData shape: {data.shape}")

# Flatten to 1D HEALPix array
healpix_map = data.flatten()
print(f"HEALPix map size: {healpix_map.size} pixels")


# Use healpy for proper visualization
print("\nCreating all-sky projection using healpy...")

# Create figure with multiple projections
fig = plt.figure(figsize=(16, 10))

# Mollweide projection (good for all-sky view)
hp.mollview(healpix_map, title='HI4PI All-Sky Map (Mollweide)', 
            unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
            coord='G', fig=fig.number, sub=(2, 2, 1))
hp.graticule()  # Add coordinate grid

# Cartesian projection
hp.cartview(healpix_map, title='HI4PI All-Sky Map (Cartesian)', 
            unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
            coord='G', fig=fig.number, sub=(2, 2, 2))
hp.graticule()

# Orthographic projection (centered on galactic center)
hp.orthview(healpix_map, title='HI4PI (Orthographic - Galactic Center)', 
            unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
            coord='G', rot=(0, 0), fig=fig.number, sub=(2, 2, 3))
hp.graticule()

# Gnomonic projection (zoomed in view)
hp.gnomview(healpix_map, title='HI4PI (Gnomonic - Galactic Center)', 
            unit='N_H [cm$^{-2}$]', norm='log', cmap='viridis',
            coord='G', rot=(0, 0), reso=10.0, fig=fig.number, sub=(2, 2, 4))
hp.graticule()

plt.tight_layout()
plt.show()

hdul.close()
