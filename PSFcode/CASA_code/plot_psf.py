import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def _default_psf_path() -> str:
	"""Return absolute path to cp/small_CASA_psf.fits next to this script."""
	base = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(base, 'cp', 'cp.SCALAR_90cm.quick.psf.fits')


def plot_psf(path: str | None = None, save_pdf: bool = True):
	"""Load a FITS PSF and plot with correct angular axes.

	- If path is None, uses cp/small_CASA_psf.fits relative to this file.
	- Axis units are degrees, derived from the FITS WCS if present.
	- Saves a PDF next to the FITS when save_pdf=True.
	"""
	if path is None:
		path = _default_psf_path()

	with fits.open(path) as hdul:
		hdu = hdul[0]
		data = np.asarray(hdu.data, dtype=float)
		header = hdu.header

	# Squeeze singleton axes
	data = np.squeeze(data)
	if data.ndim != 2:
		# Handle common shapes (ny, nx) or (1,1,ny,nx)
		ny, nx = data.shape[-2], data.shape[-1]
		data = data.reshape(ny, nx)
	ny, nx = data.shape

	# Determine pixel scale from WCS (degrees per pixel)
	try:
		wcs = WCS(header, naxis=2)
		scales_deg = proj_plane_pixel_scales(wcs)  # degrees/pixel when WCS in degrees
		dy_deg, dx_deg = float(scales_deg[1]), float(scales_deg[0])
	except Exception:
		dx_deg = abs(float(header.get('CDELT1', 1.0)))
		dy_deg = abs(float(header.get('CDELT2', 1.0)))

	half_x = (dx_deg * nx) / 2.0
	half_y = (dy_deg * ny) / 2.0

	title = os.path.basename(path)
	plt.figure()
	plt.title(title)
	plt.imshow(data, origin='lower', extent=[-half_x, half_x, -half_y, half_y], cmap='viridis')
	plt.ylabel('Sky position from centre (degrees)')
	plt.xlabel('Sky position from centre (degrees)')
	plt.colorbar(label='PSF (arbitrary)')

	if save_pdf:
		out_path = os.path.splitext(path)[0] + '.pdf'
		plt.savefig(out_path, dpi=150, bbox_inches='tight')

	plt.show()
	return data


if __name__ == '__main__':
	in_path = sys.argv[1] if len(sys.argv) > 1 else None
	data = plot_psf(in_path)
	np.savetxt('casa_PSF_maxbl90cm.txt', data)

