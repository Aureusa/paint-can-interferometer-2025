import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings

# Try to use pyfftw if available (optional)
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
    pyfftw.interfaces.cache.enable()
    USE_PYFFTW = True
    print("Using pyfftw for FFTs")
except Exception:
    import numpy.fft as fft
    USE_PYFFTW = False
    print("pyfftw not available, using numpy.fft")

# -------------------------
# User parameters
# -------------------------
file = 'maxbl2000cm'
uv_file = f'uv_points_SCALAR_{file}.csv'


# Image/Grid parameters
grid_size = 4096           
pixel_scale_deg = 5.0 / 60.0   # degrees/pixel (5 arcmin => 0.083333... deg/pix)
primary_beam_deg = 80.0    # used only for display/zoom (not applied to UV data)
wavelength_m = 0.21        # observing wav

# Gridding kernel parameters (simple Gaussian)
kernel_support = 7         # kernel half-width in pixels
kernel_sigma = 1.5         # kernel sigma in pixels

# -------------------------
# Read UV points
# -------------------------
data = np.genfromtxt(uv_file, delimiter=',', skip_header=1)
if data.ndim == 1:
    data = data[np.newaxis, :]

u_klambda = data[:, 1].astype(float)
v_klambda = data[:, 2].astype(float)

n_uv = len(u_klambda)
print(f"Loaded {n_uv} UV points from '{uv_file}'")

# -------------------------
# Derived scales
# -------------------------
pixel_scale_rad = np.radians(pixel_scale_deg)
# Nyquist: uv_max in lambda units
uv_max_lam = 1.0 / (2.0 * pixel_scale_rad)
pixScaleFFT_lam = 1.0 / (grid_size * pixel_scale_rad)   # lambda units per FFT pixel
uv_max_kl = uv_max_lam / 1000.0

fov_deg = pixel_scale_deg * grid_size
print(f"Image plane: {grid_size}x{grid_size}, pixel = {pixel_scale_deg:.6f} deg ({pixel_scale_deg*3600:.1f}\"), FOV = {fov_deg:.2f} deg")
print(f"UV plane: uv_max = ±{uv_max_kl:.4f} kλ, uv pixel scale = {pixScaleFFT_lam/1000.0:.6f} kλ")

# -------------------------
# Create UV grid
# -------------------------
uv_grid = np.zeros((grid_size, grid_size), dtype=np.complex64)
count_grid = np.zeros_like(uv_grid, dtype=np.float32)   # for weights / uniform weighting

center = grid_size // 2
u_lam = u_klambda * 1e3
v_lam = v_klambda * 1e3

# compute integer pixel offsets relative to centre (round)
u_pix_offsets = np.round(u_lam / pixScaleFFT_lam).astype(int)
v_pix_offsets = np.round(v_lam / pixScaleFFT_lam).astype(int)

# build Gaussian kernel
k = kernel_support
xs = np.arange(-k, k+1)
ys = np.arange(-k, k+1)
xx, yy = np.meshgrid(xs, ys, indexing='xy')
kernel = np.exp(-0.5 * (xx**2 + yy**2) / (kernel_sigma**2))
kernel /= np.sum(kernel)


gridded = 0
for du, dv in zip(u_pix_offsets, v_pix_offsets):
    # positions in array indices
    u_c = center + du
    v_c = center + dv
    u_neg_c = center - du
    v_neg_c = center - dv

    # helper to add kernel around a center
    def add_kernel(grid, cx, cy, karr, weight=1.0):
        x0 = cx - k
        y0 = cy - k
        x1 = cx + k + 1
        y1 = cy + k + 1

        # compute overlaps with grid
        gx0 = max(x0, 0); gy0 = max(y0, 0)
        gx1 = min(x1, grid.shape[1]); gy1 = min(y1, grid.shape[0])
        if gx0 >= gx1 or gy0 >= gy1:
            return False

        # kernel slices
        kx0 = gx0 - x0; ky0 = gy0 - y0
        kx1 = kx0 + (gx1 - gx0); ky1 = ky0 + (gy1 - gy0)
        grid[gy0:gy1, gx0:gx1] += weight * karr[ky0:ky1, kx0:kx1]
        return True

    # natural weight = 1
    w = 1.0
    added1 = add_kernel(uv_grid, u_c, v_c, kernel, weight=w)
    added2 = add_kernel(uv_grid, u_neg_c, v_neg_c, kernel, weight=np.conj(w))
    add_kernel(count_grid, u_c, v_c, kernel, weight=1.0)
    add_kernel(count_grid, u_neg_c, v_neg_c, kernel, weight=1.0)

    if added1 or added2:
        gridded += 1

print(f"Gridded {gridded} samples (with conjugates)")

# Diagnostic: fraction of sampled uv cells (magnitude > 0)
sampled_frac = 100.0 * np.count_nonzero(np.abs(uv_grid) > 0) / (grid_size*grid_size)
print(f"UV grid sampling: {sampled_frac:.6f}% of grid cells non-zero")

# -------------------------
# Compute PSF: correct shift sequence
#    - uv_grid has DC (u=0,v=0) at centre (because we indexed relative to center)
#    - before inverse FFT we must move zero-frequency to [0,0] using ifftshift
#    - after ifft2 we shift image to center with fftshift
# -------------------------
# inverse FFT (complex)
uv_for_ifft = fft.ifftshift(uv_grid)   # move centre -> origin for ifft2
img = fft.ifft2(uv_for_ifft)
psf_complex = fft.fftshift(img)        # shift image so center corresponds to centre pixel
psf = np.abs(psf_complex)

# Normalize PSF
maxval = psf.max()
psf /= maxval

# -------------------------
# Plots
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1) UV coverage scatter (kλ)
ax = axes[0]
ax.scatter(u_klambda, v_klambda, s=1, alpha=0.6, label='+ samples')
ax.scatter(-u_klambda, -v_klambda, s=1, alpha=0.4, label='conjugates')
ax.set_xlabel('u (kλ)')
ax.set_ylabel('v (kλ)')
ax.set_title('UV coverage (kλ)')
ax.legend(loc='upper right', fontsize='small')
ax.set_aspect('equal', 'box')
ax.grid(True, alpha=0.25)

# 2) Gridded UV plane (display in kλ)
ax = axes[1]
uv_extent_kl = [-uv_max_kl, uv_max_kl, -uv_max_kl, uv_max_kl]
im = ax.imshow(np.abs(uv_grid), origin='lower', extent=uv_extent_kl)
ax.set_xlabel('u (kλ)')
ax.set_ylabel('v (kλ)')
ax.set_title('Gridded UV plane (abs)')
ax.set_aspect('equal', 'box')
plt.colorbar(im, ax=ax, label='gridded amplitude')

# 3) PSF zoomed region (use primary beam to compute zoom)
ax = axes[2]
center_idx = grid_size // 2

half_angle = primary_beam_deg / 2.0
show_half_angle = 1 * half_angle
show_pixels = int(np.round(show_half_angle / pixel_scale_deg))
show_pixels = max(16, min(show_pixels, grid_size//2 - 1))

psf_zoom = psf[center_idx-show_pixels:center_idx+show_pixels, center_idx-show_pixels:center_idx+show_pixels]

extent_deg = [-show_half_angle, show_half_angle, -show_half_angle, show_half_angle]
im2 = ax.imshow(psf_zoom, origin='lower', extent=extent_deg, cmap='hot')
ax.set_xlabel('l (deg)')
ax.set_ylabel('m (deg)')
ax.set_title('PSF (zoom)')

from matplotlib.patches import Circle
circle = Circle((0, 0), primary_beam_deg/2.0, fill=False, linestyle='--', linewidth=2, alpha=0.8, color='cyan')
ax.add_patch(circle)
plt.colorbar(im2, ax=ax, label='Normalized intensity')

np.savetxt(f'gridding_PSF_{file}.txt', psf_zoom)

plt.tight_layout()
plt.savefig('quick_psf_corrected_output.png', dpi=150)
print("Saved figure 'quick_psf_corrected_output.png'")
plt.show()

# -------------------------
# Simple PSF diagnostics
# -------------------------
psf_rms = np.std(psf)
print(f"PSF RMS = {psf_rms:.4e}")
rad = int(0.05 * grid_size)
mask = np.ones_like(psf, dtype=bool)
mask[center_idx-rad:center_idx+rad, center_idx-rad:center_idx+rad] = False
noise_std = np.std(psf[mask])
dr = psf.max() / (noise_std + 1e-20)
print(f"Estimated dynamic range (peak / std_of_sidelobes) = {dr:.1f}")

center_row = psf[center_idx, :]
half_max = 0.5
inds = np.where(center_row >= half_max)[0]

fwhm_pix = inds[-1] - inds[0]
fwhm_deg = fwhm_pix * pixel_scale_deg
print(f"Approx FWHM = {fwhm_deg:.4f} deg = {fwhm_deg*60.0:.3f} arcmin = {fwhm_deg*3600.0:.1f} arcsec")

