#!/usr/bin/env python3
"""
Analyze pyuvsim simulation output and generate PSF
Compares with the quickpsf.py approach
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    """Main analysis function."""
    from pyuvdata import UVData
    
    # Configuration
    output_file = 'pyuvsim_output/scalar_sim.uvh5'

    print("=" * 60)
    print("PyUVSim Output Analysis")
    print("=" * 60)

    # Check if output file exists
    if not os.path.exists(output_file):
        print(f"ERROR: Output file not found: {output_file}")
        print("Run the simulation first:")
        print("  mpiexec -n 4 run_pyuvsim --param pyuvsim_obsparam.yaml")
        exit(1)

    # Load the simulated data
    print(f"\nLoading: {output_file}")
    uvd = UVData()
    uvd.read(output_file)

    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    # Handle telescope name attribute differences
    telescope_name = uvd.telescope.name if hasattr(uvd, 'telescope') else 'SCALAR'
    print(f"Telescope: {telescope_name}")
    print(f"Number of antennas: {uvd.Nants_telescope}")
    print(f"Number of baselines: {uvd.Nbls}")
    print(f"Number of times: {uvd.Ntimes}")
    print(f"Number of frequencies: {uvd.Nfreqs}")
    print(f"Number of polarizations: {uvd.Npols}")
    print(f"Frequency range: {uvd.freq_array.min()/1e6:.1f} - {uvd.freq_array.max()/1e6:.1f} MHz")
    print(f"Integration time: {uvd.integration_time[0]:.1f} seconds")
    print(f"Total observation time: {uvd.integration_time[0] * uvd.Ntimes / 60:.1f} minutes")

    # Get UV coordinates
    uvw = uvd.uvw_array  # Shape: (Nblts, 3) - u, v, w in meters
    # Handle freq_array - can be 1D or 2D depending on version
    freq_array = uvd.freq_array.flatten() if uvd.freq_array.ndim > 1 else uvd.freq_array
    wavelengths = 3e8 / freq_array  # Convert frequencies to wavelengths
    center_wavelength = wavelengths[len(wavelengths)//2]

    print(f"\nCenter wavelength: {center_wavelength*1000:.1f} mm")
    print(f"Center frequency: {freq_array[len(wavelengths)//2]/1e6:.1f} MHz")

    # Convert to wavelengths for center frequency
    u_lambda = uvw[:, 0] / center_wavelength
    v_lambda = uvw[:, 1] / center_wavelength

    print(f"\nUV coverage statistics:")
    print(f"  u range: [{u_lambda.min():.2f}, {u_lambda.max():.2f}] λ")
    print(f"  v range: [{v_lambda.min():.2f}, {v_lambda.max():.2f}] λ")
    print(f"  Max baseline: {np.sqrt((u_lambda**2 + v_lambda**2).max()):.2f} λ")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: UV coverage
    ax = axes[0, 0]
    for freq_idx in range(0, uvd.Nfreqs, max(1, uvd.Nfreqs//4)):  # Plot a few frequencies
        u = uvw[:, 0] / wavelengths[freq_idx]
        v = uvw[:, 1] / wavelengths[freq_idx]
        ax.scatter(u, v, s=2, alpha=0.3, label=f'{freq_array[freq_idx]/1e6:.0f} MHz')
        ax.scatter(-u, -v, s=2, alpha=0.3)

    ax.set_xlabel('u (λ)')
    ax.set_ylabel('v (λ)')
    ax.set_title('UV Coverage (All Times)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3)

    # Plot 2: UV coverage in meters (physical space)
    ax = axes[0, 1]
    ax.scatter(uvw[:, 0], uvw[:, 1], s=2, alpha=0.5)
    ax.scatter(-uvw[:, 0], -uvw[:, 1], s=2, alpha=0.5)
    ax.set_xlabel('u (m)')
    ax.set_ylabel('v (m)')
    ax.set_title('UV Coverage (Physical Space)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot 3: Visibility amplitudes vs baseline length
    ax = axes[1, 0]
    # Get visibility data for center frequency, first polarization
    # data_array shape is (Nblts, Nfreqs, Npols)
    vis_data = np.sum(uvd.data_array[:, uvd.Nfreqs//2, :], axis = 1)  # Shape: (Nblts,)
    baseline_lengths = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2)

    # Separate by time to show evolution
    colors = plt.cm.viridis(np.linspace(0, 1, uvd.Ntimes))
    for t_idx in range(0, uvd.Ntimes, max(1, uvd.Ntimes//10)):  # Plot subset of times
        mask = uvd.time_array == np.unique(uvd.time_array)[t_idx]
        bl_subset = baseline_lengths[mask]
        vis_subset = np.abs(vis_data[mask])
        ax.scatter(bl_subset, vis_subset, s=10, alpha=0.6, c=[colors[t_idx]], 
                   label=f't={t_idx}' if t_idx % max(1, uvd.Ntimes//3) == 0 else '')

    ax.set_xlabel('Baseline length (m)')
    ax.set_ylabel('Visibility amplitude')
    ax.set_title('Visibility Amplitudes')
    ax.grid(True, alpha=0.3)
    if uvd.Ntimes > 1:
        ax.legend()

    # Plot 4: Visibility phase vs baseline length
    ax = axes[1, 1]
    for t_idx in range(0, uvd.Ntimes, max(1, uvd.Ntimes//10)):
        mask = uvd.time_array == np.unique(uvd.time_array)[t_idx]
        bl_subset = baseline_lengths[mask]
        vis_subset = np.angle(vis_data[mask])
        ax.scatter(bl_subset, vis_subset, s=10, alpha=0.6, c=[colors[t_idx]],
                   label=f't={t_idx}' if t_idx % max(1, uvd.Ntimes//3) == 0 else '')

    ax.set_xlabel('Baseline length (m)')
    ax.set_ylabel('Visibility phase (radians)')
    ax.set_title('Visibility Phases')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    if uvd.Ntimes > 1:
        ax.legend()

    plt.tight_layout()
    plt.savefig('pyuvsim_analysis.png', dpi=150)
    print("\n✓ Analysis plot saved to: pyuvsim_analysis.png")
    plt.show()

    # Compute PSF using inverse FFT (similar to quickpsf.py)
    print("\n" + "=" * 60)
    print("Computing PSF from simulated data")
    print("=" * 60)

    # Use center frequency
    freq_idx = uvd.Nfreqs // 2
    wavelength = wavelengths[freq_idx]

    # Grid the UV data
    grid_size = 512 * 2
    pixel_scale_deg = 5.0 / 60.0  # 5 arcmin
    pixel_scale_rad = np.radians(pixel_scale_deg)
    fftScale_lam = 1.0 / pixel_scale_rad
    pixScaleFFT_lam = 2.0 * fftScale_lam / grid_size

    print(f"PSF calculation:")
    print(f"  Grid size: {grid_size}x{grid_size}")
    print(f"  Pixel scale: {pixel_scale_deg*3600:.1f} arcsec/pixel")
    print(f"  Field of view: {pixel_scale_deg*grid_size:.1f} degrees")

    # Create UV grid
    uv_grid = np.zeros((grid_size, grid_size), dtype=complex)

    # Average over all times
    u_lam = uvw[:, 0] / wavelength
    v_lam = uvw[:, 1] / wavelength

    u_pixels = ((u_lam / pixScaleFFT_lam) + grid_size / 2).astype(int)
    v_pixels = ((v_lam / pixScaleFFT_lam) + grid_size / 2).astype(int)

    # Grid the visibilities (average over times)
    gridded_count = 0
    for i, (u_pix, v_pix) in enumerate(zip(u_pixels, v_pixels)):
        if 0 <= u_pix < grid_size and 0 <= v_pix < grid_size:
            uv_grid[v_pix, u_pix] += 1.0
            gridded_count += 1
            # Add conjugate
            u_conj = grid_size - u_pix - 1
            v_conj = grid_size - v_pix - 1
            if 0 <= u_conj < grid_size and 0 <= v_conj < grid_size:
                uv_grid[v_conj, u_conj] += 1.0

    print(f"  Gridded {gridded_count} UV points")

    # Compute PSF
    from numpy.fft import ifft2, ifftshift
    psf = ifft2(uv_grid)
    psf = ifftshift(psf)
    psf = np.abs(psf)
    psf /= psf.max()

    # Plot PSF
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full PSF
    ax = axes[0]
    center = grid_size // 2
    extent = pixel_scale_deg * grid_size / 2
    extent_range = [-extent, extent, -extent, extent]
    im = ax.imshow(psf, origin='lower', cmap='hot', extent=extent_range, vmin=0, vmax=1)
    ax.set_xlabel('l (degrees)')
    ax.set_ylabel('m (degrees)')
    ax.set_title('Point Spread Function (PSF) - Full FOV')
    plt.colorbar(im, ax=ax, label='Normalized Intensity')

    # Zoomed PSF
    ax = axes[1]
    zoom = 100  # pixels
    psf_zoom = psf[center-zoom:center+zoom, center-zoom:center+zoom]
    extent = pixel_scale_deg * zoom
    extent_range = [-extent, extent, -extent, extent]
    im = ax.imshow(psf_zoom, origin='lower', cmap='hot', extent=extent_range, vmin=0, vmax=1)
    ax.set_xlabel('l (degrees)')
    ax.set_ylabel('m (degrees)')
    ax.set_title('Point Spread Function (PSF) - Zoomed')
    plt.colorbar(im, ax=ax, label='Normalized Intensity')

    plt.tight_layout()
    plt.savefig('pyuvsim_psf.png', dpi=150)
    print("✓ PSF plot saved to: pyuvsim_psf.png")
    plt.show()

    # PSF statistics
    print(f"\nPSF Statistics:")
    print(f"  Peak value: {psf.max():.4f}")
    print(f"  RMS: {np.std(psf):.4e}")

    # Estimate FWHM
    center_row = psf[center, :]
    half_max = center_row.max() / 2.0
    above_half = center_row > half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm_pixels = indices[-1] - indices[0]
        fwhm_deg = fwhm_pixels * pixel_scale_deg
        print(f"  FWHM: {fwhm_deg*60:.2f} arcmin = {fwhm_deg*3600:.1f} arcsec")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
