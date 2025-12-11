import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
from astropy import units as u
import glob
import os
from math import sqrt

def hdfig(subplots_def=None, scale=0.5, figsize=(8, 4.5)):
    fig = plt.figure(figsize=figsize, dpi=scale*1920/8)
    if subplots_def is None:
        return fig
    else:
        return fig, fig.subplots(*subplots_def) # What does the asterisk here mean?
    
def load_uv_from_csv(csv_path, wavelength=None, dtype=np.float32):
    """
    Load UV coordinates from a CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: baseline_id, u_klambda, v_klambda
    wavelength : astropy.units.Quantity, optional
        Wavelength to convert from kilolambda to meters. 
        If None, uses 1.42 GHz (21 cm line)
    dtype : numpy.dtype, optional
        Data type for the output array (default: np.float32)
        
    Returns
    -------
    uv_coords : astropy.units.Quantity
        UV coordinates array with shape (num_points, 2) and units of meters
        Compatible with compute_uv_coordinates output format but flattened
    """
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    u_klambda = data[:, 1]
    v_klambda = data[:, 2]
    
    # Default to 21 cm if no wavelength specified
    if wavelength is None:
        freq = 1.42e9 * u.Hz
        wavelength = (const.c / freq).to(u.m)
    
    # Convert kilolambda to meters
    u_m = u_klambda * 1000 * wavelength.value
    v_m = v_klambda * 1000 * wavelength.value
    
    # Stack into (num_points, 2) array with units
    uv_coords = np.stack([u_m, v_m], axis=1).astype(dtype) * u.m
    
    return uv_coords

def compute_uv_coordinates(ant_pos, dtype=np.float32):
    r'''
    Return an array of all uv-coordinates. The array has a shape of
    (num_ant, num_ant, 2), and should be of type `dtype`, and unit 
    `astropy.units.m`
    '''
    num_ant = len(ant_pos)
    u_ant = ant_pos[:,0]   # FILL_IN
    v_ant = ant_pos[:,1]   # FILL_IN
    result = np.zeros((num_ant, num_ant, 2), dtype=dtype)*u.m
    for ant1 in range(num_ant):
        for ant2 in range(num_ant):
            #result[ant1, ant2, 0] = u_ant[ant2] - u_ant[ant1]   # FILL_IN
            #result[ant1, ant2, 1] = v_ant[ant2] - v_ant[ant1]   # FILL_IN
            result[ant1, ant2] = ant_pos[ant2,:2] - ant_pos[ant1,:2]  #FILL_IN (alternative implementation)
    return result

def pixel_brightness_lightning(acm_single_pol, l, m, uv, frequency):
    a = acm_single_pol.astype(np.complex64)
    f = frequency.to(u.Hz).value
    uv_l = uv.to(u.m).value.astype(np.float32)
    uv_l *= f/const.c.value
    ll = l.to(u.rad).value
    mm = m.to(u.rad).value
    

    # CSV case with list of UV points (no ACM structure)
    # Assume all visibilities have amplitude 1 for PSF calculation
    num_vis = uv_l.shape[0]
    brightness = np.exp(-1*2j*np.pi*(uv_l[:,0]*ll + uv_l[:,1]*mm)).real.sum()
    return brightness/num_vis   

def make_image(acm, num_pix, uv, frequency, l_range=(-40.0, 40.0), m_range=(-40.0, 40.0)):
    r'''
    The returned image should have its origin pixel (0,0) in bottom-left corner (south-east).
    '''
    img = np.zeros((num_pix, num_pix), dtype=np.float32)
    l_coor = (np.linspace(*l_range, num_pix)*u.deg).to(u.rad)
    m_coor = (np.linspace(*m_range, num_pix)*u.deg).to(u.rad)

    for m_ix, m in enumerate(m_coor):
        for l_ix, l in enumerate(l_coor):
            img[m_ix, l_ix] = pixel_brightness_lightning(acm, l, m, uv, frequency)
    pixel_sep = m_coor[1]-m_coor[0]
    half = pixel_sep.to(u.rad).value/2.0
    img_extent =(l_coor[0].to(u.rad).value-half, l_coor[-1].to(u.rad).value+half, 
                 m_coor[0].to(u.rad).value-half, m_coor[-1].to(u.rad).value+half)    
    return img, img_extent

# Load uv points from CSV
freq = 1.42e9 * u.Hz
wavelength = (const.c / freq).to(u.m)
file = 'maxbl60cm'
uv = load_uv_from_csv(f"uv_points_SCALAR_{file}.csv", wavelength=wavelength)
over_sampling = 10
pixel_separation = ((wavelength / (0.6*u.m)).to(u.dimensionless_unscaled) / over_sampling)
num_pix = int(2.0 / pixel_separation)
psf, psf_extent = make_image(np.ones(shape=(9,9)), num_pix, uv, freq)
np.savetxt(f'michiel_PSF_{file}.txt', psf)

file = 'maxbl90cm'
uv = load_uv_from_csv(f"uv_points_SCALAR_{file}.csv", wavelength=wavelength)
over_sampling = 10
pixel_separation = ((wavelength / (0.9*u.m)).to(u.dimensionless_unscaled) / over_sampling)
num_pix = int(2.0 / pixel_separation)
psf, psf_extent = make_image(np.ones(shape=(9,9)), num_pix, uv, freq)
np.savetxt(f'michiel_PSF_{file}.txt', psf)

file = 'maxbl2000cm'
uv = load_uv_from_csv(f"uv_points_SCALAR_{file}.csv", wavelength=wavelength)
over_sampling = 3
pixel_separation = ((wavelength / (20*u.m)).to(u.dimensionless_unscaled) / over_sampling)
num_pix = int(2.0 / pixel_separation)
psf, psf_extent = make_image(np.ones(shape=(9,9)), num_pix, uv, freq)
np.savetxt(f'michiel_PSF_{file}.txt', psf)



# fig, ax = hdfig((1,1))
# ax.imshow(psf, extent=psf_extent, vmin=-0.05, vmax=1) # FILL_IN
# plt.show()
