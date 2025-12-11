from casatasks import simobserve, tclean, simanalyze, impbcor
from os.path import dirname, join, abspath

# Resolve paths relative to this file so it works regardless of the cwd
BASE = dirname(abspath(__file__))
SKYMODEL = join(BASE, 'leiden_observable_2025-11-21T16-00-00.000.fits')
ANTCFG = join(BASE, 'SCALAR_90cm.cfg')

simobserve(
    project='cp',
    skymodel=SKYMODEL,
    # Use direction from FITS header to avoid WCS issues with close antennas
    # indirection='AZELGEO 0deg 90deg',  # Causes WCS error with small arrays
    # Use the native pixel size of the model (5 arcmin) to avoid unnecessary regrids
    incell='5arcmin',
    incenter='1.42GHz',
    inwidth='10MHz',
    # Start by disabling thermal noise to verify signal path; re‑enable once validated
    thermalnoise='',
    antennalist=ANTCFG,
    # Give the array a bit more time on source so any structure is obvious once noise is re‑enabled
    totaltime='600s',
    integration='10s',
    # Match the sky model timestamp for deterministic RA/LST bookkeeping
    refdate='2025/11/21/16:00:00',
    # Explicitly set interferometer mode to avoid primary beam issues
    obsmode='int',
    # Disable graphics to skip beam pattern calculations that cause WCS errors
    graphics='file'
)

# simanalyze(
#     project='cp',
#     image=True,
#     analyze=True,
#     # Force imaging geometry consistent with the model
#     imsize=[840, 840],
#     cell='5arcmin',
#     # Create a deconvolved image so structure is more apparent; adjust after verifying
#     niter=1000,
#     threshold='0.5Jy',
#     weighting='natural',
#     showuv=False,
#     showresidual=True,
#     showconvolved=True,
#     overwrite=True
# )