import numpy as np
from astropy.table import Table
from astropy.constants import G, M_sun, pc, R_sun, h, c, k_B
from scipy.interpolate import LinearNDInterpolator #, interp1d
from scipy.optimize import curve_fit
import astropy.units as u
from astroquery.gaia import Gaia


# This is our band retrieval routines
from SEDer import band_retrieval_routines as brr
# This is our extinction routines
from SEDer import extinction_routines as extinction_routines
# This is our SED model routines
from SEDer import sed_routines as sr


# -------------------------------------------------------
#                 § Utility routines
# -------------------------------------------------------

def get_distance(parallax,parallax_err):
    ## distance in meters calculated from parallax in mas
    return (1000/parallax) * pc.value , (1000/parallax**2)*parallax_err * pc.value

def get_logg(mass,radius):
    return np.log10(G.cgs*mass*M_sun.cgs/(radius*R_sun.cgs)**2 / u.cm/u.s**2).value

def get_mass(logg,radius):
    return (10**logg*(radius*R_sun.cgs)**2/(G.cgs*M_sun.cgs)).value
# -------------------------------------------------------
#                 § Model routines
# -------------------------------------------------------

def model_flux_ms(t, r, logg, meta, av, parallax):
    """
    Calculate the model flux vector for a given effective temperature, radius, logg,  metallicity, extinction, and parallax.
    The model flux is reddened using the provided extinction value.
    Parameters:
    -----------
    t : float
        Effective temperature of the source.
    r : float
        Radius of the source.
    logg : float    
        Surface gravity of the source (between 0 and 5 for Kurucz).
    meta : float
        Metallicity of the source (dex).
    av : float
        Extinction in the V band.
    parallax : float
        Parallax in milliarcsec.
    
    Returns:
    --------
    numpy.ndarray
        Model flux vector.
    """
    bands_table = brr.get_bands_table()
    m1 = get_mass(logg,r)
    ms = sr.get_MS_sed(t,m1,r,meta,parallax) # get the model flux
    ms = sr.redden_model_table(ms,t,av) # apply reddening
    model_flux = np.array([ms[bnd][0] for bnd in bands_table['wd_band']]) # organize the model fluxes into a vector
    return model_flux


def chi2_ms(obs_tbl, mod_params, bands_to_ignore=[]):
    """
    Calculate the chi-square value for a given set of model parameters and observed photometry.
    The chi-square value is calculated using the observed fluxes and the model fluxes.
    Parameters:
    -----------
    obs_tbl : astropy.table.Table
        Table containing observed photometry.
    mod_params : tuple
        Tuple containing the model parameters (Teff, R, logg, meta, av, parallax).
    bands_to_ignore : list, optional
        List of bands to ignore in the fitting. Default is empty.
    Returns:
    --------
    float
        Chi-square value.
    """
    bands_table = brr.get_bands_table()
    bands = np.array(list(bands_table['band']))
    obs_bands = bands[np.isin(bands, obs_tbl.colnames)]

    # retrieve model parameters, calculate model flux in all bands
    t, r, logg, meta, av, parallax = mod_params
    model_flux = model_flux_ms(t, r, logg, meta, av, parallax)

    # organize the observed fluxes and wavelengths into vectors
    flux = np.array([obs_tbl[0][bnd] for bnd in obs_bands])
    flux_err = np.array([obs_tbl[0][bnd + '_err'] for bnd in obs_bands])

    # mask np.nan values (bands that are not observed), and bands to ignore
    obs_mask = np.isfinite(flux) & ~np.isin(obs_bands, bands_to_ignore)
    mod_mask = np.isin(bands, obs_tbl.colnames) & np.isin(bands, obs_bands[obs_mask])
    

    model_flux = model_flux[mod_mask]
    flux = flux[obs_mask]
    flux_err = flux_err[obs_mask]

    # calculate the chi square value 
    chi2 = np.sum((flux - model_flux)**2 / flux_err**2)
    return chi2

# -------------------------------------------------------
#                 § SED fitting routines
# -------------------------------------------------------
def fit_MS_RTlogg(obs_tbl, meta, av, source_id=None, parallax=None, init_guess=[6000, 1, 4], bounds=[(3500, 0.1, 1), (10000, 5, 5)], bands_to_ignore=[]):
    """
    Fit the SED of a source using Kurucz models.
    Fitting parameters are effective temperature (Teff), radius (R) and surface gravity (logg).
    Fixed parameters are metallicity (meta), extinction (av) and parallax.

    Parameters:
    -----------
    obs_tbl : astropy.table.Table or None
        Table containing observed photometry. If None, photometry is retrieved using source_id.
    meta : float
        Metallicity of the source.
    av : float
        Extinction in the V band.
    source_id : int, optional
        Gaia DR3 source ID. Used to retrieve photometry if obs_tbl is None.
    parallax : float, optional
        Parallax in milliarcsec. If not provided, use value from obs_tbl.
    init_guess : list, optional
        Initial guess for the fitting parameters (Teff, R, logg). Default is [6000, 1, 4].
    bounds : list of tuples, optional
        Bounds for the fitting parameters (Teff, R, logg).
        Format is [(Teff_min, R_min, logg_min), (Teff_max, R_max, logg_max)]. Default is [(3500, 0.1, 1), (10000, 5, 5)].
    bands_to_ignore : list, optional
        List of bands to ignore in the fitting. Default is empty.
        Options: ['GALEX.FUV', 'GALEX.NUV', 'Johnson.U', 'SDSS.u', 'Johnson.B', 'SDSS.g',
                  'GAIA3.Gbp', 'Johnson.V', 'GAIA3.G', 'SDSS.r', 'Johnson.R', 'SDSS.i',
                  'GAIA3.Grp', 'Johnson.I', 'SDSS.z', '2MASS.J', '2MASS.H', '2MASS.Ks', 'WISE.W1',
                  'WISE.W2', 'WISE.W3']

    Returns:
    --------
    tuple
        A tuple containing the fitted parameters (Teff, R, logg) and the reduced chi-square value.
    """
    assert not (obs_tbl is None and source_id is None), "obs_tbl and source_id cannot be None at the same time"

    bands_table = brr.get_bands_table()

    # Get the observed data, if obs_tbl was not provided
    if obs_tbl is None:
        obs_tbl, flags = brr.get_photometry_single_source(source_id)

    # fixed parameters for the model
    if parallax is None:
        parallax = obs_tbl[0]['parallax']
    else: 
        parallax = parallax 

    # organize the observed fluxes and wavelengths into vectors
    bands = np.array(list(bands_table['band']))
    obs_bands = bands[np.isin(bands, obs_tbl.colnames)]
    wl = np.array([brr.get_lambda_eff(bnd) for bnd in obs_bands])
    flux = np.array([obs_tbl[0][bnd] for bnd in obs_bands])
    flux_err = np.array([obs_tbl[0][bnd + '_err'] for bnd in obs_bands])

    # mask np.nan values (bands that are not observed), and the bands to ignore
    obs_mask = np.isfinite(flux) & ~np.isin(obs_bands, bands_to_ignore)
    mod_mask = np.isin(bands, obs_tbl.colnames) & np.isin(bands, obs_bands[obs_mask])
    
    # apply the mask to create vectors for fitting
    x = wl[obs_mask]
    y = flux[obs_mask]
    y_err = flux_err[obs_mask]

    # define the model function; use the general function, but fix the metallicity, extinction, and parallax
    model_flux = lambda wl,t,r,logg: model_flux_ms(t,r,logg,meta,av,parallax)[mod_mask]
    # do a chi-square fit
    res = curve_fit(model_flux,x,y,p0=init_guess,bounds=bounds,sigma=y_err,absolute_sigma=True)
    t1_fit,r1_fit,logg_fit = res[0]
    t1_err,r1_err,logg_err = np.sqrt(np.diag(res[1]))

    # calculate the reduced chi-square value, while considering the residuals for ALL bands (including the ignored)
    chi2 = chi2_ms(obs_tbl, (t1_fit, r1_fit, logg_fit, meta, av, parallax), bands_to_ignore=[])
    dof = len(flux) - 3 # subtract three for the three fitting parameters T,R,logg
    redchi2 = chi2 / dof
    
    return t1_fit,t1_err,r1_fit,r1_err,logg_fit,logg_err,redchi2