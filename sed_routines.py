import numpy as np
from astropy.table import Table
from astropy.constants import G, M_sun, pc, R_sun, h, c, k_B
from scipy.interpolate import LinearNDInterpolator #, interp1d
from scipy.optimize import curve_fit
import astropy.units as u
import os

# This is our band retrieval routines
import band_retrieval_routines as brr
# This is our extinction routines
import extinction_routines as extinction_routines



# -------------------------------------------------------
#                 § Utility routines
# -------------------------------------------------------

def get_distance(parallax,parallax_err):
    ## distance in meters calculated from parallax in mas
    return (1000/parallax) * pc.value , (1000/parallax**2)*parallax_err * pc.value

# -------------------------------------------------------
#                 § SED modelling routines
# -------------------------------------------------------

kurucz = Table(np.genfromtxt(os.path.join('.', 'models', 'kurucz_sed.dat'), names=True, dtype=None))
co_da = Table(np.genfromtxt(os.path.join('.', 'models', 'CO_DA.dat'), names=True, dtype=None))
co_db = Table(np.genfromtxt(os.path.join('.', 'models', 'CO_DB.dat'), names=True, dtype=None))
he_da = Table(np.genfromtxt(os.path.join('.', 'models', 'He_wd.dat'), names=True, dtype=None))
one_da = Table(np.genfromtxt(os.path.join('.', 'models', 'ONe_DA.dat'), names=True, dtype=None))
one_db = Table(np.genfromtxt(os.path.join('.', 'models', 'ONe_DB.dat'), names=True, dtype=None))

def blackbody_spectrum(wavelength, temperature):
    """
    Calculate the blackbody flux for a given wavelength and temperature.

    Parameters
    ----------
    wavelength : `~astropy.units.Quantity`
        The wavelength at which to calculate the blackbody flux. Must have units of length.
    temperature : `~astropy.units.Quantity`
        The temperature of the blackbody. Must have units of temperature.

    Returns
    -------
    flux : `~astropy.units.Quantity`
        The blackbody flux at the given wavelength and temperature, with units of erg / (s * cm^2 * Å).

    Notes
    -----
    This function uses the Planck blackbody radiation formula to calculate the flux.
    """
    ## wavelength and temperature must have astropy units
    f = (2 * np.pi * h.cgs * c.cgs**2 / wavelength.cgs**5) / (np.exp(h.cgs * c.cgs / (wavelength.cgs * k_B.cgs * temperature)) - 1)
    return f.to(u.erg / u.s / u.cm**2 / u.AA)


def get_blackbody_sed(teff, radius, parallax, parallax_err, Av, bands_table=None):
    """
      ** Function previously known as blackbody_mod_table **
    Calculate the blackbody flux table for a star given its effective temperature, radius, parallax, parallax error, and extinction.

    Parameters:
    -----------
    teff : float
        Effective temperature of the star in Kelvin.
    radius : float
        Radius of the star in solar radii.
    parallax : float
        Parallax of the star in milliarcseconds.
    parallax_err : float
        Error in the parallax measurement in milliarcseconds.
    Av : float
        Extinction in the V band.

    Returns:
    --------
    mod_tbl : astropy.table.Table
        Table containing the modeled fluxes for different bands, corrected for extinction.

    Notes:
    ------
    This function uses several extinction models to correct the fluxes for different bands. The bands considered are from 2MASS, GALEX, GAIA, WISE, Johnson, and SDSS surveys.
    """
    if bands_table is None:
        bands_table = brr.get_bands_table()

    bands    = list(bands_table['band'])
    mod_tbl  = Table(dict(zip(bands, [[float(1)] for i in range(len(bands))])))
    d, d_err = get_distance(parallax, parallax_err)
    radius   = radius * R_sun.value

    AW1, AW2, AW3, AW4 = extinction_routines.get_WISE_extinction(Av, 0, teff)
    AJ, AH, AKs        = extinction_routines.get_2MASS_extinction(Av, 0, teff)
    AU, AB, AV, AR, AI = extinction_routines.get_Johnson_extinction(Av, 0, teff)
    Au, Ag, Ar, Ai, Az = extinction_routines.get_SDSS_extinction(Av, 0, teff)
    AG, AGbp, AGrp     = extinction_routines.get_Gaia_extinction(Av, 0, teff)
    AFUV, ANUV         = extinction_routines.get_Galex_extinction(Av, 0, teff)

    ext_dict = {'2MASS.J': AJ, '2MASS.H': AH, '2MASS.Ks': AKs, 'GALEX.FUV': AFUV, 'GALEX.NUV': ANUV, 'GAIA3.G': AG, 'GAIA3.Gbp': AGbp, 'GAIA3.Grp': AGrp,
                'WISE.W1': AW1, 'WISE.W2': AW2, 'WISE.W3': AW3, 'WISE.W4': AW4, 'Johnson.U': AU, 'Johnson.B': AB, 'Johnson.V': AV, 'Johnson.R': AR, 'Johnson.I': AI,
                'SDSS.u': Au, 'SDSS.g': Ag, 'SDSS.r': Ar, 'SDSS.i': Ai, 'SDSS.z': Az}

    for b in bands:
        filepath      = os.path.join('..', 'data', 'VOSA', 'filters', b + '.dat')
        filter_tbl    = Table.read(filepath, format='ascii', names=['wavelength', 'transmission'])
        wavelength    = filter_tbl['wavelength'].data * u.AA
        transmission  = filter_tbl['transmission'].data 
        flux          = blackbody_spectrum(wavelength, teff * u.K)
        flux          = np.dot(flux, transmission)
        flux         /= np.sum(transmission)
        flux          = flux * (radius / d)**2
        mod_tbl[0][b] = flux.value
        mod_tbl[b].unit = flux.unit 
        mod_tbl[b]      = mod_tbl[b] * 10**(-0.4 * ext_dict[b])

    return mod_tbl


def get_MS_sed(teff, m1, r1, meta, parallax, bands_table=None, return_logg=False):
    """
    Calculate the SED for a main sequence star using Kurucz models and optionally return the log(g).

    Parameters:
    -----------
    teff : float
        Effective temperature of the star in Kelvin.
    m1 : float
        Mass of the star in solar masses.
    r1 : float
        Radius of the star in solar radii.
    meta : float
        Metallicity of the star.
    parallax : float
        Parallax of the star in milliarcseconds.
    bands_table : Table, optional
        Table containing band information. If None, it will be retrieved using band_retrieval_routines.
    return_logg : bool, optional
        If True, the function will return the log(g) value along with the model table.

    Returns:
    --------
    tuple
        A tuple containing the model table with fluxes for different bands and the log(g) value if return_logg is True.
    """

    kurucz_tgrid = np.unique(kurucz['teff'])
    kurucz_ggrid = np.unique(kurucz['logg'])
    kurucz_metagrid = np.unique(kurucz['meta'])


    if bands_table is None:
        bands_table = brr.get_bands_table()

    model_table = Table(data=[[np.nan] for band in bands_table['wd_band']], names=bands_table['wd_band'])
    logg = np.log10((G.cgs * m1 * M_sun.cgs / (r1 * R_sun.cgs)**2).value)

    j_t, j_g, j_meta = np.searchsorted(kurucz_tgrid, teff), np.searchsorted(kurucz_ggrid, logg), np.searchsorted(kurucz_metagrid, meta)

    if j_t == 0 or j_g == 0 or j_meta == 0:
        return model_table, logg if return_logg else model_table
    elif j_t == len(kurucz_tgrid) or j_g == len(kurucz_ggrid) or j_meta == len(kurucz_metagrid):
        return model_table, logg if return_logg else model_table

    kur = kurucz.copy()
    kur = kur[np.isin(kur['teff'], [kurucz_tgrid[j_t-1], kurucz_tgrid[j_t]])]
    kur = kur[np.isin(kur['logg'], [kurucz_ggrid[j_g-1], kurucz_ggrid[j_g]])]
    kur = kur[np.isin(kur['meta'], [kurucz_metagrid[j_meta-1], kurucz_metagrid[j_meta]])]

    for band in bands_table['wd_band']:
        interp = LinearNDInterpolator(np.array([kur['teff'], kur['logg'], kur['meta']]).T, kur[band])
        model_table[band] = interp([teff, logg, meta])[0] * (r1 * R_sun.cgs * parallax / 1000 / pc.cgs)**2
        model_table[band].unit = u.erg / u.s / u.cm**2 / u.AA

    if return_logg:
        return model_table, logg
    else:
        return model_table


def get_RG_sed(teff, m1, r1, meta, parallax, bands_table=None):
    """
    Calculate the SED for a red giant star using Kurucz models.

    Parameters:
    -----------
    teff : float
        Effective temperature of the star in Kelvin.
    m1 : float
        Mass of the star in solar masses.
    r1 : float
        Radius of the star in solar radii.
    meta : float
        Metallicity of the star.
    parallax : float
        Parallax of the star in milliarcseconds.
    bands_table : Table, optional
        Table containing band information. If None, it will be retrieved using band_retrieval_routines.

    Returns:
    --------
    model_table : Table
        Table containing the modeled fluxes for different bands.
    """

    kurucz_tgrid = np.unique(kurucz['teff'])
    kurucz_ggrid = np.unique(kurucz['logg'])
    kurucz_metagrid = np.unique(kurucz['meta'])

    if bands_table is None:
        bands_table = brr.get_bands_table()
        
    kur = kurucz.copy()
    model_table = Table(data=[[np.nan] for band in bands_table['wd_band']], names=bands_table['wd_band'])
    logg = np.log10((G.cgs * m1 * M_sun.cgs / (r1 * R_sun.cgs)**2).value)
    j_t, j_g, j_meta = np.searchsorted(kurucz_tgrid, teff), np.searchsorted(kurucz_ggrid, logg), np.searchsorted(kurucz_metagrid, meta)

    if j_t == 0 or j_g == 0 or j_meta == 0:
        return model_table
    elif j_t == len(kurucz_tgrid) or j_g == len(kurucz_ggrid) or j_meta == len(kurucz_metagrid):
        return model_table

    kur = kur[np.isin(kur['teff'], [kurucz_tgrid[j_t-1], kurucz_tgrid[j_t]])]
    kur = kur[np.isin(kur['logg'], [kurucz_ggrid[j_g-1], kurucz_ggrid[j_g]])]
    kur = kur[np.isin(kur['meta'], [kurucz_metagrid[j_meta-1], kurucz_metagrid[j_meta]])]

    for band in bands_table['wd_band']:
        interp = LinearNDInterpolator(np.array([kur['teff'], kur['logg'], kur['meta']]).T, kur[band])
        model_table[band] = interp([teff, logg, meta])[0] * (r1 * R_sun.cgs * parallax / 1000 / pc.cgs)**2
        model_table[band].unit = u.erg / u.s / u.cm**2 / u.AA

    return model_table


def get_wd_bedard_sed(teff, m, parallax, core='CO', atm='H', bands_table=None):
    """
    Calculate the SED for a white dwarf using Bedard models.

    Parameters:
    teff (float): Effective temperature of the star in Kelvin.
    m (float): Mass of the star in solar masses.
    parallax (float): Parallax of the star in milliarcseconds.
    core (str): Core type ('CO'). Default is 'CO'.
    atm (str): Atmosphere type ('H', 'He'). Default is 'H'.

    Returns:
    Table: Table containing the modeled fluxes for different bands.
    """

    if bands_table is None:
        bands_table = brr.get_bands_table()

    # Use MR relation specific to core and atmosphere type
    logg = logg_from_MR_relation(teff, m, core, atm)
    model_table = Table(data=[[np.nan] for band in bands_table['wd_band']], names=bands_table['wd_band'])

    # Select the appropriate model based on atmosphere type
    if atm == 'H':
        model = co_da.copy()
    elif atm == 'He':
        model = co_db.copy()
    else:
        print('atm either H or He')
        return None
    
    bedard_tgrid = np.unique(model['Teff'])
    bedard_ggrid = np.unique(model['log_g'])

    # Find the indices for interpolation
    j_t, j_g = np.searchsorted(bedard_tgrid, teff), np.searchsorted(bedard_ggrid, logg)
    if j_t == 0 or j_g == 0 or j_t == len(bedard_tgrid) or j_g == len(bedard_ggrid):
        return model_table

    # Filter the model for interpolation
    cut = np.isin(model['Teff'], [bedard_tgrid[j_t-1], bedard_tgrid[j_t]]) & np.isin(model['log_g'], [bedard_ggrid[j_g-1], bedard_ggrid[j_g]])
    if np.count_nonzero(cut) > 2:
        model = model[cut]

    # Interpolate and calculate the flux for each band
    for wd_band, f0, m0 in zip(bands_table['wd_band'], bands_table['f0'], bands_table['m0']):
        try:
            interp = LinearNDInterpolator(np.array([model['Teff'], model['log_g']]).T, model[wd_band])
            absmag = interp([teff, logg])[0]
            apmag = absmag + 5 * np.log10(1000 / parallax) - 5
            model_table[wd_band] = f0 * 10**(-0.4 * (apmag - m0))
        except:
            model_table[wd_band] = np.nan
        model_table[wd_band].unit = u.erg / u.s / u.cm**2 / u.AA

    return model_table


def get_wd_koester_sed(teff, m, parallax, koester_wd_model, core='CO', atm='H', bands_table=None):
    """
    Calculate the SED for a white dwarf using Koester models.

    Parameters:
    teff (float): Effective temperature of the star in Kelvin.
    m (float): Mass of the star in solar masses.
    parallax (float): Parallax of the star in milliarcseconds.
    koester_wd_model (str): Path to the Koester WD model file.
    core (str): Core type ('CO'). Default is 'CO'.
    atm (str): Atmosphere type ('H', 'He'). Default is 'H'.

    Returns:
    Table: Table containing the modeled fluxes for different bands.
    """
    if bands_table is None:
        bands_table = brr.get_bands_table()

    # Use MR relation specific to core and atmosphere type
    logg = logg_from_MR_relation(teff, m, core, atm)
    r = np.sqrt(G.cgs * m * M_sun.cgs / (10**logg) / (u.cm / u.s**2)) / R_sun.cgs  # in R_sun

    # Interpolate Teff, Logg on the Koester model to get SED
    model = np.genfromtxt(koester_wd_model, names=True, dtype=None)
    model_table = Table(data=[[np.nan] for band in bands_table['wd_band']], names=bands_table['wd_band'])
    
    for wd_band in bands_table['wd_band']:
        interp = LinearNDInterpolator(np.array([model['teff'], model['logg']]).T, model[wd_band])
        model_table[wd_band] = interp([teff, logg])[0] * (r * R_sun.cgs * parallax / 1000 / pc.cgs)**2
        model_table[wd_band].unit = u.erg / u.s / u.cm**2 / u.AA
    
    return model_table


def logg_from_MR_relation(teff, m, core='CO', atm='H'):
    """
    Calculate the surface gravity (logg) for a white dwarf given its effective temperature, mass, core type, and atmosphere type.

    Parameters:
    teff (float): Effective temperature in Kelvin.
    m (float): Mass in solar masses.
    core (str): Core type ('He', 'CO', 'ONe'). Default is 'CO'.
    atm (str): Atmosphere type ('H', 'He'). Default is 'H'.

    Returns:
    float: Surface gravity (logg) in cm/s^2.
    """
    if core == 'He':
        wd = he_da.copy()
    elif core == 'CO':
        if atm == 'H':
            wd = co_da.copy()
        elif atm == 'He':
            wd = co_db.copy()
        else:
            print('atm either H or He')
            return None
    elif core == 'ONe':
        if atm == 'H':
            wd = one_da.copy()
        elif atm == 'He':
            wd = one_db.copy()
        else:
            print('atm either H or He')
            return None

    interp = LinearNDInterpolator(np.array([wd['Teff'], wd['Mass']]).T, wd['log_g'])
    logg = interp([teff, m])[0]
    return logg


def redden_model_table(mod_tbl,teff,av, bands_table=None):
    """
    Apply reddening to a model table based on extinction values for various bands.
    Parameters:
    mod_tbl (Table): The model table containing stellar properties.
    teff (float): The effective temperature of the star.
    av (float): The visual extinction value.
    bands_table (Table, optional): A table containing band information. If None, it will be retrieved using brr.get_bands_table().
    Returns:
    Table: The reddened model table with updated values based on extinction.
    Notes:
    - The function uses various extinction routines to get extinction values for different photometric bands.
    - The extinction values are applied to the model table by modifying the flux values in the specified bands.
    """
    if bands_table is None:
        bands_table = brr.get_bands_table()
        
    AW1, AW2, AW3, AW4 = extinction_routines.get_WISE_extinction(av, 0, teff)
    AJ, AH, AKs        = extinction_routines.get_2MASS_extinction(av, 0, teff)
    AU, AB, AV, AR, AI = extinction_routines.get_Johnson_extinction(av, 0, teff)
    Au, Ag, Ar, Ai, Az = extinction_routines.get_SDSS_extinction(av, 0, teff)
    AG, AGbp, AGrp     = extinction_routines.get_Gaia_extinction(av, 0, teff)
    AFUV, ANUV         = extinction_routines.get_Galex_extinction(av, 0, teff)
    
    ext_dict = {'2MASS.J': AJ, '2MASS.H': AH, '2MASS.Ks': AKs, 'GALEX.FUV': AFUV, 'GALEX.NUV': ANUV, 'GAIA3.G': AG, 'GAIA3.Gbp': AGbp, 'GAIA3.Grp': AGrp,
                'WISE.W1': AW1, 'WISE.W2': AW2, 'WISE.W3': AW3, 'WISE.W4': AW4, 'Johnson.U': AU, 'Johnson.B': AB, 'Johnson.V': AV, 'Johnson.R': AR, 'Johnson.I': AI,
                'SDSS.u': Au, 'SDSS.g': Ag, 'SDSS.r': Ar, 'SDSS.i': Ai, 'SDSS.z': Az}
    for band, col in zip(bands_table['band'], bands_table['wd_band']):
        mod_tbl[col][0] = mod_tbl[col][0] * 10**(-0.4 * ext_dict[band])
        
    return mod_tbl


def get_cooling_age(teff, m, core='CO', atm='H', model_path=None):
    """
    Calculate the cooling age for a white dwarf given its effective temperature, mass, core type, and atmosphere type.

    Parameters:
    -----------
    teff : float
        Effective temperature of the white dwarf in Kelvin.
    m : float
        Mass of the white dwarf in solar masses.
    core : str, optional
        Core type ('He', 'CO', 'ONe'). Default is 'CO'.
    atm : str, optional
        Atmosphere type ('H', 'He'). Default is 'H'.
    model_path : str, optional
        Path to the directory containing the white dwarf models. Default is current directory.

    Returns:
    --------
    age : float
        Cooling age of the white dwarf in years.

    Notes:
    ------
    This function uses white dwarf cooling models to interpolate the cooling age based on the given parameters.
    """
    if model_path is None:
        model_path = '.'

    if core == 'He':
        model_file = os.path.join(model_path, 'He_wd.dat')
    elif core == 'CO':
        if atm == 'H':
            model_file = os.path.join(model_path, 'CO_DA.dat')
        elif atm == 'He':
            model_file = os.path.join(model_path, 'CO_DB.dat')
        else:
            print('atm either H or He')
            return None
    elif core == 'ONe':
        if atm == 'H':
            model_file = os.path.join(model_path, 'ONe_DA.dat')
        elif atm == 'He':
            model_file = os.path.join(model_path, 'ONe_DB.dat')
        else:
            print('atm either H or He')
            return None
    else:
        print('core either He, CO, or ONe')
        return None

    model = np.genfromtxt(model_file, names=True, dtype=None)
    interp = LinearNDInterpolator(np.array([model['Teff'], model['Mass']]).T, model['Age'])
    age = interp([teff, m])[0]
    if np.isnan(age):
        return np.nan
    return age


def get_cooling_temp(age, m, core='CO', atm='H', model_path=None):
    """
    Calculate the effective temperature for a white dwarf given its cooling age, mass, core type, and atmosphere type.

    Parameters:
    -----------
    age : float
        Cooling age of the white dwarf in years.
    m : float
        Mass of the white dwarf in solar masses.
    core : str, optional
        Core type ('He', 'CO', 'ONe'). Default is 'CO'.
    atm : str, optional
        Atmosphere type ('H', 'He'). Default is 'H'.
    model_path : str, optional
        Path to the directory containing the white dwarf models. Default is current directory.

    Returns:
    --------
    teff : float
        Effective temperature of the white dwarf in Kelvin.

    Notes:
    ------
    This function uses white dwarf cooling models to interpolate the effective temperature based on the given parameters.
    """
    if model_path is None:
        model_path = '.'

    if core == 'He':
        model_file = os.path.join(model_path, 'He_wd.dat')
    elif core == 'CO':
        if atm == 'H':
            model_file = os.path.join(model_path, 'CO_DA.dat')
        elif atm == 'He':
            model_file = os.path.join(model_path, 'CO_DB.dat')
        else:
            print('atm either H or He')
            return None
    elif core == 'ONe':
        if atm == 'H':
            model_file = os.path.join(model_path, 'ONe_DA.dat')
        elif atm == 'He':
            model_file = os.path.join(model_path, 'ONe_DB.dat')
        else:
            print('atm either H or He')
            return None
    else:
        print('core either He, CO, or ONe')
        return None

    model = np.genfromtxt(model_file, names=True, dtype=None)
    interp = LinearNDInterpolator(np.array([model['Age'], model['Mass']]).T, model['Teff'])
    teff_init = interp([age, m])[0]

    teff_vec = np.linspace(teff_init + 10000, teff_init - 10000, 20)
    age_vec = [get_cooling_age(teff, m, core, atm, model_path) for teff in teff_vec]
    teff = np.interp(age, age_vec, teff_vec)
    return teff


# -------------------------------------------------------
#                 § SED fitting routines
# -------------------------------------------------------

def fit_MS_RT(source_id,m1,meta,av,init_guess=[6000,1],bounds=[(3500,0.1),(10000,5)],bands_to_ignore=[]):
    """
    Fit the SED of a source using Kurucz models.
    Fitting parameters are effective temperature (Teff) and radius (R).
    Fixed parameters are mass (m1), metallicity (meta), and extinction (av).

    Parameters:
    -----------
    source_id : int
        Gaia DR3 source ID.
    m1 : float
        Mass of the source in solar masses.
    meta : float
        Metallicity of the source.
    av : float
        Extinction in the V band.
    init_guess : list, optional
        Initial guess for the fitting parameters (Teff, R). Default is [6000, 1].
    bounds : list of tuples, optional
        Bounds for the fitting parameters (Teff, R).
        Format is [(Teff_min, R_min), (Teff_max, R_max)]. Default is [(3500, 0.1), (10000, 5)].
    bands_to_ignore : list, optional
        List of bands to ignore in the fitting. Default is empty.
        Options: ['GALEX.FUV' 'GALEX.NUV' 'Johnson.U' 'SDSS.u' 'Johnson.B' 'SDSS.g'
                    'GAIA3.Gbp' 'Johnson.V' 'GAIA3.G' 'SDSS.r' 'Johnson.R' 'SDSS.i'
                        'GAIA3.Grp' 'Johnson.I' 'SDSS.z' '2MASS.J' '2MASS.H' '2MASS.Ks' 'WISE.W1'
                             'WISE.W2' 'WISE.W3']
    Returns:
    --------
    tuple
        A tuple containing the fitted parameters (Teff, R) and the reduced chi-square value.
    """
    # get observed fluxes for the source
    obs_tbl = brr.get_photometry_single_source(source_id)

    # organize the observed fluxes and wavelengths into vectors
    bands_table = brr.get_bands_table()
    bnds = list(bands_table['band'])
    wl = np.array(list(bands_table['lambda_eff']))
    flux = np.array([obs_tbl[0][bnd] for bnd in bnds])
    flux_err = np.array([obs_tbl[0][bnd + '_err'] for bnd in bnds])

    # fixed parameters for the model
    parallax = obs_tbl[0]['parallax']
    meta = meta
    av = av
    m1 = m1

    def model_flux(x,t,r):
        ms = get_MS_sed(t,m1,r,meta,parallax) # get the model flux
        ms = redden_model_table(ms,t,av) # apply reddening
        model_flux = np.array([ms[bnd][0] for bnd in bands_table['wd_band']]) # organize the model fluxes into a vector
        wl = list(bands_table['lambda_eff'])
        cut = np.isin(wl,x) # only consider the bands that are not masked
        model_flux = model_flux[cut]
        return model_flux
    
    # mask np.nan values (bands that are not observed)
    mask = np.isfinite(flux)
    bnds = np.array(bnds)[mask]
    wl = wl[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    # mask the bands to ignore
    mask = np.ones(len(bnds), dtype=bool)
    for band in bands_to_ignore:
        mask[bnds.index(band)] = False
    x = wl[mask]
    y = flux[mask]
    y_err = flux_err[mask]

    # do a chi-square fit
    res = curve_fit(model_flux,x,y,p0=init_guess,bounds=bounds,sigma=y_err,absolute_sigma=True)
    t1_fit,r1_fit = res[0]
    t1_err,r1_err = np.sqrt(np.diag(res[1]))

    # calculate the reduced chi-square value, while considering the residuals for ALL bands (including the masked ones)
    chi2 = np.sum((flux - model_flux(wl,t1_fit,r1_fit))**2 / flux_err**2)
    dof = len(flux) - 2 # subtract two for the two fitting parameters
    redchi2 = chi2 / dof
    
    return t1_fit,t1_err,r1_fit,r1_err,redchi2