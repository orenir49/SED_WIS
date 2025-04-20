import numpy as np
from astropy.table import Table
from astropy.constants import G, M_sun, pc, R_sun, h, c, k_B
from scipy.interpolate import LinearNDInterpolator #, interp1d
from scipy.optimize import curve_fit
import astropy.units as u
from astroquery.gaia import Gaia
import os

# This is our band retrieval routines
from SEDer import band_retrieval_routines as brr
# This is our extinction routines
from SEDer import extinction_routines as extinction_routines
# This is our fitting routines
from SEDer import fitting_routines as fr
import sys



# -------------------------------------------------------
#                 § Utility routines
# -------------------------------------------------------

def get_distance(parallax):
    ## distance in meters calculated from parallax in mas
    return (1000/parallax) * pc.value 

def get_orbital_parameters(source_id):
    query = f'''SELECT parallax, parallax_error, period, period_error, eccentricity, eccentricity_error FROM gaiadr3.nss_two_body_orbit WHERE source_id = {int(source_id)}'''
    result = Gaia.launch_job(query)
    result = result.get_results()
    if len(result) > 0:
        return result['parallax'][0], result['parallax_error'][0], result['period'][0], result['period_error'][0], result['eccentricity_error'][0]
    else:
        print('Source not in NSS')
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 

# -------------------------------------------------------
#                 § SED modelling routines
# -------------------------------------------------------

# Get the relative path where the current file is stored, to refer
# properly to the path where the models are stored.
def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))
current_path = get_script_path()

bb = Table(np.genfromtxt(os.path.join(current_path, 'models', 'bb_sed.dat'), names=True, dtype=None))
kurucz = Table(np.genfromtxt(os.path.join(current_path, 'models', 'kurucz_sed.dat'), names=True, dtype=None))
koester = Table(np.genfromtxt(os.path.join(current_path, 'models', 'koester_sed.dat'), names=True, dtype=None))
co_da = Table(np.genfromtxt(os.path.join(current_path, 'models', 'CO_DA.dat'), names=True, dtype=None))
co_db = Table(np.genfromtxt(os.path.join(current_path, 'models', 'CO_DB.dat'), names=True, dtype=None))
he_da = Table(np.genfromtxt(os.path.join(current_path, 'models', 'He_wd.dat'), names=True, dtype=None))
one_da = Table(np.genfromtxt(os.path.join(current_path, 'models', 'ONe_DA.dat'), names=True, dtype=None))
one_db = Table(np.genfromtxt(os.path.join(current_path, 'models', 'ONe_DB.dat'), names=True, dtype=None))

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


def get_bb_sed(teff, r1, parallax, bands_table=None):
    """
    Calculate the SED for a blackbody star.

    Parameters:
    teff (float): Effective temperature of the star in Kelvin.
    r1 (float): Radius of the star in solar radii.
    parallax (float): Parallax of the star in milliarcseconds.
    bands_table (Table, optional): Table containing band information. If None, it will be retrieved using band_retrieval_routines.

    Returns:
    Table: Table containing the modeled fluxes for different bands.
    """
    if bands_table is None:
        bands_table = brr.get_bands_table()
    
    bb_tgrid = np.unique(bb['teff'])

    if bands_table is None:
        bands_table = brr.get_bands_table()

    model_table = Table(data=[[np.nan] for band in bands_table['wd_band']], names=bands_table['wd_band'])
    
    j_t = np.searchsorted(bb_tgrid, teff)
    
    if j_t == 0 or j_t == len(bb_tgrid):
        return model_table

    bbody = bb.copy()
    bbody = bbody[np.isin(bbody['teff'], [bb_tgrid[j_t-1], bb_tgrid[j_t]])]
    
    for band in bands_table['wd_band']:
        # interp = LinearNDInterpolator(np.array([bbody['teff']]).T, bbody[band])
        # model_table[band] = interp([teff])[0] * (r1 * R_sun.cgs * parallax / 1000 / pc.cgs)**2
        model_table[band] = np.interp(teff, bb_tgrid, bb[band]) * (r1 * R_sun.cgs * parallax / 1000 / pc.cgs)**2
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


def get_wd_koester_sed(teff, logg, parallax, bands_table=None):
    """
    Calculate the SED for a white dwarf using Koester models.

    Parameters:
    teff (float): Effective temperature of the star in Kelvin.
    logg (float): Surface gravity of the star in cm/s^2.
    parallax (float): Parallax of the star in milliarcseconds.
    
    Returns:
    Table: Table containing the modeled fluxes for different bands.
    """
    if bands_table is None:
        bands_table = brr.get_bands_table()

    # Use MR relation specific to core and atmosphere type
    mass = mass_from_MR_relation(teff, logg, 'CO', 'H')
    r = np.sqrt(G.cgs * mass * M_sun.cgs / (10**logg) / (u.cm / u.s**2)) / R_sun.cgs  # in R_sun

    # Interpolate Teff, Logg on the Koester model to get SED
    model = koester
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


def mass_from_MR_relation(teff, logg, core='CO', atm='H'):
    """
    Calculate the mass for a white dwarf given its effective temperature, surface gravity, core type, and atmosphere type.

    Parameters:
    teff (float): Effective temperature in Kelvin.
    logg (float): Surface gravity in cm/s^2.
    core (str): Core type ('He', 'CO', 'ONe'). Default is 'CO'.
    atm (str): Atmosphere type ('H', 'He'). Default is 'H'.

    Returns:
    float: Mass in solar masses.
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

    interp = LinearNDInterpolator(np.array([wd['Teff'], wd['log_g']]).T, wd['Mass'])
    m = interp([teff, logg])[0]
    return m


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
    AH1, AH2, AH3, AH4 = extinction_routines.get_HST_extinction(av, 0, teff)
    
    ext_dict = {'2MASS.J': AJ, '2MASS.H': AH, '2MASS.Ks': AKs, 'GALEX.FUV': AFUV, 'GALEX.NUV': ANUV, 'GAIA3.G': AG, 'GAIA3.Gbp': AGbp, 'GAIA3.Grp': AGrp,
                'WISE.W1': AW1, 'WISE.W2': AW2, 'WISE.W3': AW3, 'WISE.W4': AW4, 'Johnson.U': AU, 'Johnson.B': AB, 'Johnson.V': AV, 'Johnson.R': AR, 'Johnson.I': AI,
                'SDSS.u': Au, 'SDSS.g': Ag, 'SDSS.r': Ar, 'SDSS.i': Ai, 'SDSS.z': Az,
                'H1': AH1, 'H2': AH2, 'H3': AH3, 'H4': AH4}
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