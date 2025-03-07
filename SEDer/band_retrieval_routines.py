import numpy as np
from astropy.table import Table
from astropy import table
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy.constants import G, M_sun, pc, R_sun, h, c, k_B
import astropy.units as u

from astroquery.exceptions import NoResultsWarning
import warnings

# Suppress the NoResultsWarning
warnings.filterwarnings('ignore', category=NoResultsWarning)


filter_band_dict = {'GALEX/GALEX.FUV':'GALEX.FUV','GALEX/GALEX.NUV':'GALEX.NUV','GAIA/GAIA3.G':'GAIA3.G','GAIA/GAIA3.Gbp':'GAIA3.Gbp','GAIA/GAIA3.Grp':'GAIA3.Grp',
                    'Generic/Johnson.U':'Johnson.U','Generic/Johnson.B':'Johnson.B','Generic/Johnson.V':'Johnson.V','Generic/Johnson.R':'Johnson.R','Generic/Johnson.I':'Johnson.I',
                    'SLOAN/SDSS.u':'SDSS.u','SLOAN/SDSS.g':'SDSS.g','SLOAN/SDSS.r':'SDSS.r','SLOAN/SDSS.i':'SDSS.i','SLOAN/SDSS.z':'SDSS.z',
                    '2MASS/2MASS.J':'2MASS.J','2MASS/2MASS.H':'2MASS.H','2MASS/2MASS.Ks':'2MASS.Ks',
                    'WISE/WISE.W1':'WISE.W1','WISE/WISE.W2':'WISE.W2','WISE/WISE.W3':'WISE.W3'}

filter_list = ['Obs.Flux_2MASS.H', 'Obs.Flux_2MASS.J', 'Obs.Flux_2MASS.Ks',
                'Obs.Flux_ACS_WFC.F606W', 'Obs.Flux_ACS_WFC.F814W',
                'Obs.Flux_APASS.B', 'Obs.Flux_APASS.V',
                'Obs.Flux_DECam.Y', 'Obs.Flux_DECam.g', 'Obs.Flux_DECam.i', 'Obs.Flux_DECam.r', 'Obs.Flux_DECam.z',
                'Obs.Flux_DENIS.I', 'Obs.Flux_DENIS.J', 'Obs.Flux_DENIS.Ks',
                'Obs.Flux_GAIA3.G', 'Obs.Flux_GAIA3.Gbp', 'Obs.Flux_GAIA3.Grp', 'Obs.Flux_GAIA3.Grvs',
                'Obs.Flux_GALEX.FUV', 'Obs.Flux_GALEX.NUV',
                'Obs.Flux_Johnson.B', 'Obs.Flux_Johnson.I', 'Obs.Flux_Johnson.R', 'Obs.Flux_Johnson.U', 'Obs.Flux_Johnson.V',
                'Obs.Flux_PS1.g', 'Obs.Flux_PS1.i', 'Obs.Flux_PS1.r', 'Obs.Flux_PS1.y', 'Obs.Flux_PS1.z',
                'Obs.Flux_SDSS.g', 'Obs.Flux_SDSS.i', 'Obs.Flux_SDSS.r', 'Obs.Flux_SDSS.u', 'Obs.Flux_SDSS.z',
                'Obs.Flux_UKIDSS.K',
                'Obs.Flux_VISTA.H', 'Obs.Flux_VISTA.J', 'Obs.Flux_VISTA.Ks', 'Obs.Flux_VISTA.Y',
                'Obs.Flux_WISE.W1', 'Obs.Flux_WISE.W2', 'Obs.Flux_WISE.W3']


## filter zero points, effective wavelengths, column names

## f = f0 * 10^(-0.4*(m-m0)) (erg/s/cm^2/AA) 

## wise_f0 = (np.array([309.540,171.787,31.674]) * u.Jy.to(u.erg/u.s/u.cm**2/u.Hz) * u.erg/u.s/u.cm**2/u.Hz * c.cgs / (np.array([33526,46030,115608]) * u.AA)**2).to(u.erg/u.s/u.cm**2/u.AA)

## gaia_f0 = np.array([3.009167e-21,1.346109e-21,1.638483e-21]) * (u.W/u.m**2/u.nm).to(u.erg/u.s/u.cm**2/u.AA)

# two_mass_f0 = np.array([3.129e-13,1.133e-13,4.283e-14]) * (u.W / u.cm**2 / u.micron).to(u.erg/u.s/u.cm**2/u.AA)


twomass_zp_table = Table({'band': ['2MASS.J','2MASS.H', '2MASS.Ks'],
                        'col':['Jmag','Hmag','Kmag'],
                        'err_col':['e_Jmag','e_Hmag','e_Kmag'],
                        'lambda_eff':[12350,16620,21590],
                        'width':[1620,2510,2640],
                        'f0':[3.129e-10,1.133e-10,4.283e-11],
                        'm0':[0,0,0],
                        'wd_f0':[3.106e-10,1.143e-10,4.206e-11], ## bedard f0 different than VOSA
                        'wd_lambda_eff':[12350,16620,21590]}) 

wise_zp_table = Table({'band': ['WISE.W1','WISE.W2','WISE.W3'],
                       'col':['W1mag','W2mag','W3mag'],
                          'err_col':['e_W1mag','e_W2mag','e_W3mag'],
                          'lambda_eff':[33526,46028,115608],
                          'width':[6620,10422,55055],
                                'f0':[8.18e-12,2.42e-12,6.52e-14],
                                'm0':[0,0,0],
                                'wd_f0':[8.18e-12,2.42e-12,6.52e-14],
                                'wd_lambda_eff':[33526,46028,115608]}) ## bedard doesn't give WISE f0

gaia_zp_table = Table({'band': ['GAIA3.Gbp','GAIA3.G','GAIA3.Grp'],
                       'col':['phot_bp_mean_mag','phot_g_mean_mag','phot_rp_mean_mag'],
                       'err_col':['phot_bp_mean_flux_error','phot_g_mean_flux_error','phot_rp_mean_flux_error'],
                       'lambda_eff':[5035,5822,7619],
                       'width':[2333,4203,2842],
                       'f0':[4.08e-9,2.5e-9,1.27e-9],
                       'm0':[0,0,0],
                       'wd_f0':[4.08e-9,2.5e-9,1.27e-9], ## bedard doesn't give GAIA f0
                       'wd_lambda_eff':[5035,5822,7619]})

galex_zp_table = Table({'band': ['GALEX.FUV', 'GALEX.NUV'],
                       'col':['fuv_mag','nuv_mag'],
                       'err_col':['fuv_magerr','nuv_magerr'],
                       'lambda_eff':[1548,2303],
                           'width':[265,770],
                       'f0':[4.6e-8,2.05e-8],
                       'm0':[0,0],
                       'wd_f0':[4.6e-8,2.05e-8], ## bedard doesn't give GALEX f0- its AB anyway
                       'wd_lambda_eff':[1548,2303]})

synt_zp_table = Table({'band': ['Johnson.U', 'Johnson.B','Johnson.V','Johnson.R','Johnson.I',
                             'SDSS.u','SDSS.g','SDSS.r','SDSS.i','SDSS.z'],
                     'lambda_eff':[3551.05,4369.53,5467.57,6695.83,8568.89,3608.04,4671.78,6141.12,7457.89,8922.78],
                     'width':[657,972,889,2070,2316,541,1064,1055,1102,1164],
                    'col':['u_jkc_flux','b_jkc_flux','v_jkc_flux','r_jkc_flux','i_jkc_flux',
                           'u_sdss_flux','g_sdss_flux','r_sdss_flux','i_sdss_flux','z_sdss_flux'],
                  'f0':[3.49719e-9,6.72553e-9,3.5833e-9,1.87529e-9,9.23651e-10,
                        3.75079e-9,5.45476e-9,2.49767e-9,1.38589e-9,8.38585e-10],
                        'm0':[0,0,0,0,0,0,0,0,0,0],
                        'wd_f0':[3.684e-9,6.548e-9,3.804e-9,2.274e-9,1.119e-9,
                                 1.1436e-8,4.9894e-9,2.8638e-9,1.9216e-9,1.3343e-9],
                        'wd_lambda_eff':[3971,4491,5423,6441,8071,
                                         3146,4670,6156,7471,8918]}) ## bedard f0 different than VOSA
sdss_zp_table = Table({'band': ['SDSS.u','SDSS.g','SDSS.r','SDSS.i','SDSS.z'],
                      'lambda_eff':[3608.04,4671.78,6141.12,7457.89,8922.78],
                      'width':[541,1064,1055,1102,1164],
                      'col':['u_sdss_flux','g_sdss_flux','r_sdss_flux','i_sdss_flux','z_sdss_flux'],
                      'f0':[3.75079e-9,5.45476e-9,2.49767e-9,1.38589e-9,8.38585e-10],
                      'm0':[0,0,0,0,0],
                      'wd_f0':[1.1436e-8,4.9894e-9,2.8638e-9,1.9216e-9,1.3343e-9],
                      'wd_lambda_eff':[3146,4670,6156,7471,8918]})
jkc_zp_table = Table({'band': ['Johnson.U', 'Johnson.B','Johnson.V','Johnson.R','Johnson.I'],
                      'lambda_eff':[3551.05,4369.53,5467.57,6695.83,8568.89],
                      'width':[657,972,889,2070,2316],
                      'col':['u_jkc_flux','b_jkc_flux','v_jkc_flux','r_jkc_flux','i_jkc_flux'],
                    'f0':[3.49719e-9,6.72553e-9,3.5833e-9,1.87529e-9,9.23651e-10],
                          'm0':[0,0,0,0,0],
                          'wd_f0':[3.684e-9,6.548e-9,3.804e-9,2.274e-9,1.119e-9],
                          'wd_lambda_eff':[3971,4491,5423,6441,8071]})

############# to get SDSS measurements #############
# synt_zp_table = jkc_zp_table # to ignore SDSS
# synt_zp_table = sdss_zp_table

bands_table = Table({'band': list(gaia_zp_table['band'])+list(wise_zp_table['band'])+list(twomass_zp_table['band'])
                     +list(synt_zp_table['band']) + list(galex_zp_table['band']),
                     'lambda_eff': list(gaia_zp_table['lambda_eff'])+list(wise_zp_table['lambda_eff'])+list(twomass_zp_table['lambda_eff'])
                     +list(synt_zp_table['lambda_eff']) + list(galex_zp_table['lambda_eff']),
                     'f0': list(gaia_zp_table['f0'])+list(wise_zp_table['f0'])+list(twomass_zp_table['f0'])
                     + list(synt_zp_table['f0']) + list(galex_zp_table['f0']),
                     'm0': list(gaia_zp_table['m0'])+list(wise_zp_table['m0'])+list(twomass_zp_table['m0']) + list(synt_zp_table['m0']) + list(galex_zp_table['m0']),
                     'wd_f0': list(gaia_zp_table['wd_f0'])+list(wise_zp_table['wd_f0'])+list(twomass_zp_table['wd_f0']) + list(synt_zp_table['wd_f0']) + list(galex_zp_table['wd_f0']),
                     'wd_lambda_eff': list(gaia_zp_table['wd_lambda_eff'])+list(wise_zp_table['wd_lambda_eff'])
                     +list(twomass_zp_table['wd_lambda_eff']) + list(synt_zp_table['wd_lambda_eff']) + list(galex_zp_table['wd_lambda_eff']),
                     'width': list(gaia_zp_table['width'])+list(wise_zp_table['width'])+list(twomass_zp_table['width'])
                       + list(synt_zp_table['width']) + list(galex_zp_table['width'])})

Table.sort(bands_table, 'lambda_eff')

bands_table.add_column(['FUV','NUV','U','u','B','g','G3_BP','V','G3','r','R','i','G3_RP','I','z','J','H','Ks','W1','W2','W3'],name='wd_band')
# bands_table.add_column(['FUV','NUV','U','B','G3_BP','V','G3','R','G3_RP','I','J','H','Ks','W1','W2','W3'],name='wd_band')
# bands_table.add_column(['FUV','NUV','u','g','G3_BP','G3','r','i','G3_RP','z','J','H','Ks','W1','W2','W3'],name='wd_band')


def get_bands_table():
    """
    Retrieve the bands table.

    Returns:
        DataFrame: A DataFrame containing the bands table.
    """
    return bands_table


def get_synthetic_photometry(source_table):
    """
    Query Gaia synthetic photometry for SDSS ugriz and Johnson UBVRI, convert to erg/s/cm^2/AA.

    Parameters:
    source_table (Table): Table containing source information with 'source_id' column.

    Returns:
    Table: Table with synthetic photometry data converted to physical units.
    """
    # Extract source IDs from the source table
    id_lst = tuple(map(int, source_table['source_id']))

    # Construct the query columns
    cols_to_query = ','.join(['source_id'] + [c + ',' + c + '_error' + ',' + c.replace('flux', 'flag') for c in synt_zp_table['col']])
    
    # Construct the query string
    query = 'SELECT ' + cols_to_query + f' FROM gaiadr3.synthetic_photometry_gspc WHERE source_id IN {id_lst}'
    
    # Launch the query job
    job = Gaia.launch_job(query)
    result = job.get_results()
    
    # Process the results
    for col in synt_zp_table['col']:
        band = synt_zp_table[synt_zp_table['col'] == col]['band'][0]
        err_col = col + '_error'
        flag_col = col.replace('flux', 'flag')
        
        # Convert Johnson fluxes from W/s/m^2/nm to erg/s/cm^2/AA
        if band.startswith('Johnson'):
            result[col] = result[col].to(u.erg / u.s / u.cm**2 / u.AA)
            result[err_col] = result[err_col].to(u.erg / u.s / u.cm**2 / u.AA)
        
        # Convert SDSS fluxes from W/s/m^2/Hz to erg/s/cm^2/AA
        elif band.startswith('SDSS'):
            wl = synt_zp_table[synt_zp_table['col'] == col]['lambda_eff'][0] * u.AA
            result[col] = (result[col].data * u.W / u.m**2 / u.Hz * c / wl.si**2).to(u.erg / u.s / u.cm**2 / u.AA)
            result[err_col] = (result[err_col].data * u.W / u.m**2 / u.Hz * c / wl.si**2).to(u.erg / u.s / u.cm**2 / u.AA)
        
        # Rename columns to band names
        result[col].name = band
        result[err_col].name = band + '_err'
    
    # Rename 'SOURCE_ID' column to 'source_id' if present
    if 'SOURCE_ID' in result.colnames:
        result.rename_column('SOURCE_ID', 'source_id')
    
    # Keep only relevant columns
    result.keep_columns(['source_id'] + list(synt_zp_table['band']) + [band + '_err' for band in synt_zp_table['band']])
    
    # Handle case where no results are returned
    if len(result) == 0:
        sid = source_table['source_id'][0]
        result.add_row([sid if col == 'source_id' else np.nan for col in result.colnames])
    
    return result



def get_gaia_photometry(source_table):
    """
    Query Gaia photometry and convert to physical units.

    Parameters:
    source_table (Table): Table containing source information with 'source_id' column.

    Returns:
    Table: Table with Gaia photometry data converted to physical units.
    """
    # Extract source IDs from the source table
    id_lst = tuple(map(int, source_table['source_id']))

    # Construct the query columns
    cols_to_query = ','.join(['source_id'] + [c for c in gaia_zp_table['col']] + 
                             [c.replace('mag', 'flux') + ',' + c.replace('mag', 'flux') + '_error' for c in gaia_zp_table['col']])
    
    # Construct the query string
    query = 'SELECT ' + cols_to_query + f' FROM gaiadr3.gaia_source WHERE source_id IN {id_lst}'
    
    # Launch the query job
    job = Gaia.launch_job(query)
    result = job.get_results()
    
    # Process the results
    for col in gaia_zp_table['col']:
        band = gaia_zp_table[gaia_zp_table['col'] == col]['band'][0]
        err_col = gaia_zp_table[gaia_zp_table['col'] == col]['err_col'][0]
        f0 = gaia_zp_table[gaia_zp_table['col'] == col]['f0'][0]
        m0 = gaia_zp_table[gaia_zp_table['col'] == col]['m0'][0]
        
        # Calculate fractional error
        err_over_f = result[col.replace('mag', 'flux') + '_error'] / result[col.replace('mag', 'flux')]
        
        # Convert magnitudes to fluxes
        mag = result[col]
        result[col] = f0 * 10**(-0.4 * (mag - m0))
        result[col].unit = u.erg / u.s / u.cm**2 / u.AA
        result[col].name = band
        
        # Convert fractional error to absolute error in physical units
        result[err_col] = result[band] * err_over_f
        result[err_col].unit = u.erg / u.s / u.cm**2 / u.AA
        result[err_col].name = band + '_err'
    
    # Rename 'SOURCE_ID' column to 'source_id' if present
    if 'SOURCE_ID' in result.colnames:
        result.rename_column('SOURCE_ID', 'source_id')
    
    # Keep only relevant columns
    result.keep_columns(['source_id'] + list(gaia_zp_table['band']) + [band + '_err' for band in gaia_zp_table['band']])
    
    return result



def get_wise_photometry(source_table):
    """
    Query WISE photometry and convert to physical units.

    Parameters:
    source_table (Table): Table containing source information with 'ra' and 'dec' columns.

    Returns:
    Table: Table with WISE photometry data converted to physical units.
    """
    coords = SkyCoord(source_table['ra'], source_table['dec'], unit=(u.deg, u.deg), frame='icrs')
    wise = Vizier.query_region(coords, radius=2 * u.arcsec, catalog=['II/311'])
    
    if len(wise) == 0:
        wise = Table({'idx': source_table['idx'],
                      'W1mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                      'W2mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                      'W3mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                      'e_W1mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                      'e_W2mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                      'e_W3mag': np.full_like(source_table['idx'], np.nan, dtype=float)})
    else:
        wise = wise[0]
        wise['_q'] = wise['_q'] - 1
        wise['_q'] = source_table['idx'][wise['_q']]
        wise['_q'].name = 'idx'
    
    wise.keep_columns(['idx'] + list(wise_zp_table['col']) + list(wise_zp_table['err_col']))
    
    for col in wise_zp_table['col']:
        band = wise_zp_table[wise_zp_table['col'] == col]['band'][0]
        err_col = wise_zp_table[wise_zp_table['col'] == col]['err_col'][0]
        f0 = wise_zp_table[wise_zp_table['col'] == col]['f0'][0]
        m0 = wise_zp_table[wise_zp_table['col'] == col]['m0'][0]
        
        wise[col] = f0 * 10**(-0.4 * (wise[col] - m0))
        wise[col].unit = u.erg / u.s / u.cm**2 / u.AA
        wise[col].name = band
        
        wise[err_col] = wise[band] * 0.4 * np.log(10) * wise[err_col]
        wise[err_col].unit = u.erg / u.s / u.cm**2 / u.AA
        wise[err_col].name = band + '_err'
    
    for band in wise_zp_table['band']:
        for i in range(len(wise)):
            if (wise[i][band] / wise[i][band + '_err'] < 3) or np.ma.is_masked(wise[i][band + '_err']):
                wise[i][band] = np.nan
                wise[i][band + '_err'] = np.nan
    
    for idx in source_table['idx']:
        if idx not in wise['idx']:
            wise.add_row([idx] + [np.nan for col in wise_zp_table['col']] + [np.nan for err_col in wise_zp_table['err_col']])
    
    wise.sort('idx')
    return wise



def get_twomass_photometry(source_table):
    """
    Query 2MASS photometry and convert to physical units.

    Parameters:
    source_table (Table): Table containing source information with 'ra' and 'dec' columns.

    Returns:
    Table: Table with 2MASS photometry data converted to physical units.
    """
    coords = SkyCoord(source_table['ra'], source_table['dec'], unit=(u.deg, u.deg), frame='icrs')
    twomass = Vizier.query_region(coords, radius=2 * u.arcsec, catalog=['II/246'])

    if len(twomass) == 0:
        twomass = Table({'idx': source_table['idx'],
                         'Jmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'Hmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'Kmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'e_Jmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'e_Hmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'e_Kmag': np.full_like(source_table['idx'], np.nan, dtype=float),
                         'Qflg': np.full_like(source_table['idx'], 'ZZZ', dtype='<U3')})
        quality_flag = twomass['Qflg']
    else:
        twomass = twomass[0]
        twomass['_q'] = twomass['_q'] - 1
        twomass['_q'] = source_table['idx'][twomass['_q']]
        twomass['_q'].name = 'idx'
        quality_flag = twomass['Qflg']

    twomass.keep_columns(['idx'] + list(twomass_zp_table['col']) + list(twomass_zp_table['err_col']))

    for col in twomass_zp_table['col']:
        band = twomass_zp_table[twomass_zp_table['col'] == col]['band'][0]
        err_col = twomass_zp_table[twomass_zp_table['col'] == col]['err_col'][0]
        f0 = twomass_zp_table[twomass_zp_table['col'] == col]['f0'][0]
        m0 = twomass_zp_table[twomass_zp_table['col'] == col]['m0'][0]

        twomass[col] = f0 * 10**(-0.4 * (twomass[col] - m0))
        twomass[col].unit = u.erg / u.s / u.cm**2 / u.AA
        twomass[col].name = band

        twomass[err_col] = twomass[band] * 0.4 * np.log(10) * twomass[err_col]
        twomass[err_col].unit = u.erg / u.s / u.cm**2 / u.AA
        twomass[err_col].name = band + '_err'

    for i in range(len(twomass)):
        for j in [0, 1, 2]:  # J, H, Ks bands respectively
            if quality_flag[i][j] != 'A':  # mask out sources according to 2MASS quality flag
                twomass[i][twomass_zp_table['band'][j]] = np.nan
                twomass[i][twomass_zp_table['band'][j] + '_err'] = np.nan

    for idx in source_table['idx']:
        if idx not in twomass['idx']:
            twomass.add_row([idx] + [np.nan for col in twomass_zp_table['col']] + [np.nan for err_col in twomass_zp_table['err_col']])

    twomass.sort('idx')
    return twomass



def get_galex_photometry(source_table):
    """
    Query GALEX photometry and convert to physical units.

    Parameters:
    source_table (Table): Table containing source information with 'ra' and 'dec' columns.
 
    Returns:
    Table: Table with GALEX photometry data converted to physical units.
    """
    coords = SkyCoord(source_table['ra'], source_table['dec'], unit=(u.deg, u.deg), frame='icrs')
    galex = Table({key: np.full_like(source_table['idx'], np.nan, float) for key in list(galex_zp_table['col']) + list(galex_zp_table['err_col'])})
    galex['idx'] = source_table['idx']
    
    galex_dr7 = Vizier.query_region(coords, radius=2 * u.arcsec, catalog=['II/335/galex_ais'])
    if len(galex_dr7) == 0:
        galex_dr7 = Table({'idx': source_table['idx'],
                           'nuv_mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                           'nuv_magerr': np.full_like(source_table['idx'], np.nan, dtype=float),
                           'fuv_mag': np.full_like(source_table['idx'], np.nan, dtype=float),
                           'fuv_magerr': np.full_like(source_table['idx'], np.nan, dtype=float)})
    else:
        galex_dr7 = galex_dr7[0]
        galex_dr7['NUVmag'].name = 'nuv_mag'
        galex_dr7['e_NUVmag'].name = 'nuv_magerr'
        galex_dr7['FUVmag'].name = 'fuv_mag'
        galex_dr7['e_FUVmag'].name = 'fuv_magerr'
        galex_dr7['_q'] = galex_dr7['_q'] - 1
        galex_dr7['_q'] = source_table['idx'][galex_dr7['_q']]
        galex_dr7['_q'].name = 'idx'

    for i in range(len(coords)):
        data = Catalogs.query_region(coords[i], radius=2 * u.arcsec, catalog='Galex')
        if len(data) == 0:
            continue
        for col in galex_zp_table['col']:
            if np.ma.is_masked(data[col][0]):
                continue
            err_col = galex_zp_table[galex_zp_table['col'] == col]['err_col'][0]
            f0 = galex_zp_table[galex_zp_table['col'] == col]['f0'][0]
            m0 = galex_zp_table[galex_zp_table['col'] == col]['m0'][0]
            galex[i][col] = f0 * 10**(-0.4 * (data[col][0] - m0))
            galex[i][err_col] = galex[i][col] * 0.4 * np.log(10) * data[err_col][0]
            if galex[i]['idx'] in galex_dr7['idx']:
                j = np.where(galex_dr7['idx'] == galex[i]['idx'])[0][0]
                if not np.ma.is_masked(galex_dr7[j][col]) and not np.isnan(galex_dr7[j][col]):
                    galex[i][col] = f0 * 10**(-0.4 * (galex_dr7[j][col] - m0))
                    galex[i][err_col] = galex[i][col] * 0.4 * np.log(10) * galex_dr7[j][err_col]

    for col in galex_zp_table['col']:
        band = galex_zp_table[galex_zp_table['col'] == col]['band'][0]
        err_col = galex_zp_table[galex_zp_table['col'] == col]['err_col'][0]
        galex[col].unit = u.erg / u.s / u.cm**2 / u.AA
        galex[col].name = band
        galex[err_col].unit = u.erg / u.s / u.cm**2 / u.AA
        galex[err_col].name = band + '_err'
    
    return galex
        


def get_photometry(source_table, snr_lim=10):
    """
    Query photometry from multiple catalogs (Gaia, synthetic, GALEX, 2MASS, WISE) and convert to physical units.

    Parameters:
    source_table (Table): Table containing source information with 'source_id', 'ra', 'dec', 'parallax', 'parallax_error', '[Fe/H]', and 'Av' columns.
    snr_lim (float): Signal-to-noise ratio limit. For all fluxes with reported SNR > snr_lim, the error is set to 10% of the flux.

    Returns:
    Table: Table with photometry data from multiple catalogs converted to physical units.
    """
    if 'idx' not in source_table.colnames:
        source_table['idx'] = np.arange(len(source_table))
    
    tbl = source_table['idx', 'source_id', 'ra', 'dec', 'parallax', 'parallax_error', '[Fe/H]', 'Av']
    tbl = table.join(tbl, get_gaia_photometry(source_table, gaia_zp_table), keys='source_id', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_synthetic_photometry(tbl, synt_zp_table), keys='source_id', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_galex_photometry(tbl, galex_zp_table), keys='idx', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_twomass_photometry(tbl, twomass_zp_table), keys='idx', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_wise_photometry(tbl, wise_zp_table), keys='idx', join_type='left', metadata_conflicts='silent')

    flux_cols = list(gaia_zp_table['band']) + list(wise_zp_table['band']) + list(twomass_zp_table['band']) + list(synt_zp_table['band']) + list(galex_zp_table['band'])
    
    if snr_lim is not None:
        for col in flux_cols:
            snr = tbl[col] / tbl[col + '_err']
            for i in range(len(tbl)):
                if snr[i] > snr_lim:  # if SNR > snr_lim, set error to 10% of flux: minimal error to account for model uncertainties
                    tbl[i][col + '_err'] = 0.1 * tbl[i][col]

    tbl.sort('idx')
    return tbl



def get_photometry_single_source(source_id, snr_lim=10):
    """
    Query photometry for a single source from multiple catalogs (Gaia, synthetic, GALEX, 2MASS, WISE) and convert to physical units.

    Parameters:
    source_id (int): Gaia DR3 source ID.
    snr_lim (float): Signal-to-noise ratio limit. For all fluxes with reported SNR > snr_lim, the error is set to 10% of the flux.

    Returns:
    Table: Table with photometry data from multiple catalogs converted to physical units for the single source.
    For all fluxes with reported SNR > snr_lim, the error is set to 10% of the flux.
    """
    query = f'''SELECT source_id, ra, dec, parallax, parallax_error, ag_gspphot 
                FROM gaiadr3.gaia_source 
                WHERE source_id = {source_id}'''
    job = Gaia.launch_job(query)
    result = job.get_results()

    tbl = Table({'source_id': [source_id],})

    # Add Gaia data to the table, including rough estimate for extinction
    tbl['idx'] = 0
    tbl['ra'] = result['ra'][0]
    tbl['dec'] = result['dec'][0]
    tbl['parallax'] = result['parallax'][0]
    tbl['parallax_error'] = result['parallax_error'][0]
    tbl['Av'] = result['ag_gspphot'][0]

    
    tbl.add_row(tbl[0])
    tbl = table.join(tbl, get_gaia_photometry(tbl), keys='source_id', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_synthetic_photometry(tbl), keys='source_id', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_galex_photometry(tbl), keys='idx', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_twomass_photometry(tbl), keys='idx', join_type='left', metadata_conflicts='silent')
    tbl = table.join(tbl, get_wise_photometry(tbl), keys='idx', join_type='left', metadata_conflicts='silent')

    flux_cols = list(gaia_zp_table['band']) + list(wise_zp_table['band']) + list(twomass_zp_table['band']) + list(synt_zp_table['band']) + list(galex_zp_table['band'])
    
    if snr_lim is not None:
        for col in flux_cols:
            snr = tbl[col] / tbl[col + '_err']
            for i in range(len(tbl)):
                if snr[i] > 10:  # if SNR > 10, set error to 10% of flux: minimal error to account for model uncertainties
                    tbl[i][col + '_err'] = 0.1 * tbl[i][col]

    band_status = {band: ('ok' if np.isfinite(tbl[band][0]) else 'no_data') for band in flux_cols}

    return Table(tbl[0]), band_status