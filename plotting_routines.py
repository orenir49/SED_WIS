import numpy as np
import matplotlib.pyplot as plt
import warnings

# This is our band retrieval routines
import band_retrieval_routines as brr
# This is our extinction routines
import extinction_routines as extinction_routines
# This is our SED fitting routines
import sed_routines as sr

warnings.simplefilter('ignore', UserWarning)

def plot_data_vs_model(x,y,yerr,mask,x_model,y_model,ax,lbl):
    """
    Plots the observed spectral energy distribution (SED) alongside the best-fit model, 
    distinguishing between included and ignored bands.

    Parameters:
    -----------
    x : np.ndarray
        Wavelengths of observed data.
    y : np.ndarray
        Observed flux values.
    yerr : np.ndarray
        Errors in observed flux values.
    mask : np.ndarray[bool]
        Boolean mask indicating which bands were used in the fit.
    x_model : np.ndarray
        Wavelengths of the model SED.
    y_model : np.ndarray
        Model SED flux values.
    ax : matplotlib.axes.Axes
        Axis to plot on.
    lbl : str
        Label for the model fit.

    Returns:
    --------
    None
    """
    x_fit = x[mask]
    y_fit = y[mask]
    yerr_fit = yerr[mask]
    x_ignored = x[~mask]
    y_ignored = y[~mask]
    yerr_ignored = yerr[~mask]

    ax.plot(x_model,y_model,color='Navy',ls='--',lw=1,label=lbl,zorder=5)
    
    ## displaying observed SED
    ax.scatter(x_fit,y_fit,marker='.',color='Crimson',s=60,zorder = 10,label=f'Used in fit')
    ax.scatter(x_ignored,y_ignored,marker='.',color='k',s=60,zorder = 10,label='Excluded from fit')
    ax.legend(fontsize=9,loc='lower center',frameon=False)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\lambda f_\lambda $ (erg$\,$s$^{-1}\,$cm$^{-2}$)',labelpad=-1.2,fontsize=12) 
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.minorticks_on()

def plot_residuals(x,y,yerr,mask,x_model,y_model,ax):
    """
    Plots the residuals between observed and model fluxes.

    Parameters:
    -----------
    x : np.ndarray
        Wavelengths of observed data.
    y : np.ndarray
        Observed flux values.
    yerr : np.ndarray
        Errors in observed flux values.
    mask : np.ndarray[bool]
        Boolean mask indicating which bands were used in the fit.
    x_model : np.ndarray
        Wavelengths of the model SED.
    y_model : np.ndarray
        Model SED flux values.
    ax : matplotlib.axes.Axes
        Axis to plot on.

    Returns:
    --------
    None
    """
    x_fit = x[mask]
    y_fit = y[mask]
    yerr_fit = yerr[mask]
    x_ignored = x[~mask]
    y_ignored = y[~mask]
    yerr_ignored = yerr[~mask]

    residuals = (y_fit - y_model[mask]) / y_model[mask]
    residuals_error = yerr_fit / y_model[mask]

    residuals_ignored = (y_ignored - y_model[~mask]) / y_model[~mask]
    residuals_error_ignored = yerr_ignored / y_model[~mask]    

    ax.errorbar(x_fit,residuals,yerr = residuals_error,fmt='.',label='observed',ecolor='k',elinewidth=1,mec='Crimson',mfc='Crimson',capsize=0,markersize=7)
    ax.errorbar(x_ignored,residuals_ignored,yerr = residuals_error_ignored,fmt='.',label='ignored',ecolor='k',elinewidth=1,mec='k',mfc='k',capsize=0,markersize=7)
    ax.set_ylabel(r'Residuals',fontsize=10,labelpad=0)
    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.axhline(0,color='Navy',ls='--',lw=1,label='model')

def create_model_label(teff1,teff1_err,r1,r1_err,chi2):
    """
    Generates a formatted label describing the best-fit model parameters.

    Parameters:
    -----------
    teff1 : float
        Best-fit effective temperature.
    teff1_err : float
        Uncertainty in effective temperature.
    r1 : float
        Best-fit radius.
    r1_err : float
        Uncertainty in radius.
    chi2 : float
        Reduced chi-square value of the fit.

    Returns:
    --------
    lbl : str
        Formatted label for the best-fit model.
    """
    sig_dig_t = -int(np.floor(np.log10(teff1_err)))
    sig_dig_r = -int(np.floor(np.log10(r1_err)))
    lbl = f'Best-fit MS model, $\chi^2/ndof$= {chi2:.2f} \n' 
    lbl += r'$T_{\text{eff},1}=$' 
    lbl += f'{int(np.round(teff1,sig_dig_t))}$\pm${int(np.round(teff1_err,sig_dig_t))} K, '
    lbl += '$R_1=$' + f'{np.round(r1,sig_dig_r)}$\pm${np.round(r1_err,sig_dig_r)} $R_\odot$'
    return lbl

def plot_kurucz_fit(source_id,m1,meta,av,fit_results,parallax=None,bands_to_ignore=[],plot=True,save=False):
    """
    Fits a stellar model to observed photometry and plots the best-fit SED along with residuals.

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
    fit_results : tuple
        Fit results (Teff, Teff_err, R, R_err, redchi2).
    parallax: float, optional
        Parallax in milliarcsec. If not provided, use value from gaia main table.
    bands_to_ignore : list, optional
        List of bands to exclude from fitting. Default is [].
        Options: ['GALEX.FUV' 'GALEX.NUV' 'Johnson.U' 'SDSS.u' 'Johnson.B' 'SDSS.g'
                    'GAIA3.Gbp' 'Johnson.V' 'GAIA3.G' 'SDSS.r' 'Johnson.R' 'SDSS.i'
                        'GAIA3.Grp' 'Johnson.I' 'SDSS.z' '2MASS.J' '2MASS.H' '2MASS.Ks' 'WISE.W1'
                            'WISE.W2' 'WISE.W3']
    plot : bool, optional
        If True, displays the plot. Default is True.
    save : bool, optional
        If True, saves the plot as an image. Default is False.

    Returns:
    --------
    None
    """
    # Get the observed data
    obs_tbl,flags = brr.get_photometry_single_source(source_id)
    bands_table = brr.get_bands_table()

    # Organize the observed data
    bnds = list(bands_table['band'])
    wl = np.array(list(bands_table['lambda_eff']))
    flux = np.array([obs_tbl[0][bnd] for bnd in bnds])
    flux = flux * wl
    flux_err = np.array([obs_tbl[0][bnd + '_err'] for bnd in bnds])
    flux_err = flux_err * wl

    # Continue organizing the observed data- separate points used in fit from those ignored
    mask = np.ones(len(bnds), dtype=bool)
    for band in bands_to_ignore:
        mask[bnds.index(band)] = False

    # All model parameters
    teff_fit, teff_err, r_fit, r_err, redchi2 = fit_results
    if parallax is None:
        parallax = obs_tbl[0]['parallax']
    else:
        parallax = parallax
    m1 = m1
    meta = meta
    av = av
    best_fit_label = create_model_label(teff_fit,teff_err,r_fit,r_err,redchi2)

    # Get the best fit model
    ms = sr.get_MS_sed(teff_fit,m1,r_fit,meta,parallax)
    ms = sr.redden_model_table(ms, teff_fit, av, bands_table=bands_table)
    flux_model = np.array([ms[bnd][0] for bnd in bands_table['wd_band']]) * wl

    # define the figure
    fig = plt.figure(figsize=(1.5*3.5,3.5),tight_layout=True, dpi=400)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
    ax = gs.subplots(sharex=True)

    # pass axes to plotting routines (best fit vs data, residuals)
    plot_data_vs_model(wl,flux,flux_err,mask,wl,flux_model,ax[0],best_fit_label)
    plot_residuals(wl,flux,flux_err,mask,wl,flux_model,ax[1])

    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'Wavelength $(\AA)$',fontsize=12)
    ax[1].tick_params(axis='x', which='major', labelsize=10)       
    fig.show()
    if save:
        fig.savefig(f'../img/kurucz_{source_id}.png')
    if not plot:
        plt.close(fig)
