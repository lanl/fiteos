""" This module contains functions for plotting strain.
"""

import uncertainties as uct
from lmfit import models
import matplotlib.pyplot as plt
import numpy

def plot_ff(nfig, x, y, sigma_x, sigma_y, y0, sigma_y0, pcov, output_file=None):
    """ Plots strain.

        Parameters
        ----------
        nfig : int
        x, y : array_like
        sigma_x, sigma_y : array_like
        y0 : float
        simga_y0 : array_like
        pov : numpy.array
        output_file : str or None, optional
    """

    # compute fF
    f = (((y0 / y)**(2.0 / 3.0)) - 1.0) / 2.0
    F = x / (3.0 * f * (1.0 + (2.0 * f))**(5.0 / 2.0))
    sigma_vo = sigma_y0
    eta = y / y0
    sigeta = numpy.abs(eta) * ((((sigma_y / y)**2.0) + ((sigma_vo / y0)**2))**(1.0 / 2.0))
    sigprime = ((7.0 * (eta**(-2.0 / 3.0)) - 5.0) * sigeta)/(2.0 * (1.0 - (eta**-2.0 / 3.0)) * eta)
    sigF = F * numpy.sqrt(((sigma_x / x)**2.0) + (sigprime**2))
    
    # fit line to fF
    line_mod = models.LinearModel()
    pars = line_mod.guess(f)
    outf = line_mod.fit(F, pars, x=f)
    ff_slope = uct.ufloat(outf.params["slope"], outf.params["slope"].stderr)
    ff_inter = uct.ufloat(outf.params["intercept"], outf.params["intercept"].stderr)
    
    k_p = ((2.0 * ff_slope) / (3 * ff_inter)) + 4
    
    # compute confidence ellipses
    n_params = len(pcov[0])
    
    plt.figure(nfig)

    plt.plot(f, outf.best_fit, '-',color='black')
    plt.errorbar(f, F, fmt='ko', xerr=0, yerr=sigF, alpha=1.0,capsize = 3.)

    plt.xlabel("Eulerian strain $\mathit{f_E}$", fontweight="bold")
    plt.ylabel("Normalized pressure $\mathbf{F_E}$ (GPa)", fontweight="bold")
    plt.tick_params(direction="in", bottom=1, top=1, left=1, right=1)
    plt.title("$\mathit{f_E}$-F", fontweight="bold")

    if output_file:
        plt.savefig(output_file, dpi=1800, bbox_inches="tight")
        plt.close()

