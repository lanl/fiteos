""" This module contains functions for plotting fits.
"""

import matplotlib.pyplot as plt
import numpy

def plot_fit(nfig, x_fit, y_fit, x, y, sigma_x, sigma_y,
             cp_band_low, cp_band_1, xlabel="Pressure (GPa)",
             ylabel="Volume ($\mathbf{\AA^3}$)", output_file=None):
    """ Plots fit.

        Parameters
        ----------
        nfig : int
        x_fit, y_fit : array_like
        x, y : array_like
        sigma_x, sigma_y : array_like
        cp_band_low, cp_band_high : array_like
        xlabel : str, optional
        ylabel : str, optional
        output_file : str or None, optional
    """

    plt.figure(nfig)

    plt.plot(x_fit, y_fit, "b")
    plt.errorbar(x, y, fmt="o", xerr=sigma_x, yerr=sigma_y,
                 alpha=1.0, capsize = 5, fillstyle="none")

    plt.plot(cp_band_low, y_fit, linestyle="--", linewidth=0.75, color="r")
    plt.plot(cp_band_1, y_fit, linestyle="--", linewidth=0.75, color="r")

    plt.xlim(numpy.min(x_fit), numpy.max(x_fit))
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")

    plt.tick_params(direction="in", bottom=1, top=1, left=1, right=1)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=1800, bbox_inches="tight")
        plt.close()

